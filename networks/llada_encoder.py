"""
networks.llada_encoder

LLaDA masked-diffusion encoder adapted for COGITAO token sequences.
The module keeps LLaDA's bidirectional transformer blocks, RoPE, stochastic
target masking, and iterative reverse-denoising generation.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import einsum


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        return F.dropout(input, self.p, self.training, self.inplace)


class BufferCache(dict):
    pass


class ModuleType:
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def _non_meta_init_device(config: DictConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(
    config: DictConfig,
    module: nn.Module,
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[str] = None,
) -> None:
    d = d if d is not None else config.embed_dim
    if config.init_fn == "normal":
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff, b=cutoff)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == "mitchell":
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == "fan_in":
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == "full_megatron":
        if type_of_module is None:
            raise RuntimeError("full_megatron init requires a module type")
        cutoff = config.init_cutoff_factor if config.init_cutoff_factor is not None else 3
        if type_of_module == ModuleType.out_module:
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.final_out:
            std = config.embed_dim**-0.5
        else:
            std = config.init_std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
    if isinstance(module, nn.Linear) and config.init_fn == "normal" and getattr(module, "_is_residual", False):
        with torch.no_grad():
            module.weight.div_(math.sqrt(2 * config.n_layers))


def activation_checkpoint_function(cfg: DictConfig):
    preserve_rng_state = (
        cfg.attention_dropout == 0.0
        and cfg.embedding_dropout == 0.0
        and cfg.residual_dropout == 0.0
    )
    from torch.utils.checkpoint import checkpoint

    return partial(checkpoint, preserve_rng_state=preserve_rng_state, use_reentrant=False)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.embed_dim,)
        if elementwise_affine or (elementwise_affine is None and config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = config.bias_for_layer_norm
            if use_bias is None:
                use_bias = config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @classmethod
    def build(cls, config: DictConfig, size: Optional[int] = None, **kwargs) -> "LayerNormBase":
        if config.layer_norm_type in ("default", "low_precision"):
            return LayerNorm(config, size=size, low_precision=config.layer_norm_type == "low_precision", **kwargs)
        if config.layer_norm_type == "rms":
            return RMSLayerNorm(config, size=size, **kwargs)
        if config.layer_norm_type == "gemma_rms":
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        raise NotImplementedError(f"Unknown LayerNorm type: {config.layer_norm_type}")

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LayerNorm(LayerNormBase):
    def __init__(self, config: DictConfig, size: Optional[int] = None, low_precision: bool = False, **kwargs):
        super().__init__(config, size=size, **kwargs)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.low_precision:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        with torch.autocast(enabled=False, device_type=x.device.type):
            return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)


class RMSLayerNorm(LayerNormBase):
    def __init__(self, config: DictConfig, size: Optional[int] = None, **kwargs):
        super().__init__(config, size=size, eps=config.rms_norm_eps, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
            y = y.to(dtype)
        if self.weight is None:
            return y
        if self.bias is None:
            return self.weight * y
        return self.weight * y + self.bias


class GemmaRMSLayerNorm(RMSLayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
            y = y.to(dtype)
        if self.weight is None:
            return y
        if self.bias is None:
            return y * (1 + self.weight)
        return y * (1 + self.weight) + self.bias


class RotaryEmbedding(nn.Module):
    def __init__(self, config: DictConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.cache = cache
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_sin = self.cache.get("rope_pos_sin")
        pos_cos = self.cache.get("rope_pos_cos")
        if pos_sin is not None and pos_cos is not None and pos_sin.shape[-2] >= seq_len:
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                pos_cos = pos_cos.to(device)
                self.cache["rope_pos_sin"] = pos_sin
                self.cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.embed_dim // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
            seq = torch.arange(seq_len, device=device).float()
            freqs = einsum("i,j->ij", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin = positions.sin()[None, None, :, :]
            pos_cos = positions.cos()[None, None, :, :]
        self.cache["rope_pos_sin"] = pos_sin
        self.cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, head_size = x.size()
        x = x.view(bsz, n_heads, seq_len, 2, head_size // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_, k_ = (q.float(), k.float()) if self.config.rope_full_precision else (q, k)
        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config: DictConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.embed_dim
        self.dropout = Dropout(config.residual_dropout)
        self._activation_checkpoint_fn = None
        self.k_norm = None
        self.q_norm = None
        if config.attention_layer_norm:
            head_dim = config.embed_dim // config.n_heads
            self.k_norm = LayerNormBase.build(
                config,
                size=head_dim * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)
        self.attn_out = nn.Linear(config.embed_dim, config.embed_dim, bias=config.include_bias, device=config.init_device)
        self.ff_out = nn.Linear(self.hidden_size, config.embed_dim, bias=config.include_bias, device=config.init_device)
        self.ff_out._is_residual = True
        if config.rope:
            self.rotary_emb = RotaryEmbedding(config, cache)
        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                self.flash_attn_func = None

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(self.config, self.attn_out, d=self.config.embed_dim, layer_id=self.layer_id, type_of_module=ModuleType.out_module)
        init_weights(self.config, self.ff_out, d=self.ff_out.in_features, layer_id=self.layer_id, type_of_module=ModuleType.out_module)

    def set_activation_checkpointing(self, strategy: Optional[str]):
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config) if strategy == "fine_grained" else None

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0) -> torch.Tensor:
        if self.flash_attn_func is not None:
            out = self.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False)
            return out.transpose(1, 2)
        if q.size(1) != k.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1, output_size=q.size(1))
            v = v.repeat_interleave(q.size(1) // v.size(1), dim=1, output_size=q.size(1))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = q.size()
        dtype = k.dtype
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)
        q = q.view(bsz, seq_len, self.config.n_heads, channels // self.config.n_heads).transpose(1, 2)
        k = k.view(bsz, seq_len, self.config.effective_n_kv_heads, channels // self.config.n_heads).transpose(1, 2)
        v = v.view(bsz, seq_len, self.config.effective_n_kv_heads, channels // self.config.n_heads).transpose(1, 2)
        if self.config.rope:
            q, k = self.rotary_emb(q, k)
        att = self._scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else self.config.attention_dropout)
        att = att.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.attn_out(att)


class LLaDALlamaBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: DictConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNormBase.build(config)
        self.ff_norm = LayerNormBase.build(config)
        head_dim = config.embed_dim // config.n_heads
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.k_proj = nn.Linear(config.embed_dim, config.effective_n_kv_heads * head_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.v_proj = nn.Linear(config.embed_dim, config.effective_n_kv_heads * head_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.ff_proj = nn.Linear(config.embed_dim, self.hidden_size, bias=config.include_bias, device=config.init_device)
        self.up_proj = nn.Linear(config.embed_dim, self.hidden_size, bias=config.include_bias, device=config.init_device)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.k_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.v_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.ff_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.up_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = self.attn_norm(x)
        att = self.attention(self.q_proj(x_normed), self.k_proj(x_normed), self.v_proj(x_normed))
        x = x + self.dropout(att)
        residual = x
        x = self.ff_norm(x)
        x = F.silu(self.ff_proj(x)) * self.up_proj(x)
        x = self.dropout(self.ff_out(x))
        return residual + x


class LLaDASequentialBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: DictConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNormBase.build(config)
        self.ff_norm = LayerNormBase.build(config)
        head_dim = config.embed_dim // config.n_heads
        self.fused_dims = (config.embed_dim, config.effective_n_kv_heads * head_dim, config.effective_n_kv_heads * head_dim)
        self.att_proj = nn.Linear(config.embed_dim, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias, device=config.init_device)
        self.ff_proj = nn.Linear(config.embed_dim, self.hidden_size, bias=config.include_bias, device=config.init_device)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.att_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.ff_proj, d=self.config.embed_dim, type_of_module=ModuleType.in_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)
        x = x + self.dropout(self.attention(q, k, v))
        residual = x
        x = self.ff_out(F.silu(self.ff_proj(self.ff_norm(x))))
        return residual + self.dropout(x)


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: DictConfig, layer_offset: int, modules: Iterable[nn.Module]):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy = None
        self._activation_checkpoint_fn = activation_checkpoint_function(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block_idx, block in enumerate(self):
            global_idx = self.layer_offset + block_idx
            if (
                self.activation_checkpointing_strategy == "whole_layer"
                or (self.activation_checkpointing_strategy == "one_in_two" and global_idx % 2 == 0)
                or (self.activation_checkpointing_strategy == "one_in_three" and global_idx % 3 == 0)
                or (self.activation_checkpointing_strategy == "one_in_four" and global_idx % 4 == 0)
            ):
                x = self._activation_checkpoint_fn(block, x)
            else:
                x = block(x)
        return x

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[str]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAEncoder(nn.Module):
    def __init__(self, cfg: DictConfig, init_params: bool = True):
        super().__init__()
        self.cfg = cfg
        self.config = cfg.network.encoder
        OmegaConf.set_struct(self.config, False)
        self.config.effective_n_kv_heads = self.effective_n_kv_heads(self.config)
        self.config.vocab_size = cfg.model.input_vocab_size
        self.config.embedding_size = cfg.model.input_vocab_size
        self.config.output_vocab_size = cfg.model.output_vocab_size
        self.config.mask_token_id = cfg.model.mask_token_id
        self.config.thinking_token_id = cfg.model.get("thinking_token_id", None)
        self.config.pad_token_id = cfg.data.pad_token_id
        OmegaConf.set_struct(self.config, True)

        self.cache = BufferCache()
        self.activation_checkpointing_strategy: Optional[str] = cfg.network.encoder.get("activation_checkpointing", None)
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

        if self.config.block_group_size <= 0 or self.config.n_layers % self.config.block_group_size != 0:
            raise ValueError("n_layers must be divisible by block_group_size")

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.config.embedding_size, self.config.embed_dim, device=self.config.init_device),
                "emb_drop": Dropout(self.config.embedding_dropout),
                "ln_f": LayerNormBase.build(self.config),
            }
        )
        self.input_embedding = self.transformer.wte

        block_cls = LLaDALlamaBlock if self.config.block_type == "llama" else LLaDASequentialBlock
        blocks = [block_cls(i, self.config, self.cache) for i in range(self.config.n_layers)]
        if self.config.block_group_size > 1:
            groups = [
                LLaDABlockGroup(self.config, i, blocks[i : i + self.config.block_group_size])
                for i in range(0, self.config.n_layers, self.config.block_group_size)
            ]
            self.transformer["block_groups"] = nn.ModuleList(groups)
        else:
            self.transformer["blocks"] = nn.ModuleList(blocks)

        if not self.config.rope:
            self.transformer["wpe"] = nn.Embedding(self.config.max_sequence_length, self.config.embed_dim, device=self.config.init_device)

        self.lm_head = nn.Linear(self.config.embed_dim, self.config.output_vocab_size, bias=self.config.include_bias, device=self.config.init_device)

        if init_params and self.config.init_device != "meta":
            self.reset_parameters()

    @staticmethod
    def effective_n_kv_heads(config: DictConfig) -> int:
        if config.n_kv_heads is None:
            return 1 if config.multi_query_attention is True else config.n_heads
        if config.multi_query_attention is None:
            return config.n_kv_heads
        expected = 1 if config.multi_query_attention else config.n_heads
        if config.n_kv_heads != expected:
            raise ValueError("n_kv_heads and multi_query_attention disagree")
        return expected

    @property
    def device(self) -> torch.device:
        device = self.transformer.wte.weight.device
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        return device

    def reset_parameters(self):
        init_weights(
            self.config,
            self.transformer.wte,
            std_factor=(0.5 * math.sqrt(self.config.embed_dim)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if "wpe" in self.transformer:
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)
        self.transformer.ln_f.reset_parameters()
        init_weights(self.config, self.lm_head, type_of_module=ModuleType.final_out)
        modules = self.transformer.blocks if "blocks" in self.transformer else self.transformer.block_groups
        for module in modules:
            module.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[str]):
        self.activation_checkpointing_strategy = strategy
        modules = self.transformer.blocks if "blocks" in self.transformer else self.transformer.block_groups
        for module in modules:
            module.set_activation_checkpointing(strategy)

    def forward_hidden(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        del attention_mask
        if not self.config.rope and input_ids.size(1) > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_sequence_length={self.config.max_sequence_length}")

        x = self.transformer.wte(input_ids.long())
        if self.config.input_emb_norm:
            x = x * (self.config.embed_dim**0.5)
        if not self.config.rope:
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = x + self.transformer.wpe(positions)
        x = self.transformer.emb_drop(x)

        all_hidden_states: List[torch.Tensor] = []
        modules = self.transformer.blocks if "blocks" in self.transformer else self.transformer.block_groups
        for block_idx, block in enumerate(modules):
            if output_hidden_states:
                all_hidden_states.append(x)
            if (
                self.activation_checkpointing_strategy == "whole_layer"
                or (self.activation_checkpointing_strategy == "one_in_two" and block_idx % 2 == 0)
                or (self.activation_checkpointing_strategy == "one_in_three" and block_idx % 3 == 0)
                or (self.activation_checkpointing_strategy == "one_in_four" and block_idx % 4 == 0)
            ):
                x = self._activation_checkpoint_fn(block, x)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)
            return x, tuple(all_hidden_states)
        return x

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        hidden = self.forward_hidden(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            hidden, all_hidden_states = hidden
            return self.lm_head(hidden), all_hidden_states
        return self.lm_head(hidden)

    def mask_input_sequence(
        self,
        target_ids: torch.LongTensor,
        eps: Optional[float] = None,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:
        eps = self.config.diffusion.get("mask_eps", 1e-3) if eps is None else eps
        batch_size, target_len = target_ids.shape
        target_for_loss = target_ids.clone()

        t = torch.rand(batch_size, device=target_ids.device)
        p_mask = ((1 - eps) * t + eps)[:, None].expand(-1, target_len)
        mask_target = (torch.rand(batch_size, target_len, device=target_ids.device) < p_mask).bool()

        keep_target = mask_target
        if self.config.diffusion.sage_thinking:
            mask_indices = mask_target.nonzero(as_tuple=True)
            num_masked = mask_indices[0].numel()
            num_keep = int(num_masked * 0.10)
            keep_target = mask_target.clone()
            if num_keep > 0:
                selected_keep = torch.randperm(num_masked, device=target_ids.device)[:num_keep]
                keep_positions = tuple(index[selected_keep] for index in mask_indices)
                # Keep these tokens visible in the noised input, but leave
                # mask_target unchanged so they still contribute to the loss.
                keep_target[keep_positions] = False
            mask_indices = keep_target.nonzero(as_tuple=True)
            num_masked = mask_indices[0].numel()

        masked_sequence = target_ids.clone()
        masked_sequence[keep_target] = self.config.mask_token_id

        if self.config.diffusion.sage_thinking:
            num_random = int(num_masked * 0.10)
            if num_random > 0:
                selected = torch.randperm(num_masked, device=target_ids.device)[:num_random]
                random_positions = tuple(index[selected] for index in mask_indices)
                random_ids = torch.randint(0, self.config.vocab_size - 2, (num_random,), device=target_ids.device)
                masked_sequence[random_positions] = random_ids
                target_for_loss[random_positions] = self.config.thinking_token_id

        return masked_sequence, mask_target, target_for_loss

    @torch.no_grad()
    def generate_masked_sequence(
        self,
        forward_f: Callable[..., torch.Tensor],
        input_ids: torch.LongTensor,
        target_ids: torch.LongTensor,
        forward_sample_params: Optional[Dict[str, Any]] = None,
    ) -> torch.LongTensor:
        forward_sample_params = forward_sample_params or {}

        def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
            if temperature == 0:
                return logits
            logits = logits.to(torch.float64)
            noise = torch.rand_like(logits, dtype=torch.float64)
            gumbel_noise = (-torch.log(noise)) ** temperature
            return logits.exp() / gumbel_noise

        def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
            mask_num = mask_index.sum(dim=1, keepdim=True)
            base = mask_num // steps
            remainder = mask_num % steps
            result = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
            for row in range(mask_num.size(0)):
                result[row, : remainder[row]] += 1
            return result

        prompt = input_ids
        gen_length = target_ids.shape[1]
        steps = self.config.diffusion.steps
        cfg_scale = self.config.diffusion.cfg_scale
        temperature = self.config.diffusion.temperature
        remasking = self.config.diffusion.remasking
        mask_id = self.config.mask_token_id
        thinking_id = self.config.thinking_token_id

        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=input_ids.device)
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        block_mask_index = x[:, prompt.shape[1] :] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for step_idx in range(steps):
            if self.config.diffusion.sage_thinking:
                num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps - step_idx)
                mask_index = (x == mask_id) | (x == thinking_id)
            else:
                mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_in = torch.cat([x, un_x], dim=0)
                logits = forward_f(x_in, **forward_sample_params)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = forward_f(x, **forward_sample_params)

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p_dtype = torch.float64 if torch.cuda.is_available() else torch.float32
                p = F.softmax(logits.to(p_dtype), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            confidence[:, : prompt.shape[1]] = -torch.inf

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for row in range(confidence.shape[0]):
                schedule_idx = 0 if self.config.diffusion.sage_thinking else step_idx
                k = int(num_transfer_tokens[row, schedule_idx].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[row], k=k)
                    transfer_index[row, select_index] = True
            x[transfer_index] = x0[transfer_index]

        return x
