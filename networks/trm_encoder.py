"""
networks/trm_encoder.py

TODO:
- Defines the TRMEncoder module used in TRMModel
- Handles input grid sequences and produces encoded representations

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class AbsolutePositionalEncoding(nn.Module):
    """
    The APE is built to be of the size of the maximum sequence length, and 
    then we slice it to match the length of grid 
    (i.e., the input sequence considering only the tokens part of the spatial manifold (e.g., no <BOS>, no <EOS>, no task tokens)).
    
    NOTE: The 2D APE encodes a spatial geometry while the 1D APE encodes a sequence geometry. 
          Hence for the 1D APE, it is ok to use the APE on the whole sequence instead of just the grid tokens. 

    Supports:
        - learned
        - 1d-sincos
        - 2d-sincos

    Controlled by:
        cfg.model.ape.type
        cfg.model.ape.mixer
    """

    def __init__(self, cfg):
        super().__init__()

        self.d_model = cfg.model.d_model
        self.max_h = cfg.model.get("max_h")
        self.max_w = cfg.model.get("max_w")
        self.max_task_seq_len = cfg.model.get("max_task_seq_len", 0)
        self.max_seq_len = cfg.model.max_seq_len
        self.total_seq_special_tokens_prepended = cfg.model.get("total_seq_special_tokens_prepended", 0)
        self.total_seq_special_tokens_appended = cfg.model.get("total_seq_special_tokens_appended", 0)

        self.dropout = nn.Dropout(cfg.model.get("dropout", 0.0))

        # -------------------------------------------------
        # Build positional embedding
        # -------------------------------------------------
        self.ape_type = cfg.model.ape.get("type", "learned")
        self.mixer = cfg.model.ape.get("mixer", "sum")

        if self.ape_type == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.max_seq_len, self.d_model)
            )
            nn.init.normal_(self.pos_embed, std=0.02)

        elif self.ape_type == "1d-sincos":
            pe = self._build_1d_sincos()
            self.register_buffer("pos_embed", pe, persistent=False)

        elif self.ape_type == "2d-sincos":
            pe = self._build_2d_sincos()
            self.register_buffer("pos_embed", pe, persistent=False)

        else:
            raise ValueError(f"Unsupported APE type: {self.ape_type}")

        # -------------------------------------------------
        # Mixer parameters
        # -------------------------------------------------

        if self.mixer == "learnable_scaling":
            self.alpha = nn.Parameter(torch.ones(1))

        elif self.mixer == "weighted_sum":
            self.input_weight = nn.Parameter(torch.ones(1))
            self.pos_weight = nn.Parameter(torch.ones(1))

        elif self.mixer == "layer_norm":
            self.layer_norm = nn.LayerNorm(self.d_model)

    # ======================================================
    # 1D SIN-COS
    # ======================================================

    def _build_1d_sincos(self):
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2)
            * (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    # ======================================================
    # 2D SIN-COS
    # ======================================================

    def _build_2d_sincos(self):
        if self.max_h is None or self.max_w is None:
            raise ValueError("max_h and max_w required for 2D sincos")

        H, W = self.max_h, self.max_w

        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sincos")

        dim = self.d_model // 4

        omega = 1.0 / (10000 ** (torch.arange(dim).float() / dim))

        y = torch.arange(H).float()
        x = torch.arange(W).float()

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        out_x = torch.einsum("m,d->md", grid_x, omega)
        out_y = torch.einsum("m,d->md", grid_y, omega)

        pos_grid = torch.cat(
            [
                torch.sin(out_x),
                torch.cos(out_x),
                torch.sin(out_y),
                torch.cos(out_y),
            ],
            dim=1,
        )

        # Prepend zeros for special tokens before grid
        if self.total_seq_special_tokens_prepended > 0:
            zeros = torch.zeros(
                self.total_seq_special_tokens_prepended,
                self.d_model
            )
            pos_grid = torch.cat([zeros, pos_grid], dim=0)

        # Append zeros after grid
        if self.total_seq_special_tokens_appended > 0:
            zeros = torch.zeros(
                self.total_seq_special_tokens_appended,
                self.d_model
            )
            pos_grid = torch.cat([pos_grid, zeros], dim=0)

        # Pad to max_seq_len if needed
        if pos_grid.size(0) < self.max_seq_len:
            pad = torch.zeros(
                self.max_seq_len - pos_grid.size(0),
                self.d_model
            )
            pos_grid = torch.cat([pos_grid, pad], dim=0)

        return pos_grid.unsqueeze(0)

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, x):
        B, S, D = x.shape

        if self.ape_type in ["learned", "1d-sincos"]:
            pos = self.pos_embed[:, :S, :].to(x.device)

        elif self.ape_type == "2d-sincos":
            pos = self.pos_embed[:, :S, :].to(x.device)

        else:
            raise ValueError(f"Unsupported APE type: {self.ape_type}")

        # -------------------------------------------------
        # Apply mixer
        # -------------------------------------------------

        if self.mixer == "sum":
            x = x + pos

        elif self.mixer == "learnable_scaling":
            x = x + self.alpha * pos

        elif self.mixer == "weighted_sum":
            x = (
                self.input_weight * x
                + self.pos_weight * pos
            )

        elif self.mixer == "layer_norm":
            x = self.layer_norm(x + pos)

        else:
            raise ValueError(f"Unsupported APE mixer: {self.mixer}")

        return self.dropout(x)


class MHSA(nn.Module):
    """ 
    Multi-Head Self-Attention block. 
    
    TODO: Implement RoPE? PoPE? etc.
    """

    def __init__(self, d_model, n_heads, attn_drop_p=0., proj_drop_p=0., qkv_bias=False):
        super().__init__()

        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=qkv_bias)  # W_qkv; *3 because we want embeddings for q, k, v from the input; this is equivalent to defining three separate linear layers (W_q, W_k, W_v) for q, k, and v
        
        self.softmax = nn.Softmax(dim=-1)

        self.attn_drop = nn.Dropout(attn_drop_p)
        self.proj = nn.Linear(d_model, d_model) # W_o
        self.proj_drop = nn.Dropout(proj_drop_p)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, seq_len, embed_dim = x.shape
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, S, 3*D]

        # Reshape the Queries, Keys, Values for multi-head
        head_embed_dim = embed_dim // self.n_heads
        x_qkv = x_qkv.reshape(B, seq_len, 3, self.n_heads, head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_embed_dim]

        # Get the Queries, Keys, Values for all heads
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, S, head_embed_dim], [B, num_heads, S, head_embed_dim], [B, num_heads, S, head_embed_dim])

        # --- MASKING LOGIC ---
        # F.scaled_dot_product_attention handles mask shapes:
        # If attn_mask is provided, it must be broadcastable to [B, H, S, S].
        # We generally use key_padding_mask for Encoder Self-Attention.
        # PyTorch SDPA expects mask to be Boolean (True=Mask/Ignore) or Float (-inf).
        
        mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask has True for Pad, but SDPA expects True for Attend
            kp_mask_attend = ~key_padding_mask.view(B, 1, 1, seq_len) 
            
            if mask is not None:
                mask = mask & kp_mask_attend # Logical AND for boolean masks
            else:
                mask = kp_mask_attend
        
        attn_out = F.scaled_dot_product_attention(
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0, # to disable dropout during evaluation, we make sure to pass a value of 0.0 when not in training mode
            scale=self.scale,
            is_causal=False
        )
        # ---

        # # Method 1: Raw compute of the attention scores
        # # Compute the attention scores
        # attn = (x_q @ x_k.transpose(-2, -1))    # [B, num_heads, S, S]; attention matrix/logits
        # attn_scaled = attn * self.scale   # [B, num_heads, S, S]; scaled attention logits
        # # NOTE: no masking
        # attn_scores = self.softmax(attn_scaled)   # [B, num_heads, S, S]; attention scores/weights
        # attn_scores = self.attn_drop(attn_scores)   # [B, num_heads, S, S]; dropout
        # self.attn_scores = attn_scores    # store the attention scores for visualization
        # attn_out = attn_scores @ x_v

        # Method 2: Memory-efficient attention (SDPA)
        attn_p = self.attn_drop.p if self.training else 0.0 # to disable dropout during evaluation, we make sure to pass a value of 0.0 when not in training mode
        
        attn_out = F.scaled_dot_product_attention(
            x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
            attn_mask=mask,
            dropout_p=attn_p,
            scale=self.scale,
            is_causal=False
        )  # [B, H, S, D]; H is the number of heads, D is the embedding dimension per head (so of the value)

        # We got the new embeddings from the Values and Attention scores at the end of SDPA, and now we concatenate back the heads through reshaping
        x = attn_out.transpose(1, 2).reshape(B, seq_len, embed_dim)  # [B, S, D] <-- [B, S, num_heads, head_embed_dim] <-- [B, num_heads, S, head_embed_dim] 
        x = self.proj(x)    # [B, S, D]; linearly project the new embeddings
        x = self.proj_drop(x)   # [B, S, D]; dropout
        return x

class FeedForward(nn.Module):
    """ Position-wise Feedforward block. """

    def __init__(self, d_model, d_ff, ff_dropout_p=0., activation_layer=nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation_fn = activation_layer
        self.dropout = nn.Dropout(ff_dropout_p)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TRMEncoderLayer(nn.Module):
    def __init__(self, cfg: DictConfig, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        
        # Self-Attention layer
        # Dropout layers for attention and projection
        attn_dropout_p = cfg.network.encoder.get("attn_dropout", 0.0)
        proj_dropout_p = cfg.network.encoder.get("proj_dropout", 0.0)
        self.proj_dropout = nn.Dropout(proj_dropout_p)  # dropout layer for the output of the feedforward block before adding the residual connection
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)   # PyTorch's built-in MultiheadAttention
        self.mhsa_block = MHSA(d_model, n_heads, attn_dropout_p, proj_dropout_p)
        
        # Activation function
        activation_fn = cfg.network.encoder.get("activation_func", "relu").lower()
        self.activation_layer = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation_fn, None)

        if self.activation_layer is None:
            raise ValueError(f"Activation function '{activation_fn}' not recognized. Choose from ['relu', 'gelu', 'leaky_relu']")

        # Feedforward layers
        ff_dropout_p = cfg.network.encoder.get("ff_dropout", 0.0)
        self.ff_block = FeedForward(d_model, d_ff, ff_dropout_p, self.activation_layer)

        # LayerNorm layers for the residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-Attention block
        src_normalized = self.norm1(src) # LayerNorm before the attention block (Pre-LN)
        src2 = self.mhsa_block(src_normalized, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + src2    # Residual connection

        # Feedforward block
        src_normalized = self.norm2(src) # LayerNorm before the feedforward block (Pre-LN)
        src2 = self.ff_block(src_normalized)
        src = src + src2    # Residual connection
        return src

class TRMEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # Global Model Params
        self.vocab_size = cfg.model.get("input_vocab_size", None)
        self.d_model = cfg.model.get("d_model", None)
        self.max_len = cfg.model.get("max_seq_len", None)

        self.dropout = cfg.model.get("dropout", 0.0)

        self.pad_token_id = cfg.model.get("pad_token_id", None)

        # Encoder Specific Params
        self.n_layers = cfg.network.encoder.get("num_layers", None)
        self.n_heads = cfg.network.encoder.get("num_heads", None)
        self.d_ff = cfg.network.encoder.get("ff_dim", None)


        # --- Input Embedding Layer ---
        # TODO: Using nn.Embedding is equivalent to using a OHE followed by a linear projection (e.g., 2DConv with kernel size 1).
        # NOTE: nn.Embedding is essentially a lookup table that maps token IDs to dense vectors (embeddings).
        #       The input token IDs are expected to be in the range [0, vocab_size-1], where each ID corresponds to a specific token in the vocabulary.
        #       The embedding layer learns a dense vector representation for each token ID during training, which allows the model to capture semantic relationships between tokens based on their usage in the training data.
        #       The embedding layer is initialized with random weights from a normal distribution (mean=0, std=0.02).
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        # nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)


        # --- Encoder Layers ---

        # Absolute Positional Encoding (APE)
        if cfg.model.ape.get("use_ape", False):
            self.use_ape = True
            self.pos_encoder = AbsolutePositionalEncoding(cfg)
        else:
            self.use_ape = False
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TRMEncoderLayer(cfg, self.d_model, self.n_heads, self.d_ff)
            for _ in range(self.n_layers)
        ])
        
        # Final LayerNorm after the encoder stack
        self.norm = nn.LayerNorm(self.d_model)

    def make_src_mask(self, src):
        # Create boolean mask: True where pad (Ignore), False where not pad (Attend)
        return (src == self.pad_token_id)

    def forward(self, src):
        """
        Args:
            src: [B, S]; input token IDs

        Outputs:
            x: [B, S, D]; encoded representations of the input sequence
        """
        # Checks if any input token ID is larger than what the embedding table can handle
        if src.max() >= self.vocab_size:
            raise ValueError(
                f"Input contains token ID {src.max().item()}, but Embedding vocab_size is {self.vocab_size}. "
                f"Check your DataModule or Config."
            )

        # Ensure type is LongTensor
        src = src.long()

        # Generate Padding Mask [B, S]
        src_key_padding_mask = self.make_src_mask(src)

        x = self.embedding(src) # [B, S, D]; embed the input token IDs to get their initial embeddings
        
        # TODO: not sure if we should scale the embeddings by sqrt(d_model) as in the original Transformer paper
        #       It is often done to help with optimization, but not sure if it makes sense with any APE
        # x = x * math.sqrt(self.d_model)
        
        if self.use_ape:
            x = self.pos_encoder(x) # [B, S, D]; add positional encodings to the input embeddings to inject positional information

        # Loop through the encoder layers, passing the output of one layer as the input to the next
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            # x = layer(x)

        x = self.norm(x) # [B, S, D]; apply layer normalization to the final output of the encoder stack
        
        return x