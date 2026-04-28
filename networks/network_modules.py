import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Norm layers
# -------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weight * x
    
def build_norm(norm_type, d_model):
    norm_type = norm_type.lower()

    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)

    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)

    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

# -------------------------------------------------
# Absolute Positional Encoding (APE)
# -------------------------------------------------
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

    Mixer options for combining the APE with the input embeddings:
        - sum: simple addition (default)
        - learnable_scaling: APE is scaled by a learnable parameter alpha before being added to the input embeddings
        - weighted_sum: learnable weights for both input and APE, so the mixed embedding is a weighted sum of the input and APE
        - layer_norm: apply layer normalization after summing the input and APE
        - rms_norm: apply RMS normalization after summing the input and APE

    If selected, 2D APE is only applied to the grid tokens as they live in a 2D space manifold, while the special tokens (prefix and suffix tokens such as <BOS>, <EOS>, task tokens) are encoded with 1D APE as they only live in a sequence.
    
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
            pe = self._build_1d_sincos(self.max_seq_len)
            self.register_buffer("pos_embed", pe, persistent=False)

        elif self.ape_type == "2d-sincos":
            pe = self._build_2d_sincos()
            self.register_buffer("pos_embed", pe, persistent=False)

        else:
            raise ValueError(f"Unsupported APE type: {self.ape_type}")

        # -------------------------------------------------
        # APE Mixer
        # -------------------------------------------------

        if self.mixer == "learnable_scaling":
            self.alpha = nn.Parameter(torch.ones(1))

        elif self.mixer == "weighted_sum":
            self.input_weight = nn.Parameter(torch.ones(1))
            self.pos_weight = nn.Parameter(torch.ones(1))

        elif self.mixer == "layer_norm":
            self.layer_norm = nn.LayerNorm(self.d_model)
        
        elif self.mixer == "rms_norm":
            self.rms_norm = RMSNorm(self.d_model)

    # -----------------------------------------------------=
    # 1D sin-cos
    # -----------------------------------------------------=

    def _build_1d_sincos(self, length):
        pe = torch.zeros(length, self.d_model)

        position = torch.arange(0, length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2)
            * (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    # -----------------------------------------------------=
    # 2D sin-cos
    # -----------------------------------------------------=

    def _build_2d_sincos_grid(self):

        grid_h = torch.arange(self.max_h)
        grid_w = torch.arange(self.max_w)

        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")

        grid = torch.stack(grid, dim=0)  # [2, H, W]
        grid = grid.reshape(2, -1)       # [2, H*W]

        pos_h = grid[0]
        pos_w = grid[1]

        dim_half = self.d_model // 2

        div_term = torch.exp(
            torch.arange(0, dim_half, 2) * (-math.log(10000.0) / dim_half)
        )

        pe_h = torch.zeros(pos_h.size(0), dim_half)
        pe_w = torch.zeros(pos_w.size(0), dim_half)

        pe_h[:, 0::2] = torch.sin(pos_h.unsqueeze(1) * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h.unsqueeze(1) * div_term)

        pe_w[:, 0::2] = torch.sin(pos_w.unsqueeze(1) * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w.unsqueeze(1) * div_term)

        return torch.cat([pe_h, pe_w], dim=1)

    def _build_2d_sincos(self):

        if self.max_h is None or self.max_w is None:
            raise ValueError("max_h and max_w required for 2D sincos")

        prefix_len = self.total_seq_special_tokens_prepended
        suffix_len = self.total_seq_special_tokens_appended

        _grid_len = self.max_h * self.max_w

        # --- Prefix: 1D PE ---
        if prefix_len > 0:
            prefix_pe = self._build_1d_sincos(prefix_len)
        else:
            prefix_pe = torch.zeros(0, self.d_model)

        # --- Grid: 2D PE ---
        grid_pe = self._build_2d_sincos_grid()

        # --- Suffix: 1D PE ---
        if suffix_len > 0:
            suffix_pe = self._build_1d_sincos(suffix_len)
        else:
            suffix_pe = torch.zeros(0, self.d_model)

        pos = torch.cat(
            [prefix_pe, grid_pe, suffix_pe],
            dim=0
        )

        # Pad to max_seq_len
        if pos.size(0) < self.max_seq_len:
            pad = torch.zeros(
                self.max_seq_len - pos.size(0),
                self.d_model
            )
            pos = torch.cat([pos, pad], dim=0)

        return pos.unsqueeze(0)

    # -----------------------------------------------------=
    # Forward
    # -----------------------------------------------------=

    def forward(self, x):

        B, S, D = x.shape

        pos = self.pos_embed[:, :S, :].to(x.device)

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

        elif self.mixer == "rms_norm":
            x = self.rms_norm(x + pos)

        return self.dropout(x)

def get_ape(cfg):
    ape_type = cfg.model.ape.get("type", "learned")

    if ape_type in ("learned", "1d-sincos", "2d-sincos"):
        return AbsolutePositionalEncoding(cfg)
    else:
        raise ValueError(f"Unsupported APE type: {ape_type}")

# -------------------------------------------------
# Relative Positional Encoding (RPE)
# -------------------------------------------------
class RotaryEmbedding1D(nn.Module):
    """
    Adapted from nano-TRM codebase: https://github.com/olivkoch/nano-trm/blob/main/src/nn/modules/trm_block.py

    max_seq_len is the maximum number of positions we want to encode

    """
    def __init__(self, head_embed_dim, max_seq_len, base=10000):
        super().__init__()
        self.enabled = base > 0
        if not self.enabled:
            return
        
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_embed_dim, 2, dtype=torch.float32) / head_embed_dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        if not self.enabled:
            return None, None
        
        return self.cos_cached, self.sin_cached
    
class RotaryEmbedding2D(nn.Module):
    """
    Adapted from nano-TRM codebase: https://github.com/olivkoch/nano-trm/blob/main/src/nn/modules/trm_block.py
    
    2D RoPE with 1D RoPE for prefix and suffix tokens.

    Layout:
    [ prefix tokens | grid tokens | suffix tokens ]
      1D RoPE          2D RoPE        1D RoPE
    
    """

    def __init__(self, head_embed_dim, max_seq_len, prefix_len=0, suffix_len=0, max_h=None, max_w=None, base=10000):
        super().__init__()

        self.head_embed_dim = head_embed_dim
        assert head_embed_dim % 4 == 0, "head_embed_dim must be divisible by 4 for 2D RoPE"
        
        self.max_seq_len = max_seq_len
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.max_h = max_h
        self.max_w = max_w

        assert self.max_seq_len == self.prefix_len + self.max_h * self.max_w + self.suffix_len, "max_seq_len must equal prefix_len + max_h*max_w + suffix_len, otherwise the RoPE cache is smaller than the sequence length"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_embed_dim, 2, dtype=torch.float32) / head_embed_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache()

    def _build_cache(self):

        n_freq = self.inv_freq.shape[0] # dim // 2
        quarter = n_freq // 2   # dim // 4  (because 2D, so need to split the frequencies into two halves for row and column)

        # -------------------------------------------------
        # Prefix (1D RoPE)
        # -------------------------------------------------

        if self.prefix_len > 0:

            prefix_pos = torch.arange(self.prefix_len, dtype=torch.float32)

            prefix_freqs = torch.outer(prefix_pos, self.inv_freq)

            prefix_emb = torch.cat([prefix_freqs, prefix_freqs], dim=-1)

        else:
            prefix_emb = torch.zeros(0, self.head_embed_dim)

        # -------------------------------------------------
        # Grid (2D RoPE)
        # -------------------------------------------------

        grid_len = self.max_h * self.max_w

        indices = torch.arange(grid_len, dtype=torch.float32)

        rows = torch.div(indices, self.max_w, rounding_mode="floor")    # indices // self.max_w
        cols = torch.remainder(indices, self.max_w) # indices % self.max_w

        row_freqs = torch.outer(rows, self.inv_freq[:quarter])
        col_freqs = torch.outer(cols, self.inv_freq[:quarter])

        grid_emb = torch.cat(
            [row_freqs, col_freqs, row_freqs, col_freqs],
            dim=-1,
        )

        # -------------------------------------------------
        # Suffix (1D RoPE)
        # -------------------------------------------------

        if self.suffix_len > 0:

            suffix_pos = torch.arange(self.suffix_len, dtype=torch.float32)

            suffix_freqs = torch.outer(suffix_pos, self.inv_freq)

            suffix_emb = torch.cat([suffix_freqs, suffix_freqs], dim=-1)

        else:
            suffix_emb = torch.zeros(0, self.head_embed_dim)

        # -------------------------------------------------
        # Combine positional embeddings for each "separated" part of the sequence and build cache
        # -------------------------------------------------

        full_emb = torch.cat(
            [
                prefix_emb,
                grid_emb,
                suffix_emb,
            ],
            dim=0,
        )

        self.register_buffer("cos_cached", full_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", full_emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached
    
def rotate_half(x: torch.Tensor):
    """
    Adapted from nano-TRM codebase: https://github.com/olivkoch/nano-trm/blob/main/src/nn/modules/trm_block.py
    """
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Adapted from nano-TRM codebase: https://github.com/olivkoch/nano-trm/blob/main/src/nn/modules/trm_block.py
    """
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)

    # Broadcast cos and sin to match the shape of q and k for element-wise multiplication
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

# -------------------------------------------------
# Multi-Head Self-Attention (MHSA) block
# -------------------------------------------------
class MHSA(nn.Module):
    """ 
    Multi-Head Self-Attention block. 
    
    TODO: Implement PoPE? Implement MTA? Etc.

    """

    def __init__(self, cfg, d_model, n_heads, attn_drop_p=0., proj_drop_p=0., qkv_bias=False):
        super().__init__()

        self.n_heads = n_heads
        self.head_embed_dim = d_model // n_heads
        self.scale = self.head_embed_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=qkv_bias)  # W_qkv; *3 because we want embeddings for q, k, v from the input; this is equivalent to defining three separate linear layers (W_q, W_k, W_v) for q, k, and v
        
        # self.softmax = nn.Softmax(dim=-1)

        self.attn_drop = nn.Dropout(attn_drop_p)
        self.proj = nn.Linear(d_model, d_model) # W_o
        self.proj_drop = nn.Dropout(proj_drop_p)

        # --- RPE ---
        if cfg.model.rpe.get("use_rpe", False):
            self.rpe_type = cfg.model.rpe.get("type", None)
        else:
            self.rpe_type = None
    
        max_seq_len = cfg.model.max_seq_len

        # RoPE
        if self.rpe_type == "1d-rope":
            self.rotary_emb = RotaryEmbedding1D(self.head_embed_dim,
                                                max_seq_len,
                                                10000
                                                )

        elif self.rpe_type == "2d-rope":
            max_h = cfg.model.max_h
            max_w = cfg.model.max_w

            prefix_len = cfg.model.total_seq_special_tokens_prepended
            suffix_len = cfg.model.total_seq_special_tokens_appended

            self.rotary_emb = RotaryEmbedding2D(
                self.head_embed_dim,
                max_seq_len,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                max_h=max_h,
                max_w=max_w,
                base=10000
            )
        
        else:
            self.rotary_emb = None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        B, seq_len, embed_dim = x.shape
        
        # Compute the Queries, Keys, Values from the input embeddings by a linear projection
        x_qkv = self.qkv_proj(x) # [B, S, 3*D]

        # Reshape the Queries, Keys, Values for multi-head
        x_qkv = x_qkv.reshape(B, seq_len, 3, self.n_heads, self.head_embed_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_embed_dim]

        # Get the Queries, Keys, Values for all heads
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]    # ([B, num_heads, S, head_embed_dim], [B, num_heads, S, head_embed_dim], [B, num_heads, S, head_embed_dim])

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb()  # [S, head_embed_dim]
            cos, sin = cos[:seq_len].to(x.device), sin[:seq_len].to(x.device)
            x_q, x_k = apply_rotary_pos_emb(x_q, x_k, cos, sin)

        # --- Masking ---
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
        
        # --- Attention Computation ---

        # # Method 1: Raw computation of the attention scores
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

def get_mhsa_block(cfg, d_model, n_heads, attn_dropout_p, proj_dropout_p):
    mhsa_type = cfg.network.encoder.get("mhsa_type", "mhsa")

    if mhsa_type == "mhsa":
        return MHSA(cfg, d_model, n_heads, attn_dropout_p, proj_dropout_p)
    else:
        raise ValueError(f"Unsupported MHSA block: {mhsa_type}")

# -------------------------------------------------
# FeedForward block
# -------------------------------------------------
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
    
class ConvSwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        _hidden_size = int(2/3 * d_ff * 2)   # TODO: see what hidden size to use instead of 2 * d_ff to keep the number of parameters similar to the MLP feedforward block (useful for comparison)
        
        self.conv1 = nn.Conv1d(d_model, 2 * d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1,2)    # [B, D, S] <-- [B, S, D]
        x_proj = self.conv1(x)
        x1, x2 = x_proj.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1,2) # [B, S, D]

        return x

def get_ff_block(cfg, d_model, d_ff, activation_layer):
    ff_dropout_p = cfg.network.encoder.get("ff_dropout", 0.0)

    ff_type = cfg.network.encoder.get("ff_type", "mlp")

    if ff_type == "mlp":
        ff_block = FeedForward(d_model, d_ff, ff_dropout_p, activation_layer)

    elif ff_type == "convswiglu":
        ff_block = ConvSwiGLU(d_model, d_ff, ff_dropout_p)

    else:
        raise ValueError(f"Unsupported FF block: {ff_type}")
        
    return ff_block

# -------------------------------------------------
# Activation function layer
# -------------------------------------------------
def get_activation_layer(cfg):
    activation_fn = cfg.network.encoder.get("activation_func", "relu").lower()
    activation_layer = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU()
    }.get(activation_fn, None)

    if activation_layer is None:
        raise ValueError(f"Activation function '{activation_fn}' not recognized. Choose from ['relu', 'gelu', 'leaky_relu']")
    
    return activation_layer

# -------------------------------------------------
# Weights initialization
# -------------------------------------------------
def initialize_weights(module, init_type="xavier"):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):

        if init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)

        elif init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight)

        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif init_type == "trunc_normal":
            nn.init.trunc_normal_(module.weight, std=0.02)

        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)