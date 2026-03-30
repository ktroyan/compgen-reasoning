"""
networks/transformer_encoder.py

- Defines the TransformerEncoder module used in TransformerModel
- Handles input grid sequences and produces encoded representations

"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig

# Personal imports
from .network_modules import AbsolutePositionalEncoding
from .network_modules import build_norm
from .network_modules import MHSA
from .network_modules import get_ff_block
from .network_modules import get_activation_layer
from .network_modules import initialize_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, cfg: DictConfig, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        
        # Self-Attention layer
        # Dropout layers for attention and projection
        attn_dropout_p = cfg.network.encoder.get("attn_dropout", 0.0)
        proj_dropout_p = cfg.network.encoder.get("proj_dropout", 0.0)
        self.proj_dropout = nn.Dropout(proj_dropout_p)  # dropout layer for the output of the feedforward block before adding the residual connection
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)   # PyTorch's built-in MultiheadAttention
        self.mhsa_block = MHSA(cfg, d_model, n_heads, attn_dropout_p, proj_dropout_p)

        # Activation function
        self.activation_layer = get_activation_layer(cfg)
        # FeedForward layers
        self.ff_block = get_ff_block(cfg, d_model, d_ff, self.activation_layer)
        

        # Norm layers for the residual connections
        norm_type = cfg.network.encoder.get("norm", "layernorm")
        self.norm1 = build_norm(norm_type, d_model)
        self.norm2 = build_norm(norm_type, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-Attention block
        src_normalized = self.norm1(src) # norm before the attention block
        src2 = self.mhsa_block(src_normalized, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + src2    # residual connection

        # Feedforward block
        src_normalized = self.norm2(src) # norm before the feedforward block
        src2 = self.ff_block(src_normalized)
        src2 = self.proj_dropout(src2)   # dropout after the feedforward block
        src = src + src2    # residual connection
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # Global Model Params
        self.vocab_size = cfg.model.get("input_vocab_size", None)
        self.d_model = cfg.model.get("d_model", None)
        self.max_seq_len = cfg.model.get("max_seq_len", None)

        self.dropout = cfg.model.get("dropout", 0.0)

        self.pad_token_id = cfg.model.get("pad_token_id", None)

        # Encoder Specific Params
        self.n_layers = cfg.network.encoder.get("num_layers", None)
        self.n_heads = cfg.network.encoder.get("num_heads", None)
        self.d_ff = cfg.network.encoder.get("ff_dim", None)


        # --- Input Embedding Layer ---
        # NOTE: Using nn.Embedding is equivalent to using a OHE followed by a linear projection (e.g., 2DConv with kernel size 1).
        # NOTE: nn.Embedding is essentially a lookup table that maps token IDs to dense vectors (embeddings).
        #       The input token IDs are expected to be in the range [0, vocab_size-1], where each ID corresponds to a specific token in the vocabulary.
        #       The embedding layer learns a dense vector representation for each token ID during training, which allows the model to capture semantic relationships between tokens based on their usage in the training data.
        #       The embedding layer is initialized with random weights from a normal distribution (mean=0, std=0.02).
        self.input_embedding = nn.Embedding(self.vocab_size, self.d_model)
        # TODO: Set the initial LR for the update of the embeddings to 1e-2 or something higher than the rest of the model to encourage faster learning of the input embeddings, especially in the early stages of training when the model is still learning to map token IDs to meaningful representations.
        #       Also see for a better initial distribution
        # nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)


        # --- Encoder Layers ---

        # Absolute Positional Encoding (APE)
        if cfg.model.ape.get("use_ape", False):
            self.use_ape = True
            self.pos_encoder = AbsolutePositionalEncoding(cfg)
        else:
            self.use_ape = False
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg, self.d_model, self.n_heads, self.d_ff)
            for _ in range(self.n_layers)
        ])
        
        # Final norm after the encoder stack
        self.norm = build_norm(cfg.network.encoder.get("norm", "layernorm"), self.d_model)

        # --- Gradient Checkpointing ---
        self.use_activations_checkpointing = cfg.network.encoder.get("use_activations_checkpointing", False)

        # --- Weights Initialization ---
        init_type = cfg.network.encoder.get("weights_init", "xavier")
        self.apply(lambda m: initialize_weights(m, init_type))

    def make_src_mask(self, src):
        # Create boolean mask: True where pad (Ignore), False where not pad (Attend)
        return (src == self.pad_token_id)

    def forward(self, src):
        """
        Transformer Encoder forward pass
        
        Args:
            src: [B, S]; input tokens (token IDs)

        Outputs:
            x: [B, S, D]; encoded representation of the input sequence
        """

        # Check if any input token ID is larger than what the embedding table was set to handle
        if src.max() >= self.vocab_size:
            raise ValueError(f"Input contains token ID {src.max().item()}, but Embedding vocab_size is {self.vocab_size}.")

        # Ensure type is LongTensor
        src = src.long()

        # Generate Padding Mask [B, S]
        src_key_padding_mask = self.make_src_mask(src)  # [B, S]; True where pad (Ignore), False where not pad (Attend)

        # ---- Embed input ----
        x = self.input_embedding(src) # [B, S, D]; embed the input tokens to get their initial embeddings
        # TODO: not sure if we should scale the embeddings by sqrt(d_model) as in the original Transformer paper
        #       It is often done to help with optimization, but not sure if it makes sense with any APE
        # x = x * math.sqrt(self.d_model)

        if self.use_ape:
            x = self.pos_encoder(x) # [B, S, D]; add positional encodings to the input embeddings to inject positional information

        B, S, D = x.shape
        
        # Loop through the encoder layers
        for layer in self.layers:
            if self.use_activations_checkpointing:
                x = checkpoint(layer, x, src_key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        x = self.norm(x) # [B, S, D]; apply layer normalization to the final output of the encoder stack
        
        return x