"""
networks/mlp_decoder.py

Defines MLPDecoder, a decoder network used in a model such as TransformerModel.

- A two-layer MLP (d_model -> hidden_dim -> output_dim) applied independently at each grid token position. 
- Projects the encoder's output embeddings to logits over the output vocabulary.
- Activation function and dropout rate are configurable via the network decoder config.

"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class MLPDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Dimensions
        self.d_model = cfg.network.encoder.get("output_dim", None)  # should match encoder output dimension
        self.output_dim = cfg.model.get("output_dim", None)   # this is the same as the output vocab size
        self.vocab_size = cfg.model.get("output_vocab_size", None)   # this should be the output vocab size

        assert self.output_dim == self.vocab_size, f"output_dim {self.output_dim} != vocab_size {self.vocab_size}"

        # Architecture parameters
        hidden_dim = cfg.network.decoder.get("hidden_dim", 256)

        dropout_p = cfg.network.decoder.get("dropout", 0.1)

        activation_fn = cfg.network.decoder.get("activation_func", "relu").lower()
        activation_layer = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU()
        }.get(activation_fn, None)

        if activation_layer is None:
            raise ValueError(f"Activation function '{activation_fn}' not recognized. Choose from ['relu', 'gelu', 'leaky_relu']")

        # MLP (2-layer)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            activation_layer,
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x, tgt=None, memory_key_padding_mask=None):
        logits = self.mlp(x)  # [B, H*W, output_dim]
        return logits
