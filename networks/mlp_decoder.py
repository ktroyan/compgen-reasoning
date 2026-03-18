import torch
import torch.nn as nn
from typing import Optional


class MLPDecoder(nn.Module):
    """
    Simple MLP decoder that outputs logits from an input vector.
    
    Args:
        input_dim: Dimension of input features (e.g., encoder output).
        hidden_dim: Dimension of hidden layers.
        output_dim: Number of output classes (logits).
        num_layers: Number of hidden layers (default: 2).
        dropout: Dropout rate (default: 0.1).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        # Build hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_features, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Logits of shape (batch_size, output_dim).
        """
        return self.mlp(x)