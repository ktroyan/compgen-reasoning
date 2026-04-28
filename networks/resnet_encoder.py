"""
networks/resnet_encoder.py

Defines ResNetEncoder, a custom ResNet-style CNN encoder for 2D grid inputs.

- Accepts a 2D grid of token IDs [B, H, W] and produces per-position embeddings [B, H*W, D].

- One-hot encodes the categorical token IDs into C channels, then passes them through a stack
  of residual blocks (without spatial downsampling) and a final 1x1 projection to d_model.

"""

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig


def one_hot_encode(x: torch.Tensor, num_token_categories: int) -> torch.Tensor:
    """
    Performs One-Hot Encoding (OHE) of the values of a 3D tensor (batch dimension and 2D tensor with possible values/tokens: 0, ..., 9, <pad_token>, ..?)
    """

    # Convert to one-hot representation
    x_ohe = torch.nn.functional.one_hot(x.long(), num_classes=num_token_categories)  # [B, H, W, C=num_token_categories]

    # Reshape to get the number of possible categorical values, which is used as channels (C), as the first dimension
    x_ohe = x_ohe.permute(0, 3, 1, 2).float()  # [B, C=num_token_categories, H, W]

    return x_ohe


class CustomBasicBlock(nn.Module):
    """ A smaller version of ResNet BasicBlock without aggressive downsampling since we use image grids of small size. """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super().__init__()

        # Sub-block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)  # consider using dilation=dilation for the first conv layer in order to increase the receptive field
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Sub-block 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x_res = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x_res)


class ResNetEncoder(nn.Module):
    """
    Custom ResNet model for 2D grid inputs.

    NOTE: We do not use Max Pooling (e.g. nn.MaxPool2d(kernel_size=2, stride=2)), which would downsample as: [B, C, H/2, W/2])
    (for better receptive field) because it is not ideal to downsample since we want to have the same spatial dimensions at the end, which means we would need to upsample back.
    NOTE: Think of using Average Pooling and dilation for increased receptive field.

    Input: 2D grid of token IDs; [B, H, W]
    Output: per-position embeddings; [B, H*W, d_model]

    """
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.embed_dim = cfg.model.d_model
        self.num_token_categories = cfg.model.input_vocab_size

        # base_channels is the width multiplier for the backbone.
        # Channel layout: C -> 2C -> 4C -> 4C -> 8C -> 4C -> embed_dim
        # Approx params: C=32 -> 1.4M | C=62 -> 5.0M | C=64 -> 5.4M
        C = cfg.network.encoder.get("base_channels", 32)

        self.backbone = nn.Sequential(
            nn.Conv2d(self.num_token_categories, C, kernel_size=1, stride=1, padding=0),    # initial convolutional layer
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            CustomBasicBlock(C, 2*C, kernel_size=3, stride=1, padding=1),
            CustomBasicBlock(2*C, 4*C, kernel_size=3, stride=1, padding=1),
            CustomBasicBlock(4*C, 4*C, kernel_size=1, stride=1, padding=0),
            CustomBasicBlock(4*C, 8*C, kernel_size=3, stride=1, padding=1), # can try: dilation=2, padding=2 (since P = D * (K - 1) / 2)
            CustomBasicBlock(8*C, 4*C, kernel_size=1, stride=1, padding=0), # can try: dilation=4, padding=4 (since P = D * (K - 1) / 2)
            nn.Conv2d(4*C, self.embed_dim, kernel_size=1, stride=1),  # project to embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] — 2D grid of token IDs
        Returns:
            [B, H*W, embed_dim]
        """

        B, H, W = x.shape

        # Create a channel dimension for the input image via OHE
        x = one_hot_encode(x, self.num_token_categories)    # [B, C, H, W] <-- [B, H, W]

        # Encode the input image by passing it through the backbone
        x = self.backbone(x)    # [B, embed_dim, H, W]

        # Flatten the spatial dimensions to get a sequence of pixels
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)  # [B, H*W, embed_dim]

        return x
