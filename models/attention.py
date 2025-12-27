"""
CBAM (Convolutional Block Attention Module) for YOLOv7 1B4H

Phase 3: Task-Aware Attention Mechanism
Provides Channel Attention and Spatial Attention to help each Head
focus on relevant features for its assigned class categories.

Reference: CBAM: Convolutional Block Attention Module (ECCV 2018)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Learns "what" features are important by analyzing channel-wise statistics.
    Uses both average pooling and max pooling to capture different aspects.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for the MLP bottleneck (default: 16)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.fc(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [B, C, 1, 1]
        attention = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Learns "where" to focus by analyzing spatial relationships.
    Concatenates channel-wise average and max features, then applies convolution.

    Args:
        kernel_size: Convolution kernel size (default: 7)
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines Channel Attention and Spatial Attention sequentially.
    First refines features along channel dimension, then along spatial dimension.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention (default: 16)
        kernel_size: Kernel size for spatial attention (default: 7)

    Example:
        >>> cbam = CBAM(256)
        >>> x = torch.randn(1, 256, 40, 40)
        >>> out = cbam(x)  # [1, 256, 40, 40], same shape as input
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention: refine "what"
        x = x * self.channel_attention(x)
        # Spatial attention: refine "where"
        x = x * self.spatial_attention(x)
        return x


# For compatibility with YOLOv7 module registration
def autopad(k, p=None):
    """Calculate same padding for convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


if __name__ == '__main__':
    # Unit test for CBAM module
    print("Testing CBAM module...")

    # Test with different channel sizes (P3=128, P4=256, P5=512)
    test_channels = [128, 256, 512]
    batch_size = 2

    for ch in test_channels:
        # Create CBAM module
        cbam = CBAM(ch)

        # Create test input (simulating feature map)
        h, w = 40, 40  # Example spatial size
        x = torch.randn(batch_size, ch, h, w)

        # Forward pass
        out = cbam(x)

        # Verify output shape
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

        # Verify attention weights are in valid range
        with torch.no_grad():
            ca_weights = cbam.channel_attention(x)
            sa_weights = cbam.spatial_attention(x)

            assert ca_weights.min() >= 0 and ca_weights.max() <= 1, \
                f"Channel attention out of range: [{ca_weights.min()}, {ca_weights.max()}]"
            assert sa_weights.min() >= 0 and sa_weights.max() <= 1, \
                f"Spatial attention out of range: [{sa_weights.min()}, {sa_weights.max()}]"

        print(f"  Channel {ch}: OK (output shape: {out.shape})")

    print("\nAll CBAM tests passed!")
