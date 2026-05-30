"""
GLAAM: Global-Local Attention Aggregation Module
Based on: Kumar et al. 2025 - "GLAAM and GLAAI: Pioneering attention models for robust automated cataract detection"

This module implements the Global-Local Attention Aggregation mechanism that combines:
1. Global Branch: Channel-wise attention via Global Average Pooling
2. Local Branch: Spatial attention via 1x1 convolutions
3. Aggregation: Element-wise multiplication of both attention outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionBranch(nn.Module):
    """
    Global Branch: Captures channel-wise feature importance.
    Uses Global Average Pooling followed by FC layers to generate channel attention weights.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input feature channels
            reduction: Reduction ratio for the bottleneck (default: 16)
        """
        super(GlobalAttentionBranch, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bottleneck FC layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        b, c, _, _ = x.size()
        
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        
        # FC layers: (B, C) -> (B, C)
        y = self.fc(y).view(b, c, 1, 1)
        
        return y


class LocalAttentionBranch(nn.Module):
    """
    Local Branch: Captures spatial attention at each position.
    Uses 1x1 convolutions to generate pixel-wise attention map.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input feature channels
            reduction: Reduction ratio for intermediate channels
        """
        super(LocalAttentionBranch, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Spatial attention map of shape (B, C, H, W)
        """
        return self.conv(x)


class GLAAM(nn.Module):
    """
    Global-Local Attention Aggregation Module (GLAAM)
    
    Combines global channel attention and local spatial attention through
    element-wise multiplication for robust feature refinement.
    
    Architecture:
        Input (B, C, H, W)
            ├── Global Branch → Channel weights (B, C, 1, 1)
            └── Local Branch  → Spatial weights (B, C, H, W)
                    ↓
            Aggregation: Input * Global * Local
                    ↓
        Output (B, C, H, W)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input feature channels
            reduction: Reduction ratio for both branches (default: 16)
        """
        super(GLAAM, self).__init__()
        
        self.global_branch = GlobalAttentionBranch(in_channels, reduction)
        self.local_branch = LocalAttentionBranch(in_channels, reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Attention-refined features of shape (B, C, H, W)
        """
        # Get attention weights from both branches
        global_weights = self.global_branch(x)  # (B, C, 1, 1)
        local_weights = self.local_branch(x)    # (B, C, H, W)
        
        # Aggregate: element-wise multiplication
        # Global weights broadcast across spatial dimensions
        out = x * global_weights * local_weights
        
        return out


class GLAAMBlock(nn.Module):
    """
    GLAAM Block with residual connection for better gradient flow.
    Can be inserted after any convolutional layer in a backbone.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, use_residual: bool = True):
        """
        Args:
            in_channels: Number of input feature channels
            reduction: Reduction ratio for attention modules
            use_residual: Whether to use residual connection (default: True)
        """
        super(GLAAMBlock, self).__init__()
        
        self.glaam = GLAAM(in_channels, reduction)
        self.use_residual = use_residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        if self.use_residual:
            return x + self.glaam(x)
        else:
            return self.glaam(x)


# For testing
if __name__ == "__main__":
    # Test GLAAM module
    batch_size = 4
    channels = 1280  # MobileNetV2 output channels
    height, width = 12, 12
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test basic GLAAM
    glaam = GLAAM(in_channels=channels, reduction=16)
    out = glaam(x)
    print(f"GLAAM Input shape:  {x.shape}")
    print(f"GLAAM Output shape: {out.shape}")
    print(f"Shapes match: {x.shape == out.shape}")
    
    # Test GLAAM Block with residual
    glaam_block = GLAAMBlock(in_channels=channels, reduction=16, use_residual=True)
    out_block = glaam_block(x)
    print(f"\nGLAAMBlock Output shape: {out_block.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in glaam.parameters())
    print(f"\nGLAAM Parameters: {total_params:,}")
