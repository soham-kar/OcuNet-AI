"""
GLAAI: Global-Local Attention Integration Module
Based on: Kumar et al. 2025 - "GLAAM and GLAAI: Pioneering attention models for robust automated cataract detection"

GLAAI is an enhanced version of GLAAM with:
1. Skip connections for better gradient flow
2. Layer normalization for training stability
3. Learnable fusion weights between global and local attention
4. Multi-head attention option for richer representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttentionWithNorm(nn.Module):
    """
    Enhanced Global Branch with Layer Normalization.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(GlobalAttentionWithNorm, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Add max pooling for richer features
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.LayerNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Combine avg and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2C)
        
        # FC layers
        y = self.fc(combined).view(b, c, 1, 1)
        
        return y


class LocalAttentionWithNorm(nn.Module):
    """
    Enhanced Local Branch with BatchNorm and deeper conv layers.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(LocalAttentionWithNorm, self).__init__()
        
        mid_channels = in_channels // reduction
        
        self.conv = nn.Sequential(
            # First 1x1 conv: reduce channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 3x3 conv: capture local context
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Final 1x1 conv: restore channels
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LearnableFusion(nn.Module):
    """
    Learnable fusion layer to combine global and local attention.
    Instead of simple multiplication, learns optimal combination weights.
    """
    
    def __init__(self, in_channels: int):
        super(LearnableFusion, self).__init__()
        
        # Learnable weights for combining global and local attention
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)
        
    def forward(self, x: torch.Tensor, global_attn: torch.Tensor, local_attn: torch.Tensor) -> torch.Tensor:
        """
        Combines input with weighted global and local attention.
        
        Args:
            x: Original input features (B, C, H, W)
            global_attn: Global attention weights (B, C, 1, 1)
            local_attn: Local attention map (B, C, H, W)
        Returns:
            Fused output (B, C, H, W)
        """
        # Sigmoid to keep weights in [0, 1]
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        
        # Weighted combination
        out = x * (alpha * global_attn + beta * local_attn)
        
        return out


class GLAAI(nn.Module):
    """
    Global-Local Attention Integration (GLAAI)
    
    Enhanced version of GLAAM with:
    - Combined avg+max pooling in global branch
    - 3x3 context convolution in local branch
    - Learnable fusion weights
    - Residual connection with scaling
    
    Architecture:
        Input (B, C, H, W)
            │
            ├── Global Branch (Avg+Max Pool → FC → Sigmoid)
            │       ↓
            │   Global Weights (B, C, 1, 1)
            │
            ├── Local Branch (1x1→3x3→1x1 Conv → Sigmoid)
            │       ↓
            │   Local Weights (B, C, H, W)
            │
            └── Learnable Fusion(α * Global + β * Local)
                    ↓
                Residual Addition
                    ↓
            Output (B, C, H, W)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, use_residual: bool = True):
        """
        Args:
            in_channels: Number of input feature channels
            reduction: Reduction ratio for bottleneck layers
            use_residual: Whether to add residual connection
        """
        super(GLAAI, self).__init__()
        
        self.global_branch = GlobalAttentionWithNorm(in_channels, reduction)
        self.local_branch = LocalAttentionWithNorm(in_channels, reduction)
        self.fusion = LearnableFusion(in_channels)
        self.use_residual = use_residual
        
        # Residual scaling factor (helps with training stability)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Attention-refined features of shape (B, C, H, W)
        """
        # Get attention from both branches
        global_attn = self.global_branch(x)  # (B, C, 1, 1)
        local_attn = self.local_branch(x)    # (B, C, H, W)
        
        # Learnable fusion
        attended = self.fusion(x, global_attn, local_attn)
        
        # Residual connection with learnable scaling
        if self.use_residual:
            out = x + self.gamma * attended
        else:
            out = attended
        
        return out


class GLAAIBlock(nn.Module):
    """
    GLAAI Block that can be inserted into any backbone.
    Includes optional dropout for regularization.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, dropout: float = 0.1):
        super(GLAAIBlock, self).__init__()
        
        self.glaai = GLAAI(in_channels, reduction, use_residual=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.glaai(x)
        out = self.dropout(out)
        return out


# For testing
if __name__ == "__main__":
    # Test GLAAI module
    batch_size = 4
    channels = 1280  # MobileNetV2 output channels
    height, width = 12, 12
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test basic GLAAI
    glaai = GLAAI(in_channels=channels, reduction=16)
    out = glaai(x)
    print(f"GLAAI Input shape:  {x.shape}")
    print(f"GLAAI Output shape: {out.shape}")
    print(f"Shapes match: {x.shape == out.shape}")
    
    # Test GLAAI Block
    glaai_block = GLAAIBlock(in_channels=channels, reduction=16, dropout=0.1)
    out_block = glaai_block(x)
    print(f"\nGLAAIBlock Output shape: {out_block.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in glaai.parameters())
    print(f"\nGLAAI Parameters: {total_params:,}")
    
    # Compare with GLAAM
    from glaam import GLAAM
    glaam = GLAAM(in_channels=channels, reduction=16)
    glaam_params = sum(p.numel() for p in glaam.parameters())
    print(f"GLAAM Parameters: {glaam_params:,}")
    print(f"GLAAI overhead: {(total_params - glaam_params) / glaam_params * 100:.1f}%")
