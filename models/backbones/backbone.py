"""
Backbone networks with integrated GLAAM/GLAAI attention.
Supports MobileNetV2 and InceptionV3 backbones.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention.glaam import GLAAM, GLAAMBlock
from attention.glaai import GLAAI, GLAAIBlock


class MobileNetV2WithAttention(nn.Module):
    """
    MobileNetV2 backbone with GLAAM/GLAAI attention inserted after the last conv layer.
    
    Output: 1280-dimensional feature vector (after global pooling)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        attention_type: Literal["glaam", "glaai"] = "glaam",
        attention_reduction: int = 16,
        freeze_backbone: bool = False
    ):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            attention_type: Which attention module to use ("glaam" or "glaai")
            attention_reduction: Reduction ratio for attention bottleneck
            freeze_backbone: Freeze backbone weights (for transfer learning)
        """
        super(MobileNetV2WithAttention, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            mobilenet = models.mobilenet_v2(weights=weights)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        
        # Extract feature layers (everything except classifier)
        self.features = mobilenet.features  # Output: (B, 1280, H/32, W/32)
        
        # Choose attention module
        if attention_type == "glaam":
            self.attention = GLAAMBlock(in_channels=1280, reduction=attention_reduction)
        elif attention_type == "glaai":
            self.attention = GLAAIBlock(in_channels=1280, reduction=attention_reduction)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension
        self.feature_dim = 1280
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            features: Pooled features of shape (B, 1280)
            attention_features: Pre-pooling features for Grad-CAM (B, 1280, H', W')
        """
        # Feature extraction
        features = self.features(x)  # (B, 1280, H/32, W/32)
        
        # Attention
        attended = self.attention(features)  # (B, 1280, H/32, W/32)
        
        # Global pooling
        pooled = self.pool(attended)  # (B, 1280, 1, 1)
        pooled = pooled.flatten(1)    # (B, 1280)
        
        return pooled, attended


class InceptionV3WithAttention(nn.Module):
    """
    InceptionV3 backbone with GLAAM/GLAAI attention.
    
    Output: 2048-dimensional feature vector (after global pooling)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        attention_type: Literal["glaam", "glaai"] = "glaam",
        attention_reduction: int = 16,
        freeze_backbone: bool = False
    ):
        super(InceptionV3WithAttention, self).__init__()
        
        # Load pretrained InceptionV3
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            inception = models.inception_v3(weights=weights, aux_logits=False)
        else:
            inception = models.inception_v3(weights=None, aux_logits=False)
        
        # Extract feature layers
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        
        # Choose attention module
        if attention_type == "glaam":
            self.attention = GLAAMBlock(in_channels=2048, reduction=attention_reduction)
        elif attention_type == "glaai":
            self.attention = GLAAIBlock(in_channels=2048, reduction=attention_reduction)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for name, param in self.named_parameters():
            if 'attention' not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W) - H, W should be >= 299
        Returns:
            features: Pooled features of shape (B, 2048)
            attention_features: Pre-pooling features for Grad-CAM
        """
        # Initial convolutions
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        features = self.Mixed_7c(x)  # (B, 2048, H', W')
        
        # Attention
        attended = self.attention(features)
        
        # Global pooling
        pooled = self.pool(attended)
        pooled = pooled.flatten(1)
        
        return pooled, attended


def get_backbone(
    name: str = "mobilenetv2",
    attention_type: str = "glaam",
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get backbone with attention.
    
    Args:
        name: Backbone name ("mobilenetv2" or "inceptionv3")
        attention_type: Attention type ("glaam" or "glaai")
        pretrained: Use pretrained weights
    
    Returns:
        Backbone module
    """
    name = name.lower()
    
    if name == "mobilenetv2":
        return MobileNetV2WithAttention(
            pretrained=pretrained,
            attention_type=attention_type,
            **kwargs
        )
    elif name == "inceptionv3":
        return InceptionV3WithAttention(
            pretrained=pretrained,
            attention_type=attention_type,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")


# For testing
if __name__ == "__main__":
    # Test MobileNetV2 with GLAAM
    print("Testing MobileNetV2 + GLAAM...")
    model = MobileNetV2WithAttention(pretrained=True, attention_type="glaam")
    x = torch.randn(2, 3, 384, 384)
    features, attn_features = model(x)
    print(f"Input: {x.shape}")
    print(f"Output features: {features.shape}")
    print(f"Attention features: {attn_features.shape}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    print("\n" + "="*50)
    
    # Test MobileNetV2 with GLAAI
    print("\nTesting MobileNetV2 + GLAAI...")
    model_glaai = MobileNetV2WithAttention(pretrained=True, attention_type="glaai")
    features, attn_features = model_glaai(x)
    print(f"Output features: {features.shape}")
    
    total_glaai = sum(p.numel() for p in model_glaai.parameters())
    print(f"Total params: {total_glaai:,}")
