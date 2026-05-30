"""
MobileNetV2 Baseline for Cataract Detection.

This is a simple baseline model WITHOUT attention mechanisms.
Used for comparison with GLAAM to show improvement.

Features:
- MobileNetV2 backbone (same as GLAAM for fair comparison)
- Binary classification (Normal vs Cataract)
- Severity grading (LOCS III: 0-6)
- XAI-friendly architecture (works with SHAP, LIME, Grad-CAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class MobileNetBaseline(nn.Module):
    """
    Simple MobileNetV2-based cataract classifier.
    
    No attention mechanism - serves as baseline for GLAAM comparison.
    
    Architecture:
        Input (3, 224, 224)
            ↓
        MobileNetV2 Backbone
            ↓
        Global Average Pooling
            ↓
        Classifier Heads
            ↓
        ├── Binary: Normal / Cataract
        └── Severity: LOCS III (0-6)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_severity: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of binary classes (default: 2 for normal/cataract)
            num_severity: Number of severity grades (default: 7 for LOCS 0-6)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
        """
        super(MobileNetBaseline, self).__init__()
        
        # Load pretrained MobileNetV2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Get feature dimension (1280 for MobileNetV2)
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Severity grading head
        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_severity)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            dict with:
                - 'binary_logits': (B, 2)
                - 'severity_logits': (B, 7)
                - 'features': (B, 1280) for XAI visualization
        """
        # Extract features
        features = self.backbone(x)
        features = self.dropout(features)
        
        # Classification heads
        binary_logits = self.binary_head(features)
        severity_logits = self.severity_head(features)
        
        return {
            'binary_logits': binary_logits,
            'severity_logits': severity_logits,
            'features': features
        }
    
    def predict_binary(self, x: torch.Tensor) -> torch.Tensor:
        """For SHAP/LIME - returns binary probabilities."""
        output = self.forward(x)
        return F.softmax(output['binary_logits'], dim=1)
    
    def predict_severity(self, x: torch.Tensor) -> torch.Tensor:
        """For SHAP/LIME - returns severity probabilities."""
        output = self.forward(x)
        return F.softmax(output['severity_logits'], dim=1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for visualization."""
        return self.backbone(x)


class SimpleLoss(nn.Module):
    """
    Simple multi-task loss for baseline model.
    """
    
    def __init__(
        self,
        binary_weight: float = 1.0,
        severity_weight: float = 1.0,
        severity_class_weights: Optional[torch.Tensor] = None
    ):
        super(SimpleLoss, self).__init__()
        
        self.binary_weight = binary_weight
        self.severity_weight = severity_weight
        
        self.binary_criterion = nn.CrossEntropyLoss()
        
        if severity_class_weights is not None:
            self.severity_criterion = nn.CrossEntropyLoss(weight=severity_class_weights)
        else:
            self.severity_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        binary_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        severity_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        binary_loss = self.binary_criterion(binary_logits, binary_targets)
        severity_loss = self.severity_criterion(severity_logits, severity_targets)
        
        total_loss = self.binary_weight * binary_loss + self.severity_weight * severity_loss
        
        return {
            'loss': total_loss,
            'binary_loss': binary_loss,
            'severity_loss': severity_loss
        }


# Test
if __name__ == "__main__":
    print("Testing MobileNetBaseline...")
    
    model = MobileNetBaseline()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Binary logits: {output['binary_logits'].shape}")
    print(f"Severity logits: {output['severity_logits'].shape}")
    print(f"Features: {output['features'].shape}")
    
    # Test loss
    criterion = SimpleLoss()
    binary_targets = torch.randint(0, 2, (4,))
    severity_targets = torch.randint(0, 7, (4,))
    
    losses = criterion(
        output['binary_logits'],
        output['severity_logits'],
        binary_targets,
        severity_targets
    )
    
    print(f"\nTotal loss: {losses['loss'].item():.4f}")
    print(f"Binary loss: {losses['binary_loss'].item():.4f}")
    print(f"Severity loss: {losses['severity_loss'].item():.4f}")
