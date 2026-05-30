"""
GLAAM Bayesian Model with Monte Carlo Dropout for Uncertainty Quantification.
Based on: Week 5 of 10-week roadmap - Add uncertainty to flag low-confidence predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention.glaam import GLAAM
from backbones.backbone import MobileNetV2WithAttention


class MCDropout(nn.Dropout):
    """
    Monte Carlo Dropout that stays ON during inference.
    Used for Bayesian uncertainty estimation.
    """
    def forward(self, x):
        # Always apply dropout (even during eval)
        return F.dropout(x, self.p, training=True)


class GLAAM_Bayesian(nn.Module):
    """
    GLAAM model with Bayesian uncertainty estimation via Monte Carlo Dropout.
    
    During inference, run forward pass N times with dropout ON to estimate:
    - Mean prediction
    - Prediction variance (uncertainty)
    
    Use cases:
    - Flag low-confidence predictions (variance > threshold)
    - Identify borderline LOCS grades (2-3) for specialist referral
    """
    
    def __init__(
        self,
        num_classes: int = 7,  # LOCS 0-6
        dropout_rate: float = 0.2,
        pretrained: bool = True
    ):
        super(GLAAM_Bayesian, self).__init__()
        
        # Backbone with attention
        self.backbone = MobileNetV2WithAttention(
            pretrained=pretrained,
            attention_type="glaam"
        )
        
        # Replace standard dropout with MC Dropout
        self.classifier = nn.Sequential(
            MCDropout(dropout_rate),
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            MCDropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (single sample).
        """
        features, _ = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
        return_all_samples: bool = False
    ) -> dict:
        """
        Run N forward passes with MC Dropout to estimate uncertainty.
        
        Args:
            x: Input tensor (B, 3, H, W)
            n_samples: Number of MC samples
            return_all_samples: If True, return all individual predictions
            
        Returns:
            dict with:
                - 'mean_probs': Mean probability across samples (B, num_classes)
                - 'variance': Variance across samples (B, num_classes)
                - 'prediction': Predicted class (B,)
                - 'confidence': Confidence of prediction (B,)
                - 'uncertainty_score': Overall uncertainty (B,)
        """
        self.train()  # Keep dropout active
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Stack: (n_samples, B, num_classes)
        all_probs = torch.stack(all_probs, dim=0)
        
        # Compute statistics
        mean_probs = all_probs.mean(dim=0)  # (B, num_classes)
        variance = all_probs.var(dim=0)     # (B, num_classes)
        
        # Prediction and confidence
        prediction = mean_probs.argmax(dim=1)  # (B,)
        confidence = mean_probs.max(dim=1).values  # (B,)
        
        # Overall uncertainty score (mean variance across classes)
        uncertainty_score = variance.mean(dim=1)  # (B,)
        
        result = {
            'mean_probs': mean_probs,
            'variance': variance,
            'prediction': prediction,
            'confidence': confidence,
            'uncertainty_score': uncertainty_score
        }
        
        if return_all_samples:
            result['all_probs'] = all_probs
        
        return result
    
    def get_uncertainty_threshold(
        self,
        val_loader,
        target_flagging_rate: float = 0.1,
        n_samples: int = 10
    ) -> float:
        """
        Compute uncertainty threshold to flag top X% uncertain predictions.
        
        Args:
            val_loader: Validation data loader
            target_flagging_rate: Fraction of samples to flag (default 10%)
            n_samples: Number of MC samples
            
        Returns:
            Uncertainty threshold value
        """
        all_uncertainties = []
        
        device = next(self.parameters()).device
        
        for batch in val_loader:
            images = batch['image'].to(device)
            result = self.predict_with_uncertainty(images, n_samples=n_samples)
            all_uncertainties.extend(result['uncertainty_score'].cpu().numpy())
        
        all_uncertainties = np.array(all_uncertainties)
        threshold = np.percentile(all_uncertainties, (1 - target_flagging_rate) * 100)
        
        return threshold


class DualViewGLAAM_Bayesian(nn.Module):
    """
    Dual-view GLAAM model for 45° + 135° slit-lamp fusion.
    
    Combines features from both viewing angles for robust prediction.
    Inspired by clinical practice where ophthalmologists examine from multiple angles.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.2,
        fusion_method: str = "concat",  # 'concat', 'add', 'attention'
        pretrained: bool = True
    ):
        super(DualViewGLAAM_Bayesian, self).__init__()
        
        # Two separate backbones for each view
        self.backbone_45 = MobileNetV2WithAttention(
            pretrained=pretrained,
            attention_type="glaam"
        )
        self.backbone_135 = MobileNetV2WithAttention(
            pretrained=pretrained,
            attention_type="glaam"
        )
        
        self.fusion_method = fusion_method
        feature_dim = self.backbone_45.feature_dim
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Identity()
            fused_dim = feature_dim * 2
        elif fusion_method == "add":
            self.fusion = nn.Identity()
            fused_dim = feature_dim
        elif fusion_method == "attention":
            self.fusion_attention = nn.Sequential(
                nn.Linear(feature_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )
            fused_dim = feature_dim
        
        # Classifier with MC Dropout
        self.classifier = nn.Sequential(
            MCDropout(dropout_rate),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            MCDropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
    
    def forward(
        self,
        x_45: torch.Tensor,
        x_135: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both views.
        
        Args:
            x_45: 45° view images (B, 3, H, W)
            x_135: 135° view images (B, 3, H, W)
        """
        # Extract features from both views
        feat_45, _ = self.backbone_45(x_45)
        feat_135, _ = self.backbone_135(x_135)
        
        # Fuse features
        if self.fusion_method == "concat":
            fused = torch.cat([feat_45, feat_135], dim=1)
        elif self.fusion_method == "add":
            fused = feat_45 + feat_135
        elif self.fusion_method == "attention":
            combined = torch.cat([feat_45, feat_135], dim=1)
            weights = self.fusion_attention(combined)  # (B, 2)
            fused = weights[:, 0:1] * feat_45 + weights[:, 1:2] * feat_135
        
        logits = self.classifier(fused)
        return logits
    
    def predict_with_uncertainty(
        self,
        x_45: torch.Tensor,
        x_135: torch.Tensor,
        n_samples: int = 10
    ) -> dict:
        """
        Bayesian prediction with uncertainty from both views.
        """
        self.train()  # Keep dropout active
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x_45, x_135)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=0)
        
        mean_probs = all_probs.mean(dim=0)
        variance = all_probs.var(dim=0)
        prediction = mean_probs.argmax(dim=1)
        confidence = mean_probs.max(dim=1).values
        uncertainty_score = variance.mean(dim=1)
        
        return {
            'mean_probs': mean_probs,
            'variance': variance,
            'prediction': prediction,
            'confidence': confidence,
            'uncertainty_score': uncertainty_score
        }


# For testing
if __name__ == "__main__":
    print("Testing Bayesian GLAAM...")
    
    # Test single-view Bayesian model
    model = GLAAM_Bayesian(num_classes=7, dropout_rate=0.2, pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    
    # Single forward pass
    logits = model(x)
    print(f"Single pass output: {logits.shape}")
    
    # Uncertainty estimation
    result = model.predict_with_uncertainty(x, n_samples=10)
    print(f"Mean probs: {result['mean_probs'].shape}")
    print(f"Uncertainty scores: {result['uncertainty_score']}")
    
    # Test dual-view model
    print("\nTesting Dual-View Bayesian GLAAM...")
    dual_model = DualViewGLAAM_Bayesian(num_classes=7, fusion_method="concat", pretrained=False)
    x_45 = torch.randn(2, 3, 384, 384)
    x_135 = torch.randn(2, 3, 384, 384)
    
    dual_logits = dual_model(x_45, x_135)
    print(f"Dual-view output: {dual_logits.shape}")
    
    dual_result = dual_model.predict_with_uncertainty(x_45, x_135, n_samples=10)
    print(f"Dual-view uncertainty: {dual_result['uncertainty_score']}")
    
    # Parameter count
    params = sum(p.numel() for p in dual_model.parameters())
    print(f"\nDual-view parameters: {params:,} (~{params/1e6:.1f}M)")
