"""
Multi-Task MobileNet for Multi-Disease Detection.

Predicts ALL 7 ODIR diseases simultaneously:
- Cataract (C)
- Diabetic Retinopathy (D)
- Glaucoma (G)
- AMD (A)
- Hypertension (H)
- Myopia (M)
- Others (O)

This is clinically more useful than single-disease models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class MobileNetMultiTask(nn.Module):
    """
    Multi-Task MobileNetV2 for 7-disease classification.
    
    Each disease gets its own classification head.
    Shared backbone learns general retina features.
    """
    
    DISEASES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 
                'amd', 'hypertension', 'myopia', 'others']
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Shared backbone
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 7 disease-specific heads (binary classification each)
        self.heads = nn.ModuleDict({
            disease: nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2)  # Binary: present/absent
            )
            for disease in self.DISEASES
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            
        Returns:
            dict with logits for each disease
        """
        # Shared feature extraction
        features = self.features(x)
        features = self.pool(features).flatten(1)
        features = self.dropout(features)
        
        # Disease-specific predictions
        outputs = {
            disease: head(features) 
            for disease, head in self.heads.items()
        }
        
        # Add combined features for XAI
        outputs['features'] = features
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, Dict]:
        """
        Get predictions with probabilities.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        
        results = {}
        for disease in self.DISEASES:
            probs = F.softmax(outputs[disease], dim=1)
            pred = probs.argmax(dim=1)
            results[disease] = {
                'prediction': pred.item(),
                'probability': probs[0, 1].item(),  # P(disease present)
                'label': 'Present' if pred.item() == 1 else 'Absent'
            }
        
        return results


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for 7 diseases.
    
    Supports per-disease class weights for imbalanced data.
    """
    
    def __init__(self, disease_weights: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        
        self.diseases = MobileNetMultiTask.DISEASES
        self.disease_weights = disease_weights or {}
        
        # Create criterion for each disease
        self.criteria = nn.ModuleDict()
        for disease in self.diseases:
            if disease in self.disease_weights:
                self.criteria[disease] = nn.CrossEntropyLoss(
                    weight=self.disease_weights[disease]
                )
            else:
                self.criteria[disease] = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        """
        losses = {}
        total_loss = 0.0
        
        for disease in self.diseases:
            if disease in predictions and disease in targets:
                loss = self.criteria[disease](predictions[disease], targets[disease])
                losses[f'{disease}_loss'] = loss
                total_loss += loss
        
        losses['total_loss'] = total_loss
        
        return losses


class BayesianMultiTask(MobileNetMultiTask):
    """
    Multi-Task model with MC Dropout for uncertainty estimation.
    
    At inference, samples multiple predictions to estimate uncertainty.
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3, n_samples: int = 10):
        super().__init__(pretrained=pretrained, dropout=dropout)
        self.n_samples = n_samples
    
    def forward_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Forward pass with uncertainty estimation.
        
        Uses MC Dropout to sample multiple predictions.
        """
        n_samples = n_samples or self.n_samples
        
        # Enable dropout during inference
        self.train()  # Keep dropout active
        
        all_predictions = {disease: [] for disease in self.DISEASES}
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = super().forward(x)
                for disease in self.DISEASES:
                    probs = F.softmax(outputs[disease], dim=1)
                    all_predictions[disease].append(probs)
        
        # Compute mean and variance
        results = {}
        for disease in self.DISEASES:
            stacked = torch.stack(all_predictions[disease])
            mean_prob = stacked.mean(dim=0)
            var_prob = stacked.var(dim=0)
            
            pred = mean_prob.argmax(dim=1)
            uncertainty = var_prob.mean().item()
            
            results[disease] = {
                'prediction': pred.item(),
                'probability': mean_prob[0, 1].item(),
                'uncertainty': uncertainty,
                'confident': uncertainty < 0.05,  # Threshold
                'label': 'Present' if pred.item() == 1 else 'Absent'
            }
        
        return results


# ========== TEST ==========
if __name__ == "__main__":
    print("Testing MobileNetMultiTask...")
    
    # Create model
    model = MobileNetMultiTask()
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    print(f"\nOutput shapes:")
    for disease in model.DISEASES:
        print(f"  {disease}: {outputs[disease].shape}")
    print(f"  features: {outputs['features'].shape}")
    
    # Test Bayesian
    print("\nTesting BayesianMultiTask...")
    bayesian = BayesianMultiTask(n_samples=5)
    
    x_single = torch.randn(1, 3, 224, 224)
    results = bayesian.forward_with_uncertainty(x_single, n_samples=5)
    
    print("\nPredictions with uncertainty:")
    for disease, result in results.items():
        conf = "✓" if result['confident'] else "⚠️"
        print(f"  {disease}: {result['label']} ({result['probability']:.1%}) "
              f"± {result['uncertainty']:.4f} {conf}")
