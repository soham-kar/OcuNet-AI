"""
Loss functions for ODIR-5K multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODIRMultiLabelLoss(nn.Module):
    """
    BCEWithLogitsLoss with class balancing for ODIR-5K diseases
    
    Args:
        pos_weights: Positive weights for each disease (adjust for class imbalance)
                    Default: [cataract, DR, glaucoma, AMD, hypertension, myopia, others]
    """
    
    def __init__(self, pos_weights=None, reduction='mean'):
        super().__init__()
        
        # Default pos_weights based on ODIR-5K typical class distribution
        # Increase weight for rare diseases (DR, Glaucoma, AMD)
        if pos_weights is None:
            self.pos_weights = torch.tensor([2.0, 5.0, 4.0, 6.0, 3.0, 2.0, 1.5])
        else:
            self.pos_weights = torch.tensor(pos_weights)
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weights,
            reduction=reduction
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 7) - model predictions
            targets: (B, 7) - binary labels [0,1]
        """
        return self.criterion(logits, targets)


class FocalLossODIR(nn.Module):
    """
    Focal loss variant for severe class imbalance in ODIR-5K
    Useful for "Others" category which is poorly represented
    """
    
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return loss.mean()


if __name__ == "__main__":
    print("Testing ODIR Loss Functions...")
    
    # Test data
    batch_size = 4
    logits = torch.randn(batch_size, 7)
    targets = torch.randint(0, 2, (batch_size, 7)).float()
    
    # Test ODIRMultiLabelLoss
    criterion = ODIRMultiLabelLoss()
    loss = criterion(logits, targets)
    print(f"✅ ODIRMultiLabelLoss: {loss.item():.4f}")
    
    # Test FocalLoss
    focal_criterion = FocalLossODIR()
    focal_loss = focal_criterion(logits, targets)
    print(f"✅ FocalLossODIR: {focal_loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("✅ Backward pass successful")
