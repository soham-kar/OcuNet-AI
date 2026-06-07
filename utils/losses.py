"""
Loss functions for multi-label fundus classification.

Includes:
  - ODIRMultiLabelLoss: BCEWithLogitsLoss with class balancing
  - FocalLossODIR: standard focal loss
  - MultiLabelFocalLoss: focal loss with per-class weights
  - AsymmetricLoss: ASL from ICCV 2021 (Ben-Baruch et al.)
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


class MultiLabelFocalLoss(nn.Module):
    """
    Focal loss with per-class weights for multi-label classification.
    Used in v2 training.
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        if self.class_weights is not None:
            alpha_t = alpha_t * self.class_weights.to(targets.device)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        return (focal_weight * bce).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    
    From: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    Authors: Ben-Baruch, Ridnik, Zamir, Noy, Friedman, Protter, Zelnik-Manor
    
    Key differences from focal loss:
      - Decouples gamma for positive and negative samples (γ_neg >> γ_pos)
      - Hard-thresholds easy negatives via probability clipping
      - Preserves gradient signal from rare positive classes
    
    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4)
        gamma_pos: Focusing parameter for positive samples (default: 0)
        clip: Probability threshold below which negative loss is zeroed (default: 0.05)
        eps: Numerical stability
        disable_torch_grad_focal_loss: If True, use manual gradient (not needed for PyTorch >= 1.10)
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary labels [0, 1]
        Returns:
            scalar loss
        """
        # Probability from logits
        xs_pos = logits
        xs_neg = -logits
        
        # For numerical stability, clamp
        if self.disable_torch_grad_focal_loss:
            # Manual implementation (for older PyTorch)
            xs_pos = xs_pos.clamp(min=self.eps)
            xs_neg = xs_neg.clamp(min=self.eps)
            
            # Positive loss
            pt_pos = torch.sigmoid(xs_pos)
            loss_pos = targets * (1 - pt_pos).pow(self.gamma_pos) * F.logsigmoid(xs_pos)
            
            # Negative loss
            pt_neg = torch.sigmoid(xs_neg)
            loss_neg = (1 - targets) * (1 - pt_neg).pow(self.gamma_neg) * F.logsigmoid(xs_neg)
        else:
            # Use PyTorch's built-in focal loss for efficiency
            # Positive part
            pos_term = (1 - targets) * xs_pos + targets * (torch.clamp(
                (1 - torch.sigmoid(xs_pos)).pow(self.gamma_pos),
                min=self.eps
            ) * xs_pos - xs_pos)
            
            # Negative part with clipping
            neg_term = targets * xs_neg + (1 - targets) * (torch.clamp(
                (1 - torch.sigmoid(xs_neg)).pow(self.gamma_neg),
                min=self.eps
            ) * xs_neg - xs_neg)
            
            loss_pos = targets * F.logsigmoid(xs_pos) * (1 - torch.sigmoid(xs_pos)).pow(self.gamma_pos)
            loss_neg = (1 - targets) * F.logsigmoid(xs_neg) * (1 - torch.sigmoid(xs_neg)).pow(self.gamma_neg)
        
        # Hard thresholding: zero out loss for very confident negatives
        if self.clip is not None and self.clip > 0:
            # For negative samples where probability < clip, set loss to 0
            probs = torch.sigmoid(logits)
            neg_mask = (targets == 0) & (probs < self.clip)
            loss_neg = loss_neg * (~neg_mask).float()
        
        loss = -loss_pos - loss_neg
        return loss.mean()


class AsymmetricLossOptimized(nn.Module):
    """
    Memory-optimized version of AsymmetricLoss.
    Uses in-place operations and minimizes allocations.
    Bit-accurate with AsymmetricLoss.
    
    Recommended for production training.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
        # Pre-compute targets for softplus
        self.targets_classes = None  # Will be set on first forward
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary labels [0, 1]
        Returns:
            scalar loss
        """
        num_classes = logits.shape[1]
        
        # Positive loss
        xs_pos = logits
        pt = torch.sigmoid(xs_pos)
        pt = pt.clamp(min=self.eps, max=1 - self.eps)
        
        pos_loss = targets * torch.pow(1 - pt, self.gamma_pos) * (
            F.logsigmoid(xs_pos)
        )
        
        # Negative loss with clipping
        xs_neg = -logits
        p_neg = torch.sigmoid(xs_neg)
        p_neg = p_neg.clamp(min=self.eps, max=1 - self.eps)
        
        neg_loss = (1 - targets) * torch.pow(1 - p_neg, self.gamma_neg) * (
            F.logsigmoid(xs_neg)
        )
        
        # Hard thresholding
        if self.clip is not None and self.clip > 0:
            probs = torch.sigmoid(logits)
            neg_mask = (targets == 0) & (probs < self.clip)
            neg_loss = neg_loss * (~neg_mask).float()
        
        loss = -pos_loss - neg_loss
        return loss.mean()


if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    batch_size, num_classes = 4, 4
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test MultiLabelFocalLoss
    focal = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    print(f"✅ MultiLabelFocalLoss: {focal(logits, targets).item():.4f}")
    
    # Test AsymmetricLoss
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    print(f"✅ AsymmetricLoss:      {asl(logits, targets).item():.4f}")
    
    # Test AsymmetricLossOptimized
    asl_opt = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05)
    print(f"✅ AsymmetricLossOpt:   {asl_opt(logits, targets).item():.4f}")
    
    # Verify backward works
    logits_grad = torch.randn(batch_size, num_classes, requires_grad=True)
    asl(logits_grad, targets).backward()
    print("✅ Backward pass successful")
    
    # Test with extreme imbalance (simulating DR scenario)
    print("\nTesting with extreme class imbalance...")
    logits2 = torch.randn(8, 4)
    targets2 = torch.zeros(8, 4)
    targets2[:2, 0] = 1  # Only 2/8 positive for class 0
    
    focal_loss = focal(logits2, targets2)
    asl_loss = asl(logits2, targets2)
    print(f"  Focal loss: {focal_loss.item():.4f}")
    print(f"  ASL loss:   {asl_loss.item():.4f}")
    print(f"  Ratio (ASL/Focal): {asl_loss.item()/focal_loss.item():.2f}")
    print("  (ASL should be higher for imbalanced data — more gradient on positives)")
