"""
YOLO-GLAAM Hybrid Model for Cataract Detection and Severity Grading.

This is the main innovation model that combines:
1. YOLOv8-nano for lens ROI detection
2. MobileNetV2 + GLAAM/GLAAI attention for feature extraction
3. Multi-task heads for binary classification + LOCS III severity grading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backbones.backbone import get_backbone, MobileNetV2WithAttention
from yolo.lens_detector import LensDetector


class MultiTaskHead(nn.Module):
    """
    Multi-task classification head for:
    1. Binary classification (Cataract vs Normal)
    2. LOCS III severity grading (0-6)
    """
    
    def __init__(
        self,
        in_features: int,
        num_severity_classes: int = 7,
        dropout: float = 0.3
    ):
        """
        Args:
            in_features: Number of input features from backbone
            num_severity_classes: Number of LOCS III grades (0-6 = 7 classes)
            dropout: Dropout rate
        """
        super(MultiTaskHead, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # [Normal, Cataract]
        )
        
        # Severity grading head (LOCS III)
        self.severity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_severity_classes)  # [0, 1, 2, 3, 4, 5, 6]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, in_features)
        Returns:
            binary_logits: (B, 2) - Cataract vs Normal
            severity_logits: (B, 7) - LOCS III grade
        """
        x = self.dropout(x)
        shared = self.shared(x)
        
        binary_logits = self.binary_head(shared)
        severity_logits = self.severity_head(shared)
        
        return binary_logits, severity_logits


class HybridCataractModel(nn.Module):
    """
    YOLO-GLAAM Hybrid Model for Cataract Detection and Severity Grading.
    
    Architecture:
        Slit-lamp Image (3264×2448)
                ↓
        [YOLOv8-nano] → Lens ROI Detection
                ↓
        Cropped ROI (384×384)
                ↓
        [MobileNetV2 + GLAAM] → Feature Extraction
                ↓
        [Multi-Task Head]
                ↓
        ├── Binary: Cataract / Normal
        └── Severity: LOCS III (0-6)
    
    This model can operate in two modes:
    1. End-to-end mode: Takes full slit-lamp images, runs YOLO, then classification
    2. ROI-only mode: Takes pre-cropped ROIs, skips YOLO (for training)
    """
    
    def __init__(
        self,
        backbone: str = "mobilenetv2",
        attention_type: Literal["glaam", "glaai"] = "glaam",
        pretrained: bool = True,
        num_severity_classes: int = 7,
        dropout: float = 0.3,
        yolo_model_path: Optional[str] = None,
        yolo_weights: Optional[str] = None,  # Alias for yolo_model_path
        use_yolo: bool = True
    ):
        """
        Args:
            backbone: Backbone network ("mobilenetv2" or "inceptionv3")
            attention_type: Attention module ("glaam" or "glaai")
            pretrained: Use pretrained backbone
            num_severity_classes: Number of LOCS III grades
            dropout: Dropout rate
            yolo_model_path: Path to trained YOLO model (None uses pretrained)
            yolo_weights: Alias for yolo_model_path (for consistency)
            use_yolo: Whether to use YOLO for ROI detection
        """
        super(HybridCataractModel, self).__init__()
        
        self.use_yolo = use_yolo
        
        # YOLO Lens Detector
        if use_yolo:
            # Accept either yolo_model_path or yolo_weights (yolo_weights takes precedence)
            yolo_path = yolo_weights or yolo_model_path
            self.lens_detector = LensDetector(model_path=yolo_path)
        else:
            self.lens_detector = None
        
        # Backbone with attention
        self.backbone = get_backbone(
            name=backbone,
            attention_type=attention_type,
            pretrained=pretrained
        )
        
        # Multi-task classification head
        self.head = MultiTaskHead(
            in_features=self.backbone.feature_dim,
            num_severity_classes=num_severity_classes,
            dropout=dropout
        )
        
        # Store config for export
        self.config = {
            "backbone": backbone,
            "attention_type": attention_type,
            "num_severity_classes": num_severity_classes
        }
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pre-cropped ROI images.
        Use this for training where images are already cropped.
        
        Args:
            x: Input tensor (B, 3, H, W) - pre-cropped ROIs
            return_attention: Whether to return attention feature maps
            
        Returns:
            dict with:
                - 'binary_logits': (B, 2)
                - 'severity_logits': (B, num_severity_classes)
                - 'attention_features': (B, C, H', W') if return_attention=True
        """
        # Feature extraction with attention
        features, attention_features = self.backbone(x)
        
        # Multi-task classification
        binary_logits, severity_logits = self.head(features)
        
        output = {
            'binary_logits': binary_logits,
            'severity_logits': severity_logits
        }
        
        if return_attention:
            output['attention_features'] = attention_features
        
        return output
    
    def predict_from_full_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (384, 384),
        device: Optional[str] = None
    ) -> Dict:
        """
        End-to-end prediction from full slit-lamp image.
        
        Args:
            image: Full slit-lamp image as numpy array (H, W, 3)
            target_size: Size to resize ROI to
            device: Device for inference
            
        Returns:
            dict with:
                - 'binary_pred': 0 (Normal) or 1 (Cataract)
                - 'binary_prob': Probability of cataract
                - 'severity_pred': LOCS III grade (0-6)
                - 'severity_probs': Probabilities for each grade
                - 'bbox': Detected lens bounding box
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Step 1: YOLO detection and crop
        if self.use_yolo and self.lens_detector is not None:
            roi, bbox = self.lens_detector.detect_and_crop(
                image,
                target_size=target_size,
                return_original_if_no_detection=True
            )
        else:
            import cv2
            roi = cv2.resize(image, target_size)
            bbox = None
        
        # Step 2: Preprocess for PyTorch
        roi_tensor = torch.from_numpy(roi).float() / 255.0
        roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        roi_tensor = (roi_tensor - mean) / std
        roi_tensor = roi_tensor.to(device)
        
        # Step 3: Forward pass
        self.eval()
        with torch.no_grad():
            output = self.forward(roi_tensor)
        
        # Step 4: Post-process
        binary_probs = F.softmax(output['binary_logits'], dim=1)[0]
        severity_probs = F.softmax(output['severity_logits'], dim=1)[0]
        
        return {
            'binary_pred': binary_probs.argmax().item(),
            'binary_prob': binary_probs[1].item(),  # Probability of cataract
            'severity_pred': severity_probs.argmax().item(),
            'severity_probs': severity_probs.cpu().numpy(),
            'bbox': bbox
        }
    
    def export_onnx(
        self,
        save_path: str,
        input_size: Tuple[int, int] = (384, 384),
        opset_version: int = 14
    ):
        """
        Export model to ONNX format for mobile deployment.
        Note: This exports only the backbone+head, not YOLO detector.
        """
        import torch.onnx
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # Temporarily disable YOLO for export
        original_use_yolo = self.use_yolo
        self.use_yolo = False
        
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['binary_logits', 'severity_logits'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'binary_logits': {0: 'batch_size'},
                'severity_logits': {0: 'batch_size'}
            }
        )
        
        self.use_yolo = original_use_yolo
        print(f"Model exported to {save_path}")


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    Weighted sum of binary classification loss and severity grading loss.
    
    CRITICAL: Supports class weights for handling LOCS III class imbalance.
    Without class weights, the model will ignore minority classes.
    """
    
    def __init__(
        self,
        binary_weight: float = 1.0,
        severity_weight = 1.0,  # Can be float or Tensor of class weights
        label_smoothing: float = 0.1
    ):
        """
        Args:
            binary_weight: Weight for binary classification loss (scalar)
            severity_weight: Weight for severity loss - can be:
                - float: simple scalar weight
                - Tensor of shape (num_classes,): per-class weights for imbalance
            label_smoothing: Label smoothing factor
        """
        super(MultiTaskLoss, self).__init__()
        
        self.binary_weight = binary_weight
        
        # Check if severity_weight is a tensor (class weights) or scalar
        if isinstance(severity_weight, torch.Tensor):
            # severity_weight is a tensor of class weights
            self.severity_task_weight = 1.0
            self.severity_criterion = nn.CrossEntropyLoss(
                weight=severity_weight, 
                label_smoothing=label_smoothing
            )
        else:
            # severity_weight is a scalar
            self.severity_task_weight = severity_weight
            self.severity_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.binary_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        binary_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        severity_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            binary_logits: (B, 2)
            severity_logits: (B, num_classes)
            binary_targets: (B,) with values in {0, 1}
            severity_targets: (B,) with values in {0, 1, ..., 6}
            
        Returns:
            dict with 'loss', 'binary_loss', 'severity_loss'
        """
        binary_loss = self.binary_criterion(binary_logits, binary_targets)
        severity_loss = self.severity_criterion(severity_logits, severity_targets)
        
        total_loss = (
            self.binary_weight * binary_loss +
            self.severity_task_weight * severity_loss
        )
        
        return {
            'loss': total_loss,
            'binary_loss': binary_loss,
            'severity_loss': severity_loss
        }


# For testing
if __name__ == "__main__":
    print("Testing HybridCataractModel...")
    
    # Create model (without YOLO for basic test)
    model = HybridCataractModel(
        backbone="mobilenetv2",
        attention_type="glaam",
        pretrained=True,
        use_yolo=False  # Skip YOLO for this test
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 384, 384)
    
    output = model(x, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Binary logits: {output['binary_logits'].shape}")
    print(f"Severity logits: {output['severity_logits'].shape}")
    print(f"Attention features: {output['attention_features'].shape}")
    
    # Test loss
    criterion = MultiTaskLoss()
    binary_targets = torch.randint(0, 2, (batch_size,))
    severity_targets = torch.randint(0, 7, (batch_size,))
    
    losses = criterion(
        output['binary_logits'],
        output['severity_logits'],
        binary_targets,
        severity_targets
    )
    
    print(f"\nTotal loss: {losses['loss'].item():.4f}")
    print(f"Binary loss: {losses['binary_loss'].item():.4f}")
    print(f"Severity loss: {losses['severity_loss'].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {param_size / 1024 / 1024:.2f} MB")
