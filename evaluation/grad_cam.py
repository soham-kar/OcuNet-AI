"""
Grad-CAM visualization for interpretability.

Generates attention heatmaps showing where the model focuses for cataract detection.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HybridCataractModel


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    
    def __init__(self, model: HybridCataractModel, target_layer: str = None):
        """
        Args:
            model: The trained model
            target_layer: Name of the layer to visualize (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Hook storage
        self.gradients = None
        self.activations = None
        
        # Register hooks on the attention layer
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Hook onto the GLAAM attention module
        if hasattr(self.model.backbone, 'attention'):
            self.model.backbone.attention.register_forward_hook(forward_hook)
            self.model.backbone.attention.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        image: torch.Tensor,
        target_class: int = None,
        task: str = 'binary'
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Target class for gradient computation (None = predicted class)
            task: 'binary' or 'severity'
            
        Returns:
            heatmap: Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(image, return_attention=True)
        
        # Get logits for the specified task
        if task == 'binary':
            logits = output['binary_logits']
        else:
            logits = output['severity_logits']
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        image_path: str,
        save_path: str = None,
        task: str = 'binary',
        alpha: float = 0.5
    ):
        """
        Generate and visualize Grad-CAM overlay.
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization (None = display)
            task: 'binary' or 'severity'
            alpha: Overlay transparency
        """
        from torchvision import transforms
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        original_np = np.array(original_image)
        
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(original_image).unsqueeze(0)
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Generate CAM
        cam = self.generate(image_tensor, task=task)
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_np.shape[1], original_np.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (1 - alpha) * original_np + alpha * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            if task == 'binary':
                pred = output['binary_logits'].argmax(dim=1).item()
                prob = F.softmax(output['binary_logits'], dim=1)[0, 1].item()
                title = f"{'Cataract' if pred == 1 else 'Normal'} (prob: {prob:.3f})"
            else:
                pred = output['severity_logits'].argmax(dim=1).item()
                title = f"LOCS III Grade: {pred}"
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay - {title}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'severity'])
    args = parser.parse_args()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = HybridCataractModel(
        backbone='mobilenetv2',
        attention_type='glaam',
        use_yolo=False
    )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    gradcam.visualize(
        image_path=args.image,
        save_path=args.output,
        task=args.task
    )


if __name__ == "__main__":
    main()
