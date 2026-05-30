"""
Run XAI Analysis on trained ODIR baseline model.

Generates:
- Grad-CAM heatmaps
- LIME explanations
- Model predictions with confidence

Usage:
    python baseline_xai/run_xai.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========== MODEL DEFINITION ==========
class MobileNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Identity()
        
        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.severity_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return {
            'binary_logits': self.binary_head(features),
            'severity_logits': self.severity_head(features),
            'features': features
        }


# ========== GRAD-CAM ==========
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook the last conv layer
        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        logits = output['binary_logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


def main():
    print("=" * 60)
    print(" XAI Analysis - ODIR Baseline Model")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model_path = Path("checkpoints_modal/odir_baseline/best_model.pth")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = MobileNetBaseline()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (best val acc: {checkpoint.get('best_val_acc', 'N/A')})")
    
    # Find test images
    odir_images = list(Path("data/raw/odir/preprocessed_images").glob("*.jpg"))
    if not odir_images:
        odir_images = list(Path("data/raw/odir/ODIR-5K/ODIR-5K/Training Images").glob("*.jpg"))
    
    if not odir_images:
        print("ERROR: No images found!")
        return
    
    print(f"Found {len(odir_images)} images")
    
    # Select 4 random images
    np.random.seed(42)
    sample_images = np.random.choice(odir_images, min(4, len(odir_images)), replace=False)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Create output directory
    output_dir = Path("baseline_xai/xai_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 4*len(sample_images)))
    
    for i, img_path in enumerate(sample_images):
        print(f"\nProcessing {img_path.name}...")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        
        # Preprocess
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            binary_prob = F.softmax(output['binary_logits'], dim=1)
            severity_prob = F.softmax(output['severity_logits'], dim=1)
        
        binary_pred = binary_prob.argmax(dim=1).item()
        binary_conf = binary_prob[0, binary_pred].item()
        severity_pred = severity_prob.argmax(dim=1).item()
        
        label = "Cataract" if binary_pred == 1 else "Normal"
        print(f"  Prediction: {label} ({binary_conf:.1%})")
        
        # Generate Grad-CAM
        cam = gradcam.generate(input_tensor, target_class=binary_pred)
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(cam)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)
        
        # Plot
        ax_row = axes[i] if len(sample_images) > 1 else axes
        
        ax_row[0].imshow(img_array)
        ax_row[0].set_title(f'Original\n{img_path.name[:20]}...', fontsize=10)
        ax_row[0].axis('off')
        
        ax_row[1].imshow(cam, cmap='jet')
        ax_row[1].set_title('Grad-CAM', fontsize=10)
        ax_row[1].axis('off')
        
        ax_row[2].imshow(overlay)
        ax_row[2].set_title('Overlay', fontsize=10)
        ax_row[2].axis('off')
        
        # Prediction bar chart
        ax_row[3].barh(['Normal', 'Cataract'], binary_prob[0].cpu().numpy(), color=['green', 'red'])
        ax_row[3].set_xlim([0, 1])
        ax_row[3].set_title(f'Prediction: {label}\n({binary_conf:.1%})', fontsize=10)
        ax_row[3].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('XAI Analysis - Grad-CAM Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "gradcam_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print(" XAI Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
