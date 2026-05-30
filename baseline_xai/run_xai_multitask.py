"""
XAI Analysis for Multi-Task 7-Disease Model.

Generates Grad-CAM visualizations for each disease:
- Cataract, Diabetic Retinopathy, Glaucoma, AMD, Hypertension, Myopia, Others

Usage:
    python baseline_xai/run_xai_multitask.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


DISEASE_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 
                 'amd', 'hypertension', 'myopia', 'others']

DISEASE_COLORS = {
    'cataract': '#FF6B6B',
    'diabetic_retinopathy': '#4ECDC4',
    'glaucoma': '#45B7D1',
    'amd': '#96CEB4',
    'hypertension': '#FFEAA7',
    'myopia': '#DDA0DD',
    'others': '#98D8C8'
}


class MobileNetMultiLabel(nn.Module):
    """Multi-Label model matching training architecture."""
    
    def __init__(self, n_diseases=7, dropout_rate=0.3):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_diseases)
        )
    
    def forward(self, x, return_logits=False):
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        drop_feat = self.dropout(pooled)
        logits = self.classifier(drop_feat)
        
        if return_logits:
            return logits
        return torch.sigmoid(logits)


class GradCAMMultiTask:
    """Grad-CAM for multi-task model with per-disease heatmaps."""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook to last conv layer
        self.model.features[-1].register_forward_hook(self._save_activation)
        self.model.features[-1].register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, disease_idx):
        """Generate Grad-CAM for specific disease."""
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor, return_logits=True)
        
        # Backward for specific disease
        target = output[0, disease_idx]
        target.backward()
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


def load_model(checkpoint_path):
    """Load the multi-task model."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = MobileNetMultiLabel(n_diseases=7)
    
    # Load checkpoint with weights_only=False to handle numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_auc = checkpoint.get('best_val_auc', 'N/A')
    if isinstance(best_auc, float):
        print(f"✓ Model loaded (best val AUC: {best_auc:.4f})")
    else:
        print(f"✓ Model loaded")
    
    return model


def get_sample_images(data_dir, n_samples=3):
    """Get sample images from ODIR dataset."""
    images = []
    
    # Try different locations
    possible_dirs = [
        Path(data_dir) / "odir" / "preprocessed_images",
        Path(data_dir) / "odir" / "ODIR-5K" / "ODIR-5K" / "Training Images",
        Path(data_dir) / "raw" / "odir" / "preprocessed_images"
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists():
            image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if image_files:
                images = image_files[:n_samples]
                print(f"Found {len(image_files)} images in {img_dir}")
                break
    
    if not images:
        print("No images found in ODIR directories, using sample from checkpoints...")
        # Fallback - use any fundus image we can find
        for root, dirs, files in os.walk(Path(data_dir)):
            for f in files:
                if f.endswith(('.jpg', '.png')) and 'fundus' in f.lower():
                    images.append(Path(root) / f)
                    if len(images) >= n_samples:
                        break
            if len(images) >= n_samples:
                break
    
    return images


def analyze_image(model, gradcam, image_path, transform, device):
    """Analyze single image for all diseases."""
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        probs = model(img_tensor).squeeze().cpu().numpy()
    
    # Generate CAMs for each disease
    cams = {}
    for i, disease in enumerate(DISEASE_NAMES):
        cam = gradcam.generate(img_tensor, i)
        cams[disease] = cam
    
    return img, probs, cams


def create_visualization(img, probs, cams, image_name, output_dir):
    """Create multi-disease Grad-CAM visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax_orig = fig.add_subplot(2, 4, 1)
    ax_orig.imshow(img)
    ax_orig.set_title("Original Image", fontsize=12, fontweight='bold')
    ax_orig.axis('off')
    
    # Grad-CAM for each disease
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    for i, disease in enumerate(DISEASE_NAMES):
        ax = fig.add_subplot(2, 4, i + 2)
        
        # Create overlay
        cam = cams[disease]
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.6 * img_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        ax.imshow(overlay)
        
        # Title with probability
        prob = probs[i]
        color = 'red' if prob > 0.5 else 'green'
        status = "⚠️" if prob > 0.5 else "✓"
        ax.set_title(f"{disease}\n{status} P={prob:.1%}", 
                    fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
    
    plt.suptitle(f"Multi-Disease Grad-CAM Analysis\n{image_name}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"gradcam_multitask_{image_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    return output_path


def create_summary_plot(all_results, output_dir):
    """Create summary showing predictions across all images."""
    
    fig, axes = plt.subplots(len(all_results), len(DISEASE_NAMES) + 1, 
                            figsize=(20, 4 * len(all_results)))
    
    for row, (name, img, probs, cams) in enumerate(all_results):
        # Original image
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(name[:20], fontsize=8)
        axes[row, 0].axis('off')
        
        img_np = np.array(img.resize((224, 224))) / 255.0
        
        # CAMs for each disease
        for col, disease in enumerate(DISEASE_NAMES):
            ax = axes[row, col + 1]
            
            cam = cams[disease]
            heatmap = plt.cm.jet(cam)[:, :, :3]
            overlay = 0.6 * img_np + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)
            
            ax.imshow(overlay)
            
            prob = probs[col]
            color = 'red' if prob > 0.5 else 'black'
            ax.set_title(f"{prob:.0%}", fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
    
    # Column headers
    for col, disease in enumerate(DISEASE_NAMES):
        axes[0, col + 1].text(0.5, 1.15, disease[:8], transform=axes[0, col + 1].transAxes,
                             ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle("Multi-Task XAI Summary - Per-Disease Attention Maps", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "gradcam_multitask_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Summary saved: {output_path}")
    
    return output_path


def main():
    print("=" * 60)
    print("Multi-Task XAI Analysis (7 Diseases)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model_path = Path("checkpoints_modal/multitask_model.pth")
    if not model_path.exists():
        # Try alternate paths
        alternates = [
            Path("checkpoints_modal/multitask/best_model.pth"),
            Path("checkpoints_modal/multitask"),
        ]
        for alt in alternates:
            if alt.exists():
                model_path = alt
                break
    
    model = load_model(model_path)
    model = model.to(device)
    
    # Initialize Grad-CAM
    gradcam = GradCAMMultiTask(model)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get sample images
    sample_images = get_sample_images("data/raw", n_samples=4)
    
    if not sample_images:
        print("ERROR: No sample images found!")
        return
    
    print(f"\nAnalyzing {len(sample_images)} images...")
    
    # Output directory
    output_dir = Path("baseline_xai/xai_outputs/multitask")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each image
    all_results = []
    
    for img_path in sample_images:
        print(f"\nProcessing {img_path.name}...")
        
        img, probs, cams = analyze_image(model, gradcam, img_path, transform, device)
        
        # Print predictions
        print("  Predictions:")
        for i, disease in enumerate(DISEASE_NAMES):
            status = "⚠️ POSITIVE" if probs[i] > 0.5 else "✓ negative"
            print(f"    {disease}: {probs[i]:.1%} {status}")
        
        # Save individual visualization
        create_visualization(img, probs, cams, img_path.stem, output_dir)
        
        all_results.append((img_path.stem, img, probs, cams))
    
    # Create summary
    if len(all_results) > 1:
        create_summary_plot(all_results, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ XAI Analysis Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
