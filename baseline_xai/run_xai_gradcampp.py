"""
Enhanced XAI with Grad-CAM vs Grad-CAM++ Comparison.

FIXED VERSION:
- Bug #1: True vanilla Grad-CAM class implemented
- Bug #2: Efficient model reuse (no per-disease deepcopy)
- Bug #3: Score-CAM for uncertain cases
- Added: Clinical-grade DR visualization

Usage:
    python baseline_xai/run_xai_gradcampp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from scipy.ndimage import gaussian_filter


DISEASE_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 
                 'amd', 'hypertension', 'myopia', 'others']


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


class GradCAM:
    """
    Vanilla Grad-CAM.
    
    Weights = mean of gradients (simpler, more spread attention)
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        target = target_layer if target_layer else model.features[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, disease_idx):
        """Generate vanilla Grad-CAM."""
        self.model.eval()
        self.model.zero_grad()
        
        input_tensor.requires_grad = True
        output = self.model(input_tensor, return_logits=True)
        
        target = output[0, disease_idx]
        self.model.zero_grad()
        target.backward()
        
        grads = self.gradients
        acts = self.activations
        
        # Vanilla Grad-CAM: weights = mean of gradients
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().detach().cpu().numpy()


class GradCAMPlusPlus:
    """
    Grad-CAM++ for improved localization.
    
    Uses second-order gradients (alpha weights) for better:
    - Multiple occurrences handling
    - Small object/lesion localization
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        target = target_layer if target_layer else model.features[-1]
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, disease_idx):
        """Generate Grad-CAM++."""
        self.model.eval()
        self.model.zero_grad()
        
        input_tensor.requires_grad = True
        output = self.model(input_tensor, return_logits=True)
        
        target = output[0, disease_idx]
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        
        # Grad-CAM++ alpha computation
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts * grads_power_3, dim=(2, 3), keepdim=True)
        
        eps = 1e-8
        alpha = grads_power_2 / (2 * grads_power_2 + sum_acts + eps)
        alpha = torch.where(torch.isnan(alpha) | torch.isinf(alpha), 
                           torch.zeros_like(alpha), alpha)
        
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / (cam.max() + eps)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().detach().cpu().numpy()


class ScoreCAM:
    """Score-CAM: Gradient-free CAM (more reliable for noisy gradients)."""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        
        target = target_layer if target_layer else model.features[-1]
        target.register_forward_hook(self._save_activation)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate(self, input_tensor, disease_idx, n_channels=32):
        """Generate Score-CAM."""
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor, return_logits=True)
            acts = self.activations
            
            n_channels = min(n_channels, acts.shape[1])
            channel_indices = np.linspace(0, acts.shape[1]-1, n_channels, dtype=int)
            
            H, W = input_tensor.shape[2:]
            
            scores = []
            for idx in channel_indices:
                act_map = acts[0, idx:idx+1, :, :]
                act_map = F.interpolate(act_map.unsqueeze(0), size=(H, W), 
                                       mode='bilinear', align_corners=False)
                act_map = act_map.squeeze()
                
                act_map = act_map - act_map.min()
                if act_map.max() > 0:
                    act_map = act_map / act_map.max()
                
                masked_input = input_tensor * act_map.unsqueeze(0).unsqueeze(0)
                out = self.model(masked_input, return_logits=True)
                score = out[0, disease_idx].item()
                scores.append(score)
            
            scores = np.array(scores)
            scores = scores - scores.min()
            if scores.max() > 0:
                scores = scores / scores.max()
            
            cam = np.zeros((H, W))
            for i, idx in enumerate(channel_indices):
                act_map = acts[0, idx, :, :].cpu().numpy()
                act_map = np.array(Image.fromarray(act_map).resize((W, H)))
                cam += scores[i] * act_map
            
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            return cam


def load_model(checkpoint_path):
    """Load the multi-task model."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = MobileNetMultiLabel(n_diseases=7)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    best_auc = checkpoint.get('best_val_auc', 'N/A')
    if isinstance(best_auc, float):
        print(f"✓ Model loaded (best val AUC: {best_auc:.4f})")
    else:
        print(f"✓ Model loaded")
    
    return model


def create_comparison_visualization(img, probs, cams_gradcam, cams_gradcampp, 
                                   image_name, output_dir):
    """Create side-by-side comparison of Grad-CAM vs Grad-CAM++."""
    
    fig = plt.figure(figsize=(24, 10))
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    # Row 1: Original + predictions
    ax_orig = fig.add_subplot(3, 8, 1)
    ax_orig.imshow(img)
    ax_orig.set_title("Original", fontsize=10, fontweight='bold')
    ax_orig.axis('off')
    
    ax_prob = fig.add_subplot(3, 8, 2)
    colors = ['red' if p > 0.5 else 'green' for p in probs]
    bars = ax_prob.barh(DISEASE_NAMES, probs, color=colors)
    ax_prob.set_xlim(0, 1)
    ax_prob.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_prob.set_title("Predictions", fontsize=10, fontweight='bold')
    
    # Row 2: Vanilla Grad-CAM
    for i, disease in enumerate(DISEASE_NAMES):
        ax = fig.add_subplot(3, 8, 9 + i)
        cam = cams_gradcam[disease]
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{disease[:8]}", fontsize=9)
        ax.axis('off')
    
    fig.text(0.02, 0.5, "Grad-CAM\n(vanilla)", ha='center', va='center', 
             fontsize=10, fontweight='bold', rotation=90)
    
    # Row 3: Grad-CAM++
    for i, disease in enumerate(DISEASE_NAMES):
        ax = fig.add_subplot(3, 8, 17 + i)
        cam = cams_gradcampp[disease]
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap
        ax.imshow(np.clip(overlay, 0, 1))
        prob = probs[i]
        color = 'red' if prob > 0.5 else 'black'
        ax.set_title(f"{prob:.0%}", fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    fig.text(0.02, 0.2, "Grad-CAM++\n(improved)", ha='center', va='center', 
             fontsize=10, fontweight='bold', rotation=90)
    
    plt.suptitle(f"XAI Comparison: Grad-CAM vs Grad-CAM++\n{image_name}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"comparison_{image_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    return output_path


def create_clinical_dr_visualization(img, probs, cam_gradcam, cam_gradcampp, 
                                     image_name, output_dir, cam_scorecam=None):
    """Clinical-grade DR visualization with medical annotations."""
    
    n_panels = 5 if cam_scorecam is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 4))
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    # 1. Original with macula overlay
    axes[0].imshow(img)
    h, w = img_np.shape[:2]
    center = (w//2, h//2)
    radius = min(w, h) // 6
    circle = plt.Circle(center, radius, color='yellow', fill=False, linewidth=2, alpha=0.7)
    axes[0].add_patch(circle)
    axes[0].text(center[0], center[1]-radius-10, "Macula", color='yellow', fontsize=8, ha='center')
    dr_prob = probs[1]
    axes[0].set_title(f"Original\nDR Prob: {dr_prob:.1%}", fontsize=11)
    axes[0].axis('off')
    
    # 2. Grad-CAM (spread)
    heatmap = plt.cm.jet(cam_gradcam)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Grad-CAM\n(Spread)", fontsize=11, color='orange')
    axes[1].axis('off')
    
    # 3. Grad-CAM++ (focused)
    heatmap = plt.cm.jet(cam_gradcampp)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Grad-CAM++\n(Lesion-Focused)", fontsize=11, color='green')
    axes[2].axis('off')
    
    # 4. Clinical hotspots
    axes[3].imshow(img)
    smoothed = gaussian_filter(cam_gradcampp, sigma=2)
    threshold = np.percentile(smoothed, 95)
    y, x = np.where(smoothed > threshold)
    if len(x) > 0:
        axes[3].scatter(x, y, c='lime', s=15, marker='+', alpha=0.8, linewidths=1)
    axes[3].set_title("CAM++ Hotspots\n(Top 5% attention)", fontsize=11)
    axes[3].axis('off')
    
    # 5. Score-CAM (if provided)
    if cam_scorecam is not None:
        heatmap = plt.cm.jet(cam_scorecam)[:, :, :3]
        overlay = 0.5 * img_np + 0.5 * heatmap
        axes[4].imshow(np.clip(overlay, 0, 1))
        axes[4].set_title("Score-CAM\n(Gradient-free)", fontsize=11, color='blue')
        axes[4].axis('off')
    
    plt.suptitle(f"Diabetic Retinopathy: Clinical XAI Analysis\n{image_name}", 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"clinical_dr_{image_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Clinical DR saved: {output_path}")
    plt.close()
    return output_path


def main():
    print("=" * 60)
    print("Enhanced XAI: Grad-CAM vs Grad-CAM++ (Fixed)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    model_path = Path("checkpoints_modal/multitask_model.pth")
    if not model_path.exists():
        for alt in [Path("checkpoints_modal/multitask/best_model.pth"), 
                    Path("checkpoints_modal/multitask")]:
            if alt.exists():
                model_path = alt
                break
    
    model = load_model(model_path)
    model = model.to(device)
    
    # Create model copies ONCE (Bug #2 fix)
    model_gc = deepcopy(model).to(device)
    model_gcpp = deepcopy(model).to(device)
    model_sc = deepcopy(model).to(device)
    
    # Initialize CAM methods with correct classes (Bug #1 fix)
    gradcam = GradCAM(model_gc)  # TRUE vanilla Grad-CAM
    gradcampp = GradCAMPlusPlus(model_gcpp)  # Grad-CAM++
    scorecam = ScoreCAM(model_sc)  # For uncertain cases
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_dir = Path("data/raw/odir/preprocessed_images")
    if not img_dir.exists():
        img_dir = Path("data/raw/odir/ODIR-5K/ODIR-5K/Training Images")
    
    image_files = list(img_dir.glob("*.jpg"))[:5]
    
    if not image_files:
        print("ERROR: No sample images found!")
        return
    
    print(f"\nAnalyzing {len(image_files)} images...")
    
    output_dir = Path("baseline_xai/xai_outputs/gradcampp_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = model(img_tensor).squeeze().cpu().numpy()
        
        # Generate CAMs for each disease (Bug #2 fix: reuse models)
        cams_gradcam = {}
        cams_gradcampp = {}
        
        for i, disease in enumerate(DISEASE_NAMES):
            # Clone only the tensor, not the model
            tensor_gc = img_tensor.clone().requires_grad_(True)
            tensor_gcpp = img_tensor.clone().requires_grad_(True)
            
            cams_gradcam[disease] = gradcam.generate(tensor_gc, i)
            cams_gradcampp[disease] = gradcampp.generate(tensor_gcpp, i)
        
        print("  Predictions:")
        for i, disease in enumerate(DISEASE_NAMES):
            status = "⚠️ POSITIVE" if probs[i] > 0.5 else "✓ negative"
            print(f"    {disease}: {probs[i]:.1%} {status}")
        
        create_comparison_visualization(img, probs, cams_gradcam, cams_gradcampp,
                                       img_path.stem, output_dir)
        
        # Clinical DR visualization for DR cases (Bug #3: use Score-CAM for uncertain)
        if probs[1] > 0.3:
            cam_sc = None
            if 0.3 < probs[1] < 0.7:  # Uncertain - add Score-CAM
                print("  Adding Score-CAM for uncertain DR...")
                cam_sc = scorecam.generate(img_tensor, 1)
            
            create_clinical_dr_visualization(
                img, probs,
                cams_gradcam['diabetic_retinopathy'],
                cams_gradcampp['diabetic_retinopathy'],
                img_path.stem, output_dir,
                cam_scorecam=cam_sc
            )
    
    print("\n" + "=" * 60)
    print("✅ Enhanced XAI Analysis Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
