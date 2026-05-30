"""
Generate disease-specific attention visualizations for publication.

This script creates a proper attention grid showing:
- Balanced representation of all disease classes
- Clear disease labels
- Demonstration of disease-specific localization

Usage:
    python evaluation/generate_disease_specific_attention.py
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import will be done after mounting code
from models.backbones.mobilenet_glaam_odir import MobileNetV2WithGLAAM_ODIR


def load_predictions(pred_file='checkpoints_glaam/glaam_final_predictions.pkl'):
    """Load predictions and labels."""
    print(f"📂 Loading predictions from: {pred_file}")
    with open(pred_file, 'rb') as f:
        data = pickle.load(f)
    
    predictions = data['predictions']  # (N, 4)
    labels = data['labels']  # (N, 4)
    paths = data['paths']  # List of paths
    disease_names = data['disease_names']  # ['Cataract', 'DR', 'Glaucoma', 'Myopia']
    
    print(f"   Loaded {len(labels)} samples")
    print(f"   Diseases: {disease_names}")
    
    return predictions, labels, paths, disease_names


def select_disease_specific_samples(labels, paths, disease_names, samples_per_disease=3):
    """
    Select balanced samples from each disease category.
    
    Returns:
        dict: {disease_name: [(idx, path, labels), ...]}
    """
    print(f"\n📊 Selecting {samples_per_disease} samples per disease...")
    
    selected = {}
    
    # For each disease
    for i, disease in enumerate(disease_names):
        # Find samples with this disease (label = 1)
        disease_indices = np.where(labels[:, i] == 1)[0]
        print(f"   {disease}: {len(disease_indices)} positive samples")
        
        if len(disease_indices) >= samples_per_disease:
            # Select random samples
            np.random.seed(42)  # Reproducible
            chosen_idx = np.random.choice(disease_indices, samples_per_disease, replace=False)
            selected[disease] = [(idx, paths[idx], labels[idx]) for idx in chosen_idx]
        else:
            print(f"   ⚠️ Warning: Only {len(disease_indices)} samples for {disease}")
            selected[disease] = [(idx, paths[idx], labels[idx]) for idx in disease_indices]
    
    # Also select true normal samples (all labels = 0)
    normal_indices = np.where((labels.sum(axis=1) == 0))[0]
    print(f"   Normal: {len(normal_indices)} samples")
    if len(normal_indices) >= samples_per_disease:
        np.random.seed(42)
        chosen_idx = np.random.choice(normal_indices, samples_per_disease, replace=False)
        selected['Normal'] = [(idx, paths[idx], labels[idx]) for idx in chosen_idx]
    else:
        selected['Normal'] = [(idx, paths[idx], labels[idx]) for idx in normal_indices]
    
    return selected


def generate_grad_cam(model, image_tensor, target_class_idx):
    """
    Generate Grad-CAM heatmap for a specific disease.
    
    Args:
        model: GLAAM model
        image_tensor: (1, 3, H, W)
        target_class_idx: Disease index (0-3)
    
    Returns:
        cam: (H, W) numpy array
    """
    model.eval()
    model.zero_grad()
    
    # Storage for hooks
    gradients = None
    activations = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    # Register hooks on attention layer
    if hasattr(model, 'attention'):
        handle_fwd = model.attention.register_forward_hook(forward_hook)
        handle_bwd = model.attention.register_full_backward_hook(backward_hook)
    else:
        print("⚠️ Warning: No attention layer found, using features")
        handle_fwd = model.features.register_forward_hook(forward_hook)
        handle_bwd = model.features.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    logits = output['logits']  # (1, 4)
    
    # Backward for target class
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class_idx] = 1
    logits.backward(gradient=one_hot, retain_graph=True)
    
    # Clean up hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Compute CAM
    if gradients is not None and activations is not None:
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros((7, 7))  # Fallback
    
    return cam


def create_disease_specific_grid(model, selected_samples, disease_names, output_path='xai_figures/glaam_disease_specific_grid.png'):
    """
    Create publication-quality disease-specific attention grid.
    """
    print(f"\n🎨 Creating disease-specific attention grid...")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = next(model.parameters()).device
    
    # Prepare figure: 5 diseases × 3 samples × 3 columns (original, heatmap, overlay)
    categories = disease_names + ['Normal']
    n_categories = len(categories)
    samples_per_cat = 3
    
    fig = plt.figure(figsize=(15, 5 * n_categories))
    gs = fig.add_gridspec(n_categories, samples_per_cat * 3, hspace=0.3, wspace=0.1)
    
    disease_idx_map = {disease: i for i, disease in enumerate(disease_names)}
    
    for row, category in enumerate(categories):
        print(f"   Processing {category}...")
        
        if category not in selected_samples:
            continue
        
        samples = selected_samples[category]
        
        for col_offset, (idx, img_path, label_vec) in enumerate(samples[:samples_per_cat]):
            # Load image
            try:
                original_img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"      ⚠️ Could not load {img_path}: {e}")
                continue
            
            original_np = np.array(original_img.resize((384, 384)))
            
            # Preprocess for model
            img_tensor = transform(original_img).unsqueeze(0).to(device)
            
            # Generate Grad-CAM
            if category == 'Normal':
                target_class = 0  # Just use first class
            else:
                target_class = disease_idx_map[category]
            
            cam = generate_grad_cam(model, img_tensor, target_class)
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
            
            # Create heatmap overlay
            import matplotlib.cm as cm
            heatmap = cm.jet(cam_resized)[:, :, :3]  # RGB
            overlay = 0.6 * original_np / 255.0 + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)
            
            # Plot: Original | Heatmap | Overlay
            base_col = col_offset * 3
            
            # Original
            ax1 = fig.add_subplot(gs[row, base_col], alpha=1.0)  # Added alpha parameter            ax1.imshow(original_np)
            ax1.axis('off')
            if col_offset == 0:
                ax1.set_title(f"{category}", fontsize=14, fontweight='bold', loc='left')
            
            # Heatmap
            ax2 = fig.add_subplot(gs[row, base_col + 1])
            ax2.imshow(cam_resized, cmap='jet')
            ax2.axis('off')
            
            # Overlay
            ax3 = fig.add_subplot(gs[row, base_col + 2])
            ax3.imshow(overlay)
            ax3.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved disease-specific grid to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("DISEASE-SPECIFIC ATTENTION VISUALIZATION")
    print("="*70)
    
    # Load predictions
    predictions, labels, paths, disease_names = load_predictions()
    
    # Select disease-specific samples
    selected_samples = select_disease_specific_samples(labels, paths, disease_names, samples_per_disease=3)
    
    # Load model
    print("\n📦 Loading GLAAM model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    model = MobileNetV2WithGLAAM_ODIR(num_classes=4, pretrained=False)
    
    checkpoint_path = 'checkpoints_glaam/glaam_final_best.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"   ✅ Loaded checkpoint from: {checkpoint_path}")
    except FileNotFoundError:
        print(f"   ⚠️ Checkpoint not found: {checkpoint_path}")
        print("   Using predictions file only (no new attention generation)")
        return
    
    model = model.to(device)
    model.eval()
    
    # Create disease-specific grid
    create_disease_specific_grid(model, selected_samples, disease_names)
    
    print("\n" + "="*70)
    print("✅ COMPLETE - Disease-specific visualization ready for publication!")
    print("="*70)


if __name__ == "__main__":
    main()
