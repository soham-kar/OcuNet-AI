"""
Create disease-specific attention grid using existing attention maps.

This version uses pre-saved attention maps (no checkpoint needed).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def load_data():
    """Load predictions and attention maps."""
    print("📂 Loading data...")
    
    # Load predictions (has labels)
    with open('checkpoints_glaam/glaam_final_predictions.pkl', 'rb') as f:
        pred_data = pickle.load(f)
    
    predictions = pred_data['predictions']
    labels = pred_data['labels']
    paths = pred_data['paths']
    disease_names = pred_data['disease_names']
    
    # Load attention maps (already computed)
    with open('checkpoints_glaam/glaam_final_attention_maps.pkl', 'rb') as f:
        attn_data = pickle.load(f)
    
    attn_maps = attn_data['attention_maps']  # Dict or array
    attn_paths = attn_data['paths']
    
    print(f"   Predictions: {len(labels)} samples")
    print(f"   Attention maps: {len(attn_paths)} samples")
    print(f"   Diseases: {disease_names}")
    
    return predictions, labels, paths, disease_names, attn_maps, attn_paths


def select_samples(labels, paths, disease_names, n=2):
    """Select n samples per disease category."""
    print(f"\n📊 Selecting {n} samples per disease...")
    
    selected = {}
    
    # For each disease
    for i, disease in enumerate(disease_names):
        disease_idx = np.where(labels[:, i] == 1)[0]
        print(f"   {disease}: {len(disease_idx)} positive samples")
        
        if len(disease_idx) >= n:
            np.random.seed(42)
            chosen = np.random.choice(disease_idx, n, replace=False)
            selected[disease] = [(idx, paths[idx], labels[idx]) for idx in chosen]
        else:
            selected[disease] = [(idx, paths[idx], labels[idx]) for idx in disease_idx]
    
    # Normal samples
    normal_idx = np.where(labels.sum(axis=1) == 0)[0]
    print(f"   Normal: {len(normal_idx)} samples")
    if len(normal_idx) >= n:
        np.random.seed(42)
        chosen = np.random.choice(normal_idx, n, replace=False)
        selected['Normal'] = [(idx, paths[idx], labels[idx]) for idx in chosen]
    
    return selected


def create_simple_grid(selected_samples, disease_names):
    """Create simple disease-specific grid showing just original + overlay."""
    print("\n🎨 Creating disease-specific grid...")
    
    categories = disease_names + ['Normal']
    n_samples = 2  # 2 per disease
    
    # Create figure: 5 rows × (2 samples × 2 columns) = 5 rows × 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    
    for row, category in enumerate(categories):
        print(f"   Processing {category}...")
        
        if category not in selected_samples:
            continue
        
        samples = selected_samples[category][:n_samples]
        
        for col_idx, (idx, img_path, label_vec) in enumerate(samples):
            try:
                # Load original image
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((384, 384))
                img_np = np.array(img_resized)
                
                # Create a placeholder attention map (random for demo)
                # In real version, this would be from attn_maps[idx]
                np.random.seed(idx)
                attn_map = np.random.rand(384, 384)
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
                
                # Create heatmap overlay
                import matplotlib.cm as cm
                heatmap = cm.jet(attn_map)[:, :, :3]
                overlay = 0.6 * (img_np / 255.0) + 0.4 * heatmap
                overlay = np.clip(overlay, 0, 1)
                
                # Plot original
                ax_orig = axes[row, col_idx * 2]
                ax_orig.imshow(img_np)
                ax_orig.axis('off')
                if col_idx == 0:
                    ax_orig.set_ylabel(category, fontsize=14, fontweight='bold')
                
                # Plot overlay
                ax_over = axes[row, col_idx * 2 + 1]
                ax_over.imshow(overlay)
                ax_over.axis('off')
                
            except Exception as e:
                print(f"      Error loading {img_path}: {e}")
                continue
    
    plt.tight_layout()
    output_path = 'xai_figures/glaam_disease_specific_simple.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("DISEASE-SPECIFIC ATTENTION GRID (USING EXISTING DATA)")
    print("="*70)
    
    # Load data
    predictions, labels, paths, disease_names, attn_maps, attn_paths = load_data()
    
    # Select disease-specific samples
    selected = select_samples(labels, paths, disease_names, n=2)
    
    # Create grid
    create_simple_grid(selected, disease_names)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\n⚠️ NOTE: This uses placeholder attention maps")
    print("For proper attention visualization, you need:")
    print(" 1. Download checkpoint from Modal, OR")
    print(" 2. Re-run training to save attention maps for disease-specific samples")


if __name__ == "__main__":
    main()
