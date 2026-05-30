"""
GLAAM XAI Visualization Script
Generates publication-quality attention heatmaps for thesis figures.

Usage:
    python generate_xai_figures.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import torch
import torch.nn.functional as F

# Configuration
ATTENTION_MAPS_PATH = "checkpoints_glaam/glaam_final_attention_maps.pkl"
OUTPUT_DIR = Path("xai_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

# ImageNet denormalization for visualization
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(img_tensor):
    """Convert normalized tensor to displayable image"""
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return img

def create_attention_heatmap(attention_map, size=(224, 224)):
    """Create a heatmap from attention weights"""
    # Handle different attention map formats
    if isinstance(attention_map, torch.Tensor):
        attn = attention_map.detach().cpu().numpy()
    else:
        attn = attention_map
    
    # Average across channels if multi-channel
    if len(attn.shape) == 3:
        attn = attn.mean(axis=0)
    elif len(attn.shape) == 4:
        attn = attn[0].mean(axis=0)
    
    # Normalize to [0, 1]
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    # Resize to image size
    attn_resized = F.interpolate(
        torch.tensor(attn).unsqueeze(0).unsqueeze(0).float(),
        size=size,
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()
    
    return attn_resized

def overlay_attention_on_image(image, attention_map, alpha=0.5, cmap='jet'):
    """Overlay attention heatmap on original image"""
    # Create colored heatmap
    heatmap = cm.get_cmap(cmap)(attention_map)[:, :, :3]
    
    # Blend with original image
    overlay = alpha * heatmap + (1 - alpha) * image
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def generate_single_figure(image, attention_maps, labels, path, idx, save_path):
    """Generate a single figure with original + attention overlays"""
    fig, axes = plt.subplots(1, len(attention_maps) + 1, figsize=(4 * (len(attention_maps) + 1), 4))
    
    # Original image
    img = denormalize_image(image)
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Add disease labels
    active_diseases = [DISEASE_NAMES[i] for i, v in enumerate(labels) if v > 0.5]
    if active_diseases:
        label_text = "Diseases: " + ", ".join(active_diseases)
    else:
        label_text = "Normal"
    axes[0].text(0.5, -0.1, label_text, transform=axes[0].transAxes, 
                  ha='center', fontsize=10, color='red' if active_diseases else 'green')
    
    # Attention maps from each stage
    for i, (stage_name, attn_data) in enumerate(attention_maps.items()):
        if stage_name == 'alpha':
            continue
            
        if isinstance(attn_data, dict):
            attn = attn_data.get('combined_attention', attn_data)
        else:
            attn = attn_data
        
        if isinstance(attn, torch.Tensor):
            attn = attn[idx] if attn.dim() == 4 else attn
        
        heatmap = create_attention_heatmap(attn)
        overlay = overlay_attention_on_image(img, heatmap)
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"Attention: {stage_name}", fontsize=12, fontweight='bold')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  💾 Saved: {save_path}")

def generate_comparison_grid(data, save_path):
    """Generate a grid of all samples for thesis figure"""
    images = data['images']
    attention_maps = data['attention_maps']
    labels = data['labels']
    
    n_samples = min(6, len(images))  # Max 6 samples for readability
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    for i in range(n_samples):
        img = denormalize_image(images[i])
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original" if i == 0 else "", fontsize=11)
        axes[i, 0].axis('off')
        
        # Get final stage attention
        final_stage = [k for k in attention_maps.keys() if k not in ['alpha', 'global_weight', 'local_weight']]
        if final_stage:
            attn = attention_maps[final_stage[-1]]
            if isinstance(attn, dict):
                attn = attn.get('combined_attention', list(attn.values())[0])
            if isinstance(attn, torch.Tensor) and attn.dim() == 4:
                attn = attn[i]
            
            heatmap = create_attention_heatmap(attn)
            
            # Attention heatmap alone
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title("GLAAM Attention" if i == 0 else "", fontsize=11)
            axes[i, 1].axis('off')
            
            # Overlay
            overlay = overlay_attention_on_image(img, heatmap, alpha=0.4)
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("Overlay" if i == 0 else "", fontsize=11)
            axes[i, 2].axis('off')
        
        # Add disease label on the left
        label = labels[i] if isinstance(labels, np.ndarray) else labels[i].numpy()
        active = [DISEASE_NAMES[j] for j, v in enumerate(label) if v > 0.5]
        label_text = ", ".join(active) if active else "Normal"
        axes[i, 0].text(-0.15, 0.5, label_text, transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10,
                        fontweight='bold', color='red' if active else 'green')
    
    plt.suptitle("GLAAM Attention Visualization on ODIR-5K Fundus Images", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 Saved grid figure: {save_path}")

def generate_alpha_analysis(data, save_path):
    """Analyze the learned alpha values across stages"""
    attention_maps = data['attention_maps']
    
    # Extract alpha values if available
    alpha_values = {}
    for stage_name, attn_data in attention_maps.items():
        if isinstance(attn_data, dict) and 'alpha' in attn_data:
            alpha_values[stage_name] = attn_data['alpha']
    
    if not alpha_values:
        print("  ⚠️ No alpha values found in attention maps")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stages = list(alpha_values.keys())
    alphas = [alpha_values[s] for s in stages]
    
    bars = ax.bar(stages, alphas, color='steelblue', edgecolor='black')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Balanced (α=0.5)')
    
    ax.set_xlabel('Attention Stage', fontsize=12)
    ax.set_ylabel('Learned Alpha (Global vs Local Balance)', fontsize=12)
    ax.set_title('GLAAM Learnable Attention Fusion Parameters', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels on bars
    for bar, alpha in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{alpha:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"💾 Saved alpha analysis: {save_path}")

def main():
    print("=" * 60)
    print("🎨 GLAAM XAI Visualization Generator")
    print("=" * 60)
    
    # Load attention maps
    print(f"\n📁 Loading attention maps from: {ATTENTION_MAPS_PATH}")
    
    if not Path(ATTENTION_MAPS_PATH).exists():
        print(f"❌ File not found: {ATTENTION_MAPS_PATH}")
        print("   Please download from Modal volume first:")
        print("   modal volume get cataract-checkpoints glaam_final_attention_maps.pkl ./checkpoints/ --force")
        return
    
    with open(ATTENTION_MAPS_PATH, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded {len(data['images'])} samples")
    print(f"   Attention stages: {list(data['attention_maps'].keys())}")
    
    # Generate individual figures
    print("\n📸 Generating individual attention figures...")
    for i in range(len(data['images'])):
        save_path = OUTPUT_DIR / f"attention_sample_{i+1}.png"
        generate_single_figure(
            data['images'][i],
            data['attention_maps'],
            data['labels'][i],
            data['paths'][i] if 'paths' in data else f"sample_{i}",
            i,
            save_path
        )
    
    # Generate comparison grid (main thesis figure)
    print("\n📊 Generating comparison grid (thesis figure)...")
    generate_comparison_grid(data, OUTPUT_DIR / "glaam_attention_grid.png")
    
    # Generate alpha analysis
    print("\n📈 Generating alpha analysis...")
    generate_alpha_analysis(data, OUTPUT_DIR / "alpha_analysis.png")
    
    print("\n" + "=" * 60)
    print("✅ XAI figures generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    
    # List generated files
    print("\n📋 Generated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")

if __name__ == "__main__":
    main()
