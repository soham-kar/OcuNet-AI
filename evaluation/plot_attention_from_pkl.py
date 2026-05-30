"""
Generate disease-specific attention grid directly from saved pkl files.
No model or checkpoint required - uses saved attention maps from training.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
import torch

# ── Load attention maps (saved during training - disease-specific!) ────────
print("Loading saved attention maps...")
with open('checkpoints_glaam/glaam_final_attention_maps.pkl', 'rb') as f:
    attn_data = pickle.load(f)

with open('checkpoints_glaam/glaam_final_predictions.pkl', 'rb') as f:
    pred_data = pickle.load(f)

print(f"Attention keys : {list(attn_data.keys())}")
print(f"Number of samples: {len(attn_data['paths'])}")

# ── Inspect what we have ───────────────────────────────────────────────────
raw_paths   = attn_data['paths']        # list of 10 image paths (Modal paths)
labels      = attn_data['labels']       # tensor (10, 4)

# Remap Modal container paths → local paths
# Modal: /data/odir/preprocessed_images/0_left.jpg
# Local: data/raw/odir/preprocessed_images/0_left.jpg
LOCAL_IMG_ROOT = Path("data/raw/odir/preprocessed_images")
def remap_path(p):
    fname = Path(p).name
    local = LOCAL_IMG_ROOT / fname
    return str(local) if local.exists() else p
paths = [remap_path(p) for p in raw_paths]
attn_maps   = attn_data['attention_maps']  # dict of stage → tensor
disease_names = pred_data['disease_names']

# Print each sample's disease label
print("\nSamples in attention file:")
for i, (p, lbl) in enumerate(zip(paths, labels)):
    active = [disease_names[j] for j in range(len(disease_names)) if lbl[j] == 1] or ['Normal']
    print(f"  [{i+1}] {Path(p).name} → {active}")

# ── Get last-stage attention map (most semantically informative) ───────────
last_stage_key = list(attn_maps.keys())[-1]
print(f"\nUsing attention stage: {last_stage_key}")
attn_tensor = attn_maps[last_stage_key]  # (10, C, H, W)

# Average across channels to get spatial attention
if attn_tensor.dim() == 4:
    spatial_attn = attn_tensor.mean(dim=1)  # (10, H, W)
elif attn_tensor.dim() == 3:
    spatial_attn = attn_tensor
else:
    spatial_attn = attn_tensor.squeeze()

spatial_attn = spatial_attn.cpu().numpy()

# ── Create publication-quality grid ───────────────────────────────────────
n_samples = len(paths)
fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5 * n_samples))
fig.patch.set_facecolor('white')

for i, (img_path, lbl) in enumerate(zip(paths, labels)):
    active = [disease_names[j] for j in range(len(disease_names)) if lbl[j] == 1] or ['Normal']
    category = active[0] if len(active) == 1 else '+'.join(active)

    # ── Load original image ───────────────────────────────────────────
    try:
        orig = Image.open(img_path).convert('RGB').resize((384, 384))
        orig_np = np.array(orig)
    except Exception as e:
        print(f"  ⚠️ Could not load {img_path}: {e}")
        orig_np = np.zeros((384, 384, 3), dtype=np.uint8)

    # ── Prepare attention map ─────────────────────────────────────────
    attn = spatial_attn[i]
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn_up = np.array(
        Image.fromarray((attn_norm * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)
    ) / 255.0

    # ── Overlay ───────────────────────────────────────────────────────
    heatmap_rgb = cm.jet(attn_up)[:, :, :3]
    overlay = 0.6 * (orig_np / 255.0) + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    # ── Disease label colour ──────────────────────────────────────────
    colour_map = {
        'Cataract': '#E74C3C',
        'DR':       '#E67E22',
        'Glaucoma': '#8E44AD',
        'Myopia':   '#2980B9',
        'Normal':   '#27AE60',
    }
    label_colour = colour_map.get(active[0], '#555555')

    # ── Row of 3 panels ───────────────────────────────────────────────
    row_axes = axes[i] if n_samples > 1 else axes

    # Panel 1: Original
    row_axes[0].imshow(orig_np)
    row_axes[0].axis('off')
    row_axes[0].set_title(
        f"{category}", fontsize=13, fontweight='bold',
        color=label_colour, loc='left', pad=6
    )
    if i == 0:
        row_axes[0].set_title("Original Image", fontsize=10,
                               color='grey', loc='right', pad=6)

    # Panel 2: Raw attention heatmap
    row_axes[1].imshow(attn_up, cmap='jet', vmin=0, vmax=1)
    row_axes[1].axis('off')
    if i == 0:
        row_axes[1].set_title("Attention Map", fontsize=10,
                               color='grey', loc='right', pad=6)

    # Panel 3: Overlay
    row_axes[2].imshow(overlay)
    row_axes[2].axis('off')
    if i == 0:
        row_axes[2].set_title("Overlay", fontsize=10,
                               color='grey', loc='right', pad=6)

plt.suptitle(
    "GLAAM Disease-Specific Attention Visualizations (ODIR-5K)",
    fontsize=16, fontweight='bold', y=1.01
)
plt.tight_layout(rect=[0, 0, 1, 1])

out_path = 'xai_figures/glaam_disease_specific_grid.png'
Path('xai_figures').mkdir(exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_path}")
plt.close()
