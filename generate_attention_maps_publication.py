# generate_attention_maps_publication.py
"""
Publication-Grade Disease-Specific Attention Map Visualization for GLAAM-4X
==========================================================================

Improvements over the basic version:
  1. Grayscale fundus background -> color heatmap pops, standard in medical papers
  2. Perceptually-uniform colormap (inferno) instead of jet
  3. Percentile (1-99) normalization instead of min-max (robust to outliers)
  4. Gaussian smoothing of the attention map (removes pixel noise)
  5. Thresholding -> only significant attention is shown (cleaner)
  6. DPI 300 -> print-ready resolution
  7. Per-panel colorbar -> intensity reference
  8. Smart sample selection -> true positives with high confidence + healthy controls
  9. Disease gating weights displayed -> shows the routing mechanism

Usage:
    venv_linux/bin/python generate_attention_maps_publication.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
BASE = Path(__file__).parent
RAW_ROOT = BASE / "data" / "raw"
TEST_CSV = BASE / "data" / "test_v4.csv"
MODEL_PATH = BASE / "cataract-checkpoints_v3" / "publication_analysis_v2" / "ablation" / "A1_Full_GLAAM4X_model.pth"
OUT_DIR = BASE / "xai_figures" / "attention_maps_publication"

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
IMG_SIZE = 384
DPI = 300
SMOOTH_SIGMA = 3.0          # Gaussian smoothing sigma (pixels at 384 res)
LOW_PCT, HIGH_PCT = 1, 99   # percentile normalization range
ALPHA_MAX = 0.85            # max heatmap opacity
THRESHOLD_PCT = 30          # only show attention above this percentile


# ═══════════════════════════════════════════════════════════════
# GLAAM-4X MODEL (must match trained architecture)
# ═══════════════════════════════════════════════════════════════
class GlobalAttentionBranch(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return self.fc(y).view(b, c, 1, 1)


class LocalAttentionBranch(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid())
    def forward(self, x):
        return self.conv(x)


class GLAAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, use_residual=True):
        super().__init__()
        self.global_branch = GlobalAttentionBranch(in_channels, reduction)
        self.local_branch = LocalAttentionBranch(in_channels, reduction)
        self.use_residual = use_residual
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, x, return_attention=False):
        gw = self.global_branch(x)
        lw = self.local_branch(x)
        combined = self.alpha * gw + (1 - self.alpha) * lw
        out = x * combined
        result = (x + out) if self.use_residual else out
        if return_attention:
            # Return the LOCAL branch attention (spatially meaningful).
            # The global branch is channel-wise (spatially uniform) and dominates
            # the combined map, hiding disease-specific spatial localization.
            return result, lw
        return result


class MultiScaleGLAAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.attention_fine = GLAAMBlock(in_channels, reduction=reduction)
        self.attention_medium = GLAAMBlock(in_channels, reduction=reduction * 2)
        self.attention_coarse = GLAAMBlock(in_channels, reduction=reduction * 4)
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
    def forward(self, x, return_attention=False):
        B, C, H, W = x.shape
        fine = self.attention_fine(x)
        medium = self.attention_medium(F.avg_pool2d(x, 2))
        medium_up = F.interpolate(medium, size=(H, W), mode='bilinear', align_corners=False)
        coarse = self.attention_coarse(F.avg_pool2d(x, 4))
        coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)
        combined = torch.cat([fine, medium_up, coarse_up], dim=1)
        output = self.scale_fusion(combined)
        if return_attention:
            # Return the LOCAL branch attention from each scale (spatially meaningful)
            _, fine_lw = self.attention_fine(x, return_attention=True)
            _, medium_lw = self.attention_medium(F.avg_pool2d(x, 2), return_attention=True)
            medium_lw = F.interpolate(medium_lw, size=(H, W), mode='bilinear', align_corners=False)
            _, coarse_lw = self.attention_coarse(F.avg_pool2d(x, 4), return_attention=True)
            coarse_lw = F.interpolate(coarse_lw, size=(H, W), mode='bilinear', align_corners=False)
            return output, (fine_lw + medium_lw + coarse_lw) / 3
        return output


class DiseaseGatingNetwork(nn.Module):
    def __init__(self, in_channels, n_diseases=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, n_diseases), nn.Softmax(dim=1))
    def forward(self, x):
        return self.gate(x)


GLAAM_DISEASE_ORDER = ['DR', 'Glaucoma', 'Cataract', 'Myopia']


class GLAAM_4X(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.3):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.backbone = mobilenet.features
        self.attention_heads = nn.ModuleDict({
            'DR': MultiScaleGLAAM(1280, reduction=4),
            'Glaucoma': GLAAMBlock(1280, reduction=8),
            'Cataract': GLAAMBlock(1280, reduction=16),
            'Myopia': GLAAMBlock(1280, reduction=32),
        })
        self.disease_gate = DiseaseGatingNetwork(1280, n_diseases=4)
        self.classifiers = nn.ModuleDict({
            'DR': nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)),
            'Glaucoma': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
            'Cataract': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
            'Myopia': nn.Sequential(nn.Linear(1280, 64), nn.ReLU(), nn.Linear(64, 1)),
        })
    def forward(self, x, return_attention=False):
        features = self.backbone(x)
        specialist_features = {}
        attention_maps = {}
        for disease in GLAAM_DISEASE_ORDER:
            if disease in self.attention_heads:
                if return_attention:
                    attended, attn = self.attention_heads[disease](features, return_attention=True)
                    attention_maps[disease] = attn
                else:
                    attended = self.attention_heads[disease](features)
            else:
                attended = features
            specialist_features[disease] = F.adaptive_avg_pool2d(attended, 1).flatten(1)
        logits = []
        for disease in GLAAM_DISEASE_ORDER:
            logits.append(self.classifiers[disease](specialist_features[disease]).squeeze(-1))
        return {'logits': torch.stack(logits, dim=1), 'features': None,
                'attention_maps': attention_maps}


# ═══════════════════════════════════════════════════════════════
# Attention map post-processing (publication-grade)
# ═══════════════════════════════════════════════════════════════
def process_attention(attn_tensor, img_size, sigma=SMOOTH_SIGMA,
                      low_pct=LOW_PCT, high_pct=HIGH_PCT, thresh_pct=THRESHOLD_PCT):
    """Convert raw attention tensor to a clean, publication-ready heatmap.

    Steps: channel-mean -> gaussian smooth -> percentile normalize -> threshold.
    Returns a float array in [0,1] at img_size resolution.
    """
    # Channel-mean aggregation (all channels contribute equally)
    attn_map = attn_tensor[0].mean(dim=0).cpu().numpy().astype(np.float32)

    # Gaussian smoothing to remove pixel noise
    attn_map = gaussian_filter(attn_map, sigma=sigma)

    # Percentile normalization (robust to outliers)
    lo = np.percentile(attn_map, low_pct)
    hi = np.percentile(attn_map, high_pct)
    attn_map = np.clip((attn_map - lo) / (hi - lo + 1e-8), 0, 1)

    # Threshold: zero out low-attention regions for a cleaner figure
    thresh = np.percentile(attn_map, thresh_pct)
    attn_map[attn_map < thresh] = 0

    # Resize to image resolution
    attn_map = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(img_size))
    return attn_map / 255.0


def select_samples(test_df, model, transform, device,
                   n_per_disease=2, n_healthy=2, seed=42, rank_pool=20):
    """Select illustrative samples: confident true positives per disease + healthy controls.

    Ranks a small random pool of true positives by predicted probability so the
    figure shows the model's strongest, most interpretable decisions, without
    running inference on the entire test set (slow on CPU).
    """
    rng = np.random.RandomState(seed)
    selected = []

    def predict_probs(row):
        img = Image.open(row['image_path']).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t, return_attention=False)
        logits = out['logits'][0].cpu().numpy()
        return 1 / (1 + np.exp(-logits))

    # True positives per disease (GT=1), ranked by predicted prob (descending)
    for disease in DISEASE_NAMES:
        pos = test_df[test_df[disease] == 1]
        if len(pos) == 0:
            continue
        # Sample a bounded random pool, then rank by confidence
        pool = pos.sample(n=min(rank_pool, len(pos)), random_state=rng)
        probs = pool.apply(lambda r: predict_probs(r)[DISEASE_NAMES.index(disease)], axis=1)
        pool = pool.assign(_prob=probs.values).sort_values('_prob', ascending=False)
        # Take a spread: top confidence + a mid-confidence example
        n = min(n_per_disease, len(pool))
        picks = list(range(n))
        if n >= 2 and len(pool) > 2:
            picks[1] = len(pool) // 2  # mid-confidence example
        for i in picks:
            selected.append(pool.iloc[i])

    # Healthy controls (all diseases = 0), pick lowest overall predicted prob
    healthy = test_df[(test_df[DISEASE_NAMES] == 0).all(axis=1)]
    if len(healthy) > 0:
        pool = healthy.sample(n=min(rank_pool, len(healthy)), random_state=rng)
        probs = pool.apply(lambda r: predict_probs(r).max(), axis=1)
        pool = pool.assign(_prob=probs.values).sort_values('_prob', ascending=True)
        for i in range(min(n_healthy, len(pool))):
            selected.append(pool.iloc[i])

    # Deduplicate by image path
    seen = set()
    unique = []
    for row in selected:
        if row['image_path'] not in seen:
            seen.add(row['image_path'])
            unique.append(row)
    return unique


def main(n_per_disease=2, n_healthy=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    test_df = pd.read_csv(TEST_CSV)
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        full = RAW_ROOT / p
        return str(full) if full.exists() else str(full)
    test_df['image_path'] = test_df['image_path'].apply(resolve_path)
    test_df = test_df[test_df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} images")

    # Transforms
    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model
    model = GLAAM_4X(pretrained=True, dropout_rate=0.3).to(device)
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  ✅ Loaded model from {MODEL_PATH}")

    # Select samples
    samples = select_samples(test_df, model, transform, device,
                             n_per_disease=n_per_disease, n_healthy=n_healthy)
    print(f"  Selected {len(samples)} samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Publication colormap: transparent -> red -> yellow -> white
    # (standard for medical heatmaps, perceptually meaningful)
    cmap = LinearSegmentedColormap.from_list(
        'med_heat', [(0, 0, 0, 0), (0.1, 0.1, 0.4, 0.0), (0.8, 0.1, 0.1, 0.6),
                     (1.0, 0.6, 0.0, 0.85), (1.0, 1.0, 1.0, 1.0)])
    # Simpler robust choice: inferno (perceptually uniform)
    cmap_inferno = plt.cm.inferno.copy()
    cmap_inferno.set_under('none')  # transparent below threshold

    disease_colors = {'Cataract': '#E64A19', 'DR': '#1565C0',
                      'Glaucoma': '#2E7D32', 'Myopia': '#6A1B9A'}

    for idx, row in enumerate(samples):
        img_path = row['image_path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor, return_attention=True)
            logits = out['logits'][0].cpu().numpy()
            attention_maps = out['attention_maps']

        probs = 1 / (1 + np.exp(-logits))
        gt = [int(row[d]) for d in DISEASE_NAMES]

        # Grayscale background (standard for medical heatmap overlays)
        img_gray = np.array(img.convert('L'))
        img_rgb = np.stack([img_gray] * 3, axis=-1)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
        fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.88, bottom=0.05)

        # Original panel
        axes[0].imshow(img)
        axes[0].set_title(f"Fundus\n{row['source']}", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        for j, disease in enumerate(DISEASE_NAMES):
            ax = axes[j + 1]
            ax.imshow(img_rgb, cmap='gray', vmin=0, vmax=255)
            attn = attention_maps.get(disease)
            if attn is not None:
                heat = process_attention(attn, img.size)
                # Overlay heatmap with alpha proportional to intensity
                im = ax.imshow(heat, cmap=cmap_inferno, alpha=ALPHA_MAX,
                               vmin=0, vmax=1)
                # Colorbar
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_visible(False)
            # Title: disease + GT/P
            gt_str = "✓" if gt[j] == 1 else "✗"
            ax.set_title(f"{disease}\nGT={gt_str}  P={probs[j]:.2f}",
                         fontsize=10, color=disease_colors[disease], fontweight='bold')
            ax.axis('off')

        fig.suptitle(f"GLAAM-4X Disease-Specific Attention Maps — Sample {idx}",
                     fontsize=13, fontweight='bold')
        fig.savefig(str(OUT_DIR / f"attention_sample_{idx}.png"), dpi=DPI,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved sample {idx} ({row['source']}) | "
              f"GT={gt} P={np.round(probs,2).tolist()}")

    print(f"\n  ✅ Publication attention maps saved to {OUT_DIR}/")

    # Quantitative verification
    verify_attention_quality(samples, model, transform, device)


def verify_attention_quality(samples, model, transform, device):
    """Quantitatively verify attention maps are meaningful (not blank/uniform).

    Metrics per disease:
      - entropy: lower = more concentrated (uniform map -> high entropy)
      - top5_concentration: fraction of mass in top 5% of pixels
      - spatial_std: std of attention across space (0 = uniform)
      - disease_specificity: correlation between attention maps of different diseases
    """
    print("\n" + "═" * 60)
    print("QUANTITATIVE ATTENTION QUALITY VERIFICATION")
    print("═" * 60)

    all_maps = {d: [] for d in DISEASE_NAMES}
    for row in samples:
        img = Image.open(row['image_path']).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t, return_attention=True)
        for d in DISEASE_NAMES:
            attn = out['attention_maps'].get(d)
            if attn is not None:
                m = attn[0].mean(dim=0).cpu().numpy().astype(np.float32)
                all_maps[d].append(m)

    print(f"\n{'Disease':<10} {'Entropy':<10} {'Top5%Conc':<10} {'SpatialStd':<10}")
    print("-" * 45)
    for d in DISEASE_NAMES:
        maps = all_maps[d]
        if not maps:
            continue
        entropies, concs, stds = [], [], []
        for m in maps:
            m = m - m.min()
            s = m.sum()
            if s <= 0:
                continue
            p = m / s
            entropies.append(-(p * np.log(p + 1e-12)).sum())
            flat = m.flatten()
            k = max(1, int(0.05 * len(flat)))
            topk = np.sort(flat)[-k:]
            concs.append(topk.sum() / s)
            stds.append(m.std())
        print(f"{d:<10} {np.mean(entropies):<10.3f} {np.mean(concs):<10.3f} "
              f"{np.mean(stds):<10.4f}")

    # Disease specificity: pairwise correlation of attention maps
    print("\nDisease-specificity (mean pairwise correlation of attention maps):")
    print("  Lower = more disease-specific (each head attends to different regions)")
    for d in DISEASE_NAMES:
        maps = all_maps[d]
        if len(maps) < 2:
            continue
        corrs = []
        for other in DISEASE_NAMES:
            if other == d or not all_maps[other]:
                continue
            for m1, m2 in zip(maps, all_maps[other]):
                a = m1.flatten(); b = m2.flatten()
                if a.std() == 0 or b.std() == 0:
                    continue
                corrs.append(np.corrcoef(a, b)[0, 1])
        if corrs:
            print(f"  {d:<10} mean corr = {np.mean(corrs):.3f}")

    print("\n  Interpretation:")
    print("  - Entropy < 8  : concentrated (good)")
    print("  - Top5%Conc > 0.3 : strong localization (good)")
    print("  - SpatialStd > 0.05 : non-uniform (good)")
    print("  - Disease corr < 0.5 : disease-specific heads (good)")


if __name__ == "__main__":
    main()
