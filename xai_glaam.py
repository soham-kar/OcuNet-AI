"""
xai_glaam.py — Explainable AI for the trained GLAAM fundus model
=================================================================

Produces publication-ready XAI figures for your *trained* checkpoint
(`checkpoints_glaam/glaam_final_best.pth`):

  1. **Per-disease Grad-CAM++**  — 4 heatmaps (Cataract / DR / Glaucoma / Myopia)
  2. **Native GLAAM attention**   — spatial attention from the injected
     GLAAM blocks at stages 13 & 17 (what the model learned to look at)
  3. **Gating / confidence panel** — bar chart + threshold lines
  4. **Clinical hotspot overlay**  — top-5 % attention regions highlighted

Usage
-----
    # Single image
    python xai_glaam.py --image demo_images/cataract_1.jpg

    # Folder of images
    python xai_glaam.py --folder demo_images/

    # Custom checkpoint
    python xai_glaam.py --image fundus.jpg --checkpoint checkpoints_glaam/glaam_final_best.pth

    # Skip the native-attention panel (faster)
    python xai_glaam.py --image fundus.jpg --no-native

Requirements
------------
    pip install torch torchvision pillow numpy matplotlib scipy
    # Optional (sharper CAMs): pip install grad-cam
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# ── Diseases the model detects (CSV / output order) ────────────────────────
DISEASES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

# ── Per-disease temperature scaling (from calibration) ─────────────────────
TEMPERATURES = {
    'Cataract': 1.286,
    'DR':       1.000,
    'Glaucoma': 1.213,
    'Myopia':   0.471,
}

DEFAULT_CHECKPOINT = 'checkpoints_glaam/glaam_final_best.pth'

# Per-disease optimal thresholds (F1-optimised)
THRESHOLD_PATH = Path(__file__).parent / 'configs' / 'optimal_thresholds.json'
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, 'r') as f:
        _thr_data = json.load(f)
    DISEASE_THRESHOLDS = {d: float(v['threshold']) for d, v in _thr_data.items()}
else:
    DISEASE_THRESHOLDS = {d: 0.5 for d in DISEASES}

DISEASE_COLOURS = {
    'Cataract': '#E74C3C',
    'DR':       '#E67E22',
    'Glaucoma': '#8E44AD',
    'Myopia':   '#2980B9',
}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match training exactly — copied from inference_local.py)
# ═══════════════════════════════════════════════════════════════════════════

class GlobalAttentionBranch(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return self.fc(y).view(b, c, 1, 1)


class LocalAttentionBranch(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class GLAAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, use_residual=True, dropout_rate=0.0):
        super().__init__()
        self.global_branch = GlobalAttentionBranch(in_channels, reduction)
        self.local_branch  = LocalAttentionBranch(in_channels, reduction)
        self.use_residual  = use_residual
        self.dropout_rate  = dropout_rate
        self.alpha         = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, return_attention=False):
        global_w = self.global_branch(x)
        local_w  = self.local_branch(x)
        combined = self.alpha * global_w + (1 - self.alpha) * local_w
        out      = x * combined
        if self.dropout_rate > 0 and self.training:
            out = F.dropout(out, p=self.dropout_rate, training=True)
        result = (x + out) if self.use_residual else out
        if return_attention:
            return result, {'combined_attention': combined,
                            'global_weight': global_w,
                            'local_weight':  local_w,
                            'alpha':         self.alpha.item()}
        return result


class MobileNetWithGLAAM(nn.Module):
    STAGE_CHANNELS = {3: 24, 6: 32, 10: 64, 13: 96, 16: 160, 17: 320}

    def __init__(self, n_diseases=4, dropout_rate=0.3,
                 attention_stages=(13, 17), reduction=16, attn_dropout=0.0):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.attention_blocks = nn.ModuleDict()
        for stage in attention_stages:
            if stage in self.STAGE_CHANNELS:
                self.attention_blocks[f'glaam_{stage}'] = GLAAMBlock(
                    self.STAGE_CHANNELS[stage],
                    reduction=reduction,
                    dropout_rate=attn_dropout
                )
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(256, n_diseases)
        )
        self._attention_storage = {}

    def forward(self, x, return_attention=False):
        if return_attention:
            self._attention_storage = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            key = f'glaam_{i}'
            if key in self.attention_blocks:
                if return_attention:
                    x, attn = self.attention_blocks[key](x, return_attention=True)
                    self._attention_storage[f'stage_{i}'] = attn
                else:
                    x = self.attention_blocks[key](x)
        x        = F.adaptive_avg_pool2d(x, 1)
        features = torch.flatten(x, 1)
        logits   = self.classifier(features)
        return {'logits': logits, 'features': features,
                'attention_maps': self._attention_storage}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> MobileNetWithGLAAM:
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MobileNetWithGLAAM(
        n_diseases       = 4,
        dropout_rate     = config.get('dropout_rate', 0.3),
        attention_stages = config.get('attention_stages', [13, 17]),
        reduction        = config.get('reduction_ratio', 16),
        attn_dropout     = config.get('attention_dropout', 0.0),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"✅ Loaded checkpoint  | Val AUC: {ckpt['val_auc']:.4f} | Epoch: {ckpt['epoch']}")
    return model


def get_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def temperature_scale(logits: np.ndarray) -> np.ndarray:
    calibrated = np.zeros_like(logits)
    for i, disease in enumerate(DISEASES):
        T = TEMPERATURES[disease]
        calibrated[:, i] = 1.0 / (1.0 + np.exp(-logits[:, i] / T))
    return calibrated


def predict(model, image_path, transform, device):
    with torch.no_grad():
        img    = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        outputs = model(tensor)
        logits  = outputs['logits'].cpu().numpy()
        probs   = temperature_scale(logits)[0]
    return probs, tensor


# ═══════════════════════════════════════════════════════════════════════════
# XAI METHOD 1 — Grad-CAM++ (per-disease, multi-layer fusion)
# ═══════════════════════════════════════════════════════════════════════════

class GradCAMPlusPlus:
    """Grad-CAM++ with forward/backward hooks on a target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, inp, out):
        self.activations = out.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, input_tensor, disease_idx):
        self.model.eval()
        self.model.zero_grad()
        t = input_tensor.clone().requires_grad_(True)
        logits = self.model(t)['logits']
        target = logits[0, disease_idx]
        self.model.zero_grad()
        target.backward(retain_graph=True)

        grads = self.gradients
        acts  = self.activations
        if grads is None or acts is None:
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        # Grad-CAM++ alpha
        g2 = grads ** 2
        g3 = grads ** 3
        sum_acts = torch.sum(acts * g3, dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha = g2 / (2 * g2 + sum_acts + eps)
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


def compute_per_disease_cam(model, tensor, device, target_layers):
    """Return dict {disease_name: cam_2d} using multi-layer Grad-CAM++ fusion."""
    cams = {}
    for d_idx, disease in enumerate(DISEASES):
        layer_cams = []
        for layer in target_layers:
            cam_fn = GradCAMPlusPlus(model, layer)
            layer_cams.append(cam_fn.generate(tensor, d_idx))
        fused = np.mean(layer_cams, axis=0)
        fused = gaussian_filter(fused.astype(np.float32), sigma=2)
        lo, hi = fused.min(), fused.max()
        fused = (fused - lo) / (hi - lo + 1e-8)
        cams[disease] = fused
    return cams


# ═══════════════════════════════════════════════════════════════════════════
# XAI METHOD 2 — Native GLAAM attention maps (learned spatial attention)
# ═══════════════════════════════════════════════════════════════════════════

def compute_native_attention(model, tensor, device):
    """
    Extract the *learned* spatial attention from GLAAM blocks.
    Returns dict {stage_name: cam_2d} normalised to [0,1].
    """
    with torch.no_grad():
        out = model(tensor.to(device), return_attention=True)
    attn_maps = out['attention_maps']
    results = {}
    for stage_name, attn_dict in attn_maps.items():
        # local_weight is (B, C, H, W) — average over channels → spatial map
        local = attn_dict['local_weight']    # (1, C, H, W)
        spatial = local.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
        spatial = gaussian_filter(spatial, sigma=1)
        lo, hi = spatial.min(), spatial.max()
        spatial = (spatial - lo) / (hi - lo + 1e-8)
        results[stage_name] = spatial
    return results


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def overlay_heatmap(img_np, cam, alpha=0.45, cmap_name='jet'):
    heatmap = plt.get_cmap(cmap_name)(cam)[:, :, :3]
    return np.clip((1 - alpha) * img_np + alpha * heatmap, 0, 1)


def make_full_figure(img_path, probs, cams_gradcampp, native_attn,
                     out_path, include_native=True):
    """
    Publication-ready figure:
      Row 1: Original | Prob bars | Native GLAAM stage-13 | Native GLAAM stage-17
      Row 2: Grad-CAM++ per disease (Cataract / DR / Glaucoma / Myopia)
      Row 3: Overlay per disease + hotspot markers
    """
    img_pil = Image.open(img_path).convert('RGB').resize((384, 384))
    img_np  = np.array(img_pil) / 255.0

    n_native = len(native_attn) if include_native else 0
    # Layout: 4 columns (diseases) + dynamic rows
    n_rows = 3 if include_native and n_native > 0 else 2
    n_cols = max(4, n_native + 2)

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows), facecolor='white')
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.25, wspace=0.1)

    # ── Row 1: Original + probabilities + native attention ─────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_np)
    ax0.set_title('Original Fundus', fontsize=11, fontweight='bold')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    disease_list = list(probs.keys())
    prob_vals    = [probs[d] for d in disease_list]
    bar_colours  = [DISEASE_COLOURS.get(d, '#aaa') for d in disease_list]
    bars = ax1.barh(disease_list, prob_vals, color=bar_colours, height=0.5)
    ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    for bar, val, d in zip(bars, prob_vals, disease_list):
        thr = DISEASE_THRESHOLDS.get(d, 0.5)
        marker = ' ◄' if val >= thr else ''
        ax1.text(min(val + 0.03, 0.88), bar.get_y() + bar.get_height() / 2,
                 f'{val:.0%}{marker}', va='center', fontsize=9, fontweight='bold')
    ax1.set_title('Disease Probabilities', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Confidence', fontsize=9)

    if include_native and n_native > 0:
        for j, (stage_name, attn) in enumerate(native_attn.items()):
            ax = fig.add_subplot(gs[0, 2 + j])
            attn_resized = np.array(Image.fromarray(
                (attn * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
            ax.imshow(overlay_heatmap(img_np, attn_resized, alpha=0.5, cmap_name='viridis'))
            ax.set_title(f'Native GLAAM\n{stage_name}', fontsize=10, fontweight='bold')
            ax.axis('off')

    # ── Row 2: Grad-CAM++ per disease ──────────────────────────────────────
    row_offset = 1
    for j, disease in enumerate(DISEASES):
        ax = fig.add_subplot(gs[row_offset, j])
        cam = cams_gradcampp[disease]
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
        ax.imshow(overlay_heatmap(img_np, cam_resized, alpha=0.5, cmap_name='jet'))
        thr = DISEASE_THRESHOLDS.get(disease, 0.5)
        detected = probs[disease] >= thr
        colour = DISEASE_COLOURS.get(disease, 'black')
        title = f'{disease}\n{probs[disease]:.0%}'
        if detected:
            title += ' ⚠️'
        ax.set_title(title, fontsize=10, fontweight='bold',
                     color=colour if detected else 'black')
        ax.axis('off')

    # ── Row 3: Hotspot overlay (top 5% attention) ──────────────────────────
    row_offset = 2
    for j, disease in enumerate(DISEASES):
        ax = fig.add_subplot(gs[row_offset, j])
        ax.imshow(img_np)
        cam = cams_gradcampp[disease]
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
        smoothed = gaussian_filter(cam_resized, sigma=2)
        threshold = np.percentile(smoothed, 95)
        y, x = np.where(smoothed > threshold)
        if len(x) > 0:
            ax.scatter(x, y, c='lime', s=12, marker='+', alpha=0.8, linewidths=1)
        thr = DISEASE_THRESHOLDS.get(disease, 0.5)
        detected = probs[disease] >= thr
        ax.set_title(f'{disease} Hotspots\n(Top 5%)', fontsize=10,
                     fontweight='bold',
                     color=DISEASE_COLOURS.get(disease, 'black') if detected else 'gray')
        ax.axis('off')

    # ── Suptitle ────────────────────────────────────────────────────────────
    flagged = [d for d in DISEASES if probs[d] >= DISEASE_THRESHOLDS.get(d, 0.5)]
    status = f"⚠️ {', '.join(flagged)} DETECTED" if flagged else '✅ No Disease Detected'
    top_d = max(probs, key=probs.get)
    fig.suptitle(
        f'GLAAM Explainable AI  —  {Path(img_path).name}\n'
        f'{status}  |  Top: {top_d} {probs[top_d]:.0%}  |  '
        f'Grad-CAM++ + Native GLAAM Attention',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def make_compact_figure(img_path, probs, cams_gradcampp, out_path):
    """
    Compact 5-panel figure: Original | 4 disease overlays (Grad-CAM++).
    Best for quick clinical use.
    """
    img_pil = Image.open(img_path).convert('RGB').resize((384, 384))
    img_np  = np.array(img_pil) / 255.0

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), facecolor='white')
    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    for j, disease in enumerate(DISEASES):
        ax = axes[j + 1]
        cam = cams_gradcampp[disease]
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)) / 255.0
        ax.imshow(overlay_heatmap(img_np, cam_resized, alpha=0.5, cmap_name='jet'))
        thr = DISEASE_THRESHOLDS.get(disease, 0.5)
        detected = probs[disease] >= thr
        colour = DISEASE_COLOURS.get(disease, 'black')
        title = f'{disease}\n{probs[disease]:.0%}'
        if detected:
            title += ' ⚠️'
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=colour if detected else 'black')
        ax.axis('off')

    flagged = [d for d in DISEASES if probs[d] >= DISEASE_THRESHOLDS.get(d, 0.5)]
    status = f"⚠️ {', '.join(flagged)}" if flagged else '✅ Normal'
    fig.suptitle(
        f'GLAAM XAI  —  {Path(img_path).name}  |  {status}',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Explainable AI for GLAAM fundus disease detection')
    parser.add_argument('--image',      type=str, default=None,
                        help='Path to a single fundus image')
    parser.add_argument('--folder',     type=str, default=None,
                        help='Path to a folder of fundus images')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--img_size',   type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--device',     type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--no-native',  action='store_true',
                        help='Skip native GLAAM attention panel')
    parser.add_argument('--compact',    action='store_true',
                        help='Also save a compact 5-panel figure')
    parser.add_argument('--out-dir',    type=str, default='xai_figures',
                        help='Folder to save XAI figures (default: xai_figures)')
    args = parser.parse_args()

    # ── Device ─────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️  Device     : {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model     = load_model(args.checkpoint, device)
    transform = get_transform(args.img_size)

    # ── Determine target layers for Grad-CAM++ ─────────────────────────────
    target_layers = []
    for stage_key in ['glaam_13', 'glaam_17']:
        if stage_key in model.attention_blocks:
            # Hook the last conv in the local branch (spatial, most informative)
            target_layers.append(
                model.attention_blocks[stage_key].local_branch.conv[-2]
            )
    if not target_layers:
        target_layers = [list(model.features)[-2]]
    print(f"🎯 CAM layers : {[l.__class__.__name__ for l in target_layers]}")

    # ── Collect images ─────────────────────────────────────────────────────
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.folder:
        folder = Path(args.folder)
        image_paths = sorted(str(p) for p in folder.iterdir()
                             if p.suffix.lower() in img_extensions)
        print(f"📂 Found {len(image_paths)} images in {args.folder}")
    else:
        parser.error("Provide either --image or --folder")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each image ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  GLAAM Explainable AI")
    print(f"  Per-disease thresholds (F1-optimised):")
    for d in DISEASES:
        print(f"    {d:<12} {DISEASE_THRESHOLDS.get(d, 0.5):.0%}")
    print(f"{'═'*60}")

    for path in image_paths:
        try:
            print(f"\n🔍 Processing {Path(path).name}...")
            probs, tensor = predict(model, path, transform, device)
            probs_dict = {d: float(p) for d, p in zip(DISEASES, probs)}

            # Print predictions
            for d, p in zip(DISEASES, probs):
                thr = DISEASE_THRESHOLDS.get(d, 0.5)
                marker = ' ◄ DETECTED' if p >= thr else ''
                print(f"    {d:<12} {p:5.1%}{marker}")

            # Grad-CAM++ per disease
            cams = compute_per_disease_cam(model, tensor, device, target_layers)

            # Native GLAAM attention
            native = {} if args.no_native else compute_native_attention(model, tensor, device)

            # Full figure
            full_path = out_dir / f'{Path(path).stem}_xai_full.png'
            make_full_figure(path, probs_dict, cams, native,
                             str(full_path),
                             include_native=not args.no_native)
            print(f"  📊 Full figure   → {full_path}")

            # Compact figure
            if args.compact:
                compact_path = out_dir / f'{Path(path).stem}_xai_compact.png'
                make_compact_figure(path, probs_dict, cams, str(compact_path))
                print(f"  📊 Compact figure → {compact_path}")

        except Exception as e:
            print(f"  ⚠️  Skipped {Path(path).name}: {e}")

    print(f"\n✅ XAI complete! Figures saved to: {out_dir}/")


if __name__ == '__main__':
    main()