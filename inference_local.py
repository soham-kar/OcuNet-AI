"""
inference_local.py
------------------
Run GLAAM fundus disease detection locally on one image or a folder.
Generates a visual explanation (EigenGradCAM heatmap) showing WHERE in the
fundus the model detected the disease — answering "why should I believe you?"

Usage:
    # Single image  → saves explanation PNG automatically
    python inference_local.py --image path/to/fundus.jpg

    # Folder of images
    python inference_local.py --folder path/to/images/

    # Custom threshold / skip visuals
    python inference_local.py --image fundus.jpg --threshold 0.3
    python inference_local.py --image fundus.jpg --no-explain

Requirements:
    pip install torch torchvision pillow numpy matplotlib grad-cam scipy
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
import matplotlib.cm as cm


# ── Diseases the model detects ─────────────────────────────────────────────
DISEASES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

# ── Temperature scaling values (from calibration) ──────────────────────────
# Lower T  → probabilities spread out (model was underconfident)
# Higher T → probabilities compressed (model was overconfident)
TEMPERATURES = {
    'Cataract': 1.286,
    'DR':       1.000,
    'Glaucoma': 1.213,
    'Myopia':   0.471,
}

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = 'checkpoints_glaam/glaam_final_best.pth'

# Load per-disease optimal thresholds (computed to maximise F1)
# Falls back to 0.5 if the JSON is missing.
THRESHOLD_PATH = Path(__file__).parent / 'configs' / 'optimal_thresholds.json'
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, 'r') as f:
        _thr_data = json.load(f)
    DISEASE_THRESHOLDS = {d: float(v['threshold']) for d, v in _thr_data.items()}
else:
    DISEASE_THRESHOLDS = {d: 0.5 for d in DISEASES}

DEFAULT_THRESHOLD = DISEASE_THRESHOLDS.get('Cataract', 0.5)  # legacy compat


# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════════

class GlobalAttentionBranch(nn.Module):
    """Channel-wise global attention via squeeze-and-excitation."""
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
    """Spatial local attention via 1×1 convolutions."""
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
    """
    Global-Local Attention-Aware Module.
    Fuses channel (global) and spatial (local) attention with a learnable alpha.
    """
    def __init__(self, in_channels, reduction=16, use_residual=True, dropout_rate=0.0):
        super().__init__()
        self.global_branch  = GlobalAttentionBranch(in_channels, reduction)
        self.local_branch   = LocalAttentionBranch(in_channels, reduction)
        self.use_residual   = use_residual
        self.dropout_rate   = dropout_rate
        self.alpha          = nn.Parameter(torch.tensor(0.5))   # learnable fusion weight

    def forward(self, x, return_attention=False):
        global_w  = self.global_branch(x)                               # (B,C,1,1)
        local_w   = self.local_branch(x)                                # (B,C,H,W)
        combined  = self.alpha * global_w + (1 - self.alpha) * local_w
        out       = x * combined

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
    """
    MobileNetV2 backbone with GLAAM blocks injected at specified stages.
    Mirrors the training definition exactly.
    """
    # Channel counts per MobileNetV2 inverted-residual stage index
    STAGE_CHANNELS = {3: 24, 6: 32, 10: 64, 13: 96, 16: 160, 17: 320}

    def __init__(self, n_diseases=4, dropout_rate=0.3,
                 attention_stages=(13, 17), reduction=16, attn_dropout=0.0):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)          # load weights from checkpoint
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
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> MobileNetWithGLAAM:
    """Load GLAAM model from checkpoint. Architecture auto-read from saved config."""
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = MobileNetWithGLAAM(
        n_diseases        = 4,                                   # always 4 after rare-disease filter
        dropout_rate      = config.get('dropout_rate', 0.3),
        attention_stages  = config.get('attention_stages', [13, 17]),
        reduction         = config.get('reduction_ratio', 16),
        attn_dropout      = config.get('attention_dropout', 0.0),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"✅ Loaded checkpoint  | Val AUC: {ckpt['val_auc']:.4f} | Epoch: {ckpt['epoch']}")
    return model


def get_transform(img_size: int = 224) -> transforms.Compose:
    """Preprocessing pipeline — must match training exactly."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def temperature_scale(logits: np.ndarray) -> np.ndarray:
    """Apply per-disease temperature scaling (calibration from val set)."""
    calibrated = np.zeros_like(logits)
    for i, disease in enumerate(DISEASES):
        T = TEMPERATURES[disease]
        calibrated[:, i] = 1.0 / (1.0 + np.exp(-logits[:, i] / T))
    return calibrated


# ═══════════════════════════════════════════════════════════════════════════
# CAM VISUALIZATION  (Improvement 1: EigenGradCAM + multi-layer fusion)
# ═══════════════════════════════════════════════════════════════════════════

# Try to import pytorch-grad-cam library (pip install grad-cam)
try:
    from pytorch_grad_cam import EigenGradCAM, GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_LIB = True
except ImportError:
    _GRADCAM_LIB = False


class _GLAAMWrapper(torch.nn.Module):
    """
    Thin wrapper so pytorch-grad-cam can call the model.
    It expects model(x) to return a plain tensor (logits), not a dict.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


def compute_eigengradcam(model, image_path: str, transform, device,
                         target_disease: str) -> tuple[np.ndarray, str]:
    """
    Generate attention heatmap using EigenGradCAM fused across two GLAAM layers.

    Improvements over basic Grad-CAM:
      1. EigenGradCAM  — uses SVD of activations instead of mean-pooled
                         gradients → sharper, less noisy
      2. Multi-layer   — fuses stage-13 (fine features) + stage-17 (coarse)
                         → captures both lesion-level and region-level signals
      3. Gaussian blur — sigma=2 smoothing to remove quantisation artefacts

    Returns:
        cam        : np.ndarray (H,W) normalised [0,1]
        method_str : human-readable method name used
    """
    from scipy.ndimage import gaussian_filter
    disease_idx = DISEASES.index(target_disease)

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    img_size = img.size  # (W, H)

    if _GRADCAM_LIB:
        # ── pytorch-grad-cam path ───────────────────────────────────────────
        wrapped = _GLAAMWrapper(model)
        wrapped.eval()

        # Determine which GLAAM layers exist
        target_layers = []
        for stage_key in ['glaam_13', 'glaam_17']:
            if stage_key in model.attention_blocks:
                # Hook onto the local_branch conv (spatial, most informative)
                target_layers.append(
                    model.attention_blocks[stage_key].local_branch.conv[-2]
                )

        if not target_layers:                        # safety fallback
            target_layers = [list(wrapped.model.features)[-2]]

        targets = [ClassifierOutputTarget(disease_idx)]

        # Compute per-layer EigenGradCAM maps then average
        layer_cams = []
        for layer in target_layers:
            with EigenGradCAM(model=wrapped, target_layers=[layer]) as cam_fn:
                layer_cam = cam_fn(input_tensor=tensor, targets=targets)[0]  # (H,W)
                layer_cams.append(layer_cam)

        # Fuse: equal-weight average of all layers
        cam_fused = np.mean(layer_cams, axis=0)
        method_str = f'EigenGradCAM (layers: {len(target_layers)})'

    else:
        # ── Manual Grad-CAM fallback (no library) ──────────────────────────
        gradients_, activations_ = [], []

        def fwd_hook(m, inp, out): activations_.append(out)
        def bwd_hook(m, gi, go):   gradients_.append(go[0])

        if 'glaam_17' in model.attention_blocks:
            tgt = model.attention_blocks['glaam_17']
        elif 'glaam_13' in model.attention_blocks:
            tgt = model.attention_blocks['glaam_13']
        else:
            tgt = list(model.features)[-2]

        h1 = tgt.register_forward_hook(fwd_hook)
        h2 = tgt.register_full_backward_hook(bwd_hook)

        t = tensor.clone().requires_grad_(True)
        model.zero_grad()
        logits = model(t)['logits']
        logits[0, disease_idx].backward()
        h1.remove(); h2.remove()

        if gradients_ and activations_:
            grads   = gradients_[0].detach().cpu()
            acts    = activations_[0].detach().cpu()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam_fused = F.relu((weights * acts).sum(dim=1).squeeze()).numpy()
        else:
            cam_fused = np.zeros((7, 7))
        method_str = 'GradCAM (fallback)'

    # ── Improvement 3: Gaussian smoothing ──────────────────────────────────
    cam_smooth = gaussian_filter(cam_fused.astype(np.float32), sigma=2)

    # Normalise to [0, 1]
    lo, hi = cam_smooth.min(), cam_smooth.max()
    cam_norm = (cam_smooth - lo) / (hi - lo + 1e-8)

    # Upsample to original image size
    cam_up = np.array(
        Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(img_size, Image.BILINEAR)
    ) / 255.0

    return cam_up, method_str



def generate_explanation(model, image_path: str, transform, device,
                         result: dict, out_dir: str = 'explanations') -> str:
    """
    Generate and save a 4-panel visual explanation figure:
      [Original]  [Heatmap]  [Overlay]  [Probability bars]

    Upgrades vs v1:
      - EigenGradCAM + multi-layer fusion (stage 13 + 17)
      - Gaussian-smoothed heatmap (no blocky artefacts)
      - Viridis colormap (perceptually uniform, journal-accepted)
    """
    import matplotlib.gridspec as gridspec

    Path(out_dir).mkdir(exist_ok=True)
    img_orig = np.array(Image.open(image_path).convert('RGB').resize((384, 384)))

    probs = result['diseases']
    top_d = max(probs, key=probs.get)
    top_p = probs[top_d]

    # ── Improvement 1+2+3: EigenGradCAM + multi-layer + smoothing ──────────
    cam_up, method_str = compute_eigengradcam(
        model, image_path, transform, device, top_d
    )
    # Resize CAM to figure size
    cam_384 = np.array(
        Image.fromarray((cam_up * 255).astype(np.uint8)).resize((384, 384), Image.BILINEAR)
    ) / 255.0

    # ── Improvement 3: Viridis colormap (perceptually uniform) ─────────────
    cmap_name = 'viridis'
    heatmap   = plt.get_cmap(cmap_name)(cam_384)[:, :, :3]
    overlay   = np.clip(0.55 * img_orig / 255.0 + 0.45 * heatmap, 0, 1)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 4.5), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

    disease_colours = {
        'Cataract': '#E74C3C', 'DR': '#E67E22',
        'Glaucoma': '#8E44AD', 'Myopia': '#2980B9'
    }
    panel_data = [
        (img_orig / 255.0,  'Original Fundus'),
        (heatmap,           f'{method_str}'),
        (overlay,           'Overlay'),
    ]

    for col, (panel, title) in enumerate(panel_data):
        ax = fig.add_subplot(gs[col])
        ax.imshow(panel)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.axis('off')

    # ── Bar chart ───────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor('#12122a')
    disease_list = list(probs.keys())
    prob_vals    = [probs[d] for d in disease_list]
    bar_colours  = [disease_colours.get(d, '#aaaaaa') for d in disease_list]
    bars = ax4.barh(disease_list, prob_vals, color=bar_colours, height=0.5, edgecolor='none')
    ax4.set_xlim(0, 1)
    ax4.axvline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    for bar, val in zip(bars, prob_vals):
        ax4.text(min(val + 0.03, 0.92), bar.get_y() + bar.get_height() / 2,
                 f'{val:.0%}', va='center', color='white', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Confidence', color='white', fontsize=10)
    ax4.tick_params(colors='white', labelsize=10)
    ax4.spines[:].set_color('#333355')
    ax4.set_title('Disease Probabilities', color='white', fontsize=11, fontweight='bold', pad=8)

    # ── Main title ──────────────────────────────────────────────────────────
    status = f"⚠️  {', '.join(result['flagged'])} DETECTED" if result['flagged'] else '✅ No Disease Detected'
    fig.suptitle(
        f"GLAAM Fundus Analysis  —  {Path(image_path).name}\n"
        f"{status}  |  Top: {top_d} {top_p:.0%}",
        color='white', fontsize=13, fontweight='bold', y=1.02
    )

    # ── Colorbar (Viridis) ──────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.01, aspect=25)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.set_label('Attention intensity', color='white', fontsize=8)

    out_path = str(Path(out_dir) / (Path(image_path).stem + '_explanation.png'))
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def predict_single(model, image_path: str, transform, device,
                     thresholds: dict = None) -> dict:
    """Run inference on a single image, return formatted results.

    Args:
        thresholds: per-disease decision thresholds.  If None, uses
                    DISEASE_THRESHOLDS loaded from configs/optimal_thresholds.json.
    """
    if thresholds is None:
        thresholds = DISEASE_THRESHOLDS

    with torch.no_grad():
        img    = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(tensor)
        logits  = outputs['logits'].cpu().numpy()
        probs   = temperature_scale(logits)[0]

    results = {
        'path':     image_path,
        'diseases': {},
        'flagged':  [],
        'logits':   logits[0].tolist(),
    }
    for disease, prob in zip(DISEASES, probs):
        results['diseases'][disease] = float(prob)
        thr = thresholds.get(disease, 0.5)
        if prob >= thr:
            results['flagged'].append(disease)

    return results


def print_result(r: dict):
    """Pretty-print prediction for one image (uses per-disease thresholds)."""
    fname = Path(r['path']).name
    print(f"\n{'─'*55}")
    print(f"  Image : {fname}")
    print(f"{'─'*55}")
    for disease, prob in r['diseases'].items():
        bar    = '█' * int(prob * 20)
        thr    = DISEASE_THRESHOLDS.get(disease, 0.5)
        marker = ' ◄ DETECTED' if prob >= thr else ''
        print(f"  {disease:<12} {prob:5.1%}  {bar}{marker}")
    print(f"{'─'*55}")
    if r['flagged']:
        print(f"  ⚠️  FLAGGED : {', '.join(r['flagged'])}")
    else:
        print(f"  ✅  NORMAL  : No disease detected above per-disease thresholds")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='GLAAM local inference for fundus images')
    parser.add_argument('--image',      type=str, default=None,
                        help='Path to a single fundus image')
    parser.add_argument('--folder',     type=str, default=None,
                        help='Path to a folder of fundus images')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--threshold',  type=float, default=DEFAULT_THRESHOLD,
                        help=f'Detection threshold 0-1 (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--img_size',   type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--device',     type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--no-explain', action='store_true',
                        help='Skip generating visual explanation images')
    parser.add_argument('--out-dir',    type=str, default='explanations',
                        help='Folder to save explanation images (default: explanations)')
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

    # ── Collect images ─────────────────────────────────────────────────────
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths    = []

    if args.image:
        image_paths.append(args.image)
    elif args.folder:
        folder = Path(args.folder)
        image_paths = [str(p) for p in folder.iterdir()
                       if p.suffix.lower() in img_extensions]
        image_paths.sort()
        print(f"📂 Found {len(image_paths)} images in {args.folder}")
    else:
        parser.error("Provide either --image or --folder")

    # ── Run inference ──────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  GLAAM Fundus Disease Detection")
    print(f"  Per-disease thresholds (F1-optimised):")
    for d in DISEASES:
        print(f"    {d:<12} {DISEASE_THRESHOLDS.get(d, 0.5):.0%}")
    print(f"{'═'*55}")

    results        = []
    flagged_count  = 0

    for path in image_paths:
        try:
            r = predict_single(model, path, transform, device)
            print_result(r)
            results.append(r)
            if r['flagged']:
                flagged_count += 1
            # Generate visual explanation
            if not args.no_explain:
                exp_path = generate_explanation(
                    model, path, transform, device, r, out_dir=args.out_dir)
                print(f"  📊 Explanation saved → {exp_path}")
        except Exception as e:
            print(f"\n  ⚠️  Skipped {Path(path).name}: {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'═'*55}")
        print(f"  SUMMARY")
        print(f"{'═'*55}")
        print(f"  Images processed : {len(results)}")
        print(f"  Flagged          : {flagged_count} ({flagged_count/len(results):.0%})")
        print(f"\n  Mean probabilities across all images:")
        for disease in DISEASES:
            mean_p = np.mean([r['diseases'][disease] for r in results])
            print(f"    {disease:<12} {mean_p:5.1%}")


if __name__ == '__main__':
    main()
