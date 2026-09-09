"""
GLAAM-4X v4 — Fundus Disease Detection Demo
============================================
Run:  python app.py
Open: http://localhost:5000

Upload a retinal fundus image → get disease probabilities as a bar plot
plus an EigenGradCAM heatmap showing where the model looked.
"""

import io
import os
import json
import time
import base64
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, send_from_directory

# ── Model architecture import ───────────────────────────────────────────────
from models.glaam_4x import GLAAM_4X

# ── Constants ─────────────────────────────────────────────────────────────────
DISEASES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
DISEASE_INFO = {
    'Cataract':  {'color': '#E74C3C', 'icon': '👁️', 'desc': 'Clouding of the eye lens — causes blurry vision'},
    'DR':        {'color': '#E67E22', 'icon': '🩸', 'desc': 'Diabetic Retinopathy — retinal blood vessel damage'},
    'Glaucoma':  {'color': '#8E44AD', 'icon': '🔵', 'desc': 'Optic nerve damage, often from high eye pressure'},
    'Myopia':    {'color': '#2980B9', 'icon': '🔍', 'desc': 'Nearsightedness — difficulty seeing distant objects'},
}
IMG_SIZE = 384
WEIGHTS_PATH = 'model_weights.pth'
THRESHOLDS_PATH = 'thresholds.json'

# ── Model wrapper (reorder logits to match UI disease order) ──────────────────
class GLAAM4XClassifier(nn.Module):
    """Wraps GLAAM_4X so output logits are in [Cataract, DR, Glaucoma, Myopia] order."""
    REORDER_IDX = [2, 0, 1, 3]  # native [DR, Glaucoma, Cataract, Myopia] → UI order

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = GLAAM_4X(pretrained=False, dropout_rate=dropout_rate)

    def forward(self, x):
        return self.backbone(x)['logits'][:, self.REORDER_IDX]


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ── Load model & thresholds ONCE at startup ──────────────────────────────────
print("Loading GLAAM-4X v4 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GLAAM4XClassifier(dropout_rate=0.3)
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)

# Strip _orig_mod. prefix left by torch.compile() during training
state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device).eval()
print(f"✅ Model loaded on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")

with open(THRESHOLDS_PATH) as f:
    THRESHOLDS = json.load(f)
print(f"✅ Thresholds: {THRESHOLDS}")

# ── Preprocessing ────────────────────────────────────────────────────────────
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def run_inference(image_path: str) -> dict:
    """Run GLAAM-4X on a single image. Returns probs dict + logits."""
    with torch.no_grad():
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor).cpu().numpy()[0]  # (4,) already reordered

    probs = {}
    for i, disease in enumerate(DISEASES):
        probs[disease] = float(1.0 / (1.0 + np.exp(-logits[i])))  # sigmoid
    return probs, logits


# ═════════════════════════════════════════════════════════════════════════════
# BAR PLOT (the main visual output)
# ═════════════════════════════════════════════════════════════════════════════

def make_bar_plot_b64(image_path: str, probs: dict, thresholds: dict) -> str:
    """
    Generate a side-by-side figure:
      Left:  Original fundus image
      Right: Horizontal bar chart of disease probabilities with threshold markers

    Returns base64-encoded PNG string.
    """
    img_orig = np.array(Image.open(image_path).convert('RGB').resize((384, 384)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
    fig.subplots_adjust(wspace=0.15)

    # ── Left panel: original image ──────────────────────────────────────────
    axes[0].imshow(img_orig)
    axes[0].axis('off')
    axes[0].set_title('Input Fundus Image', color='#1a1d2e', fontsize=14, fontweight='bold', pad=12)

    # ── Right panel: bar chart ──────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('#f7f8fc')

    disease_list = list(probs.keys())
    prob_vals = [probs[d] for d in disease_list]
    bar_colors = [DISEASE_INFO[d]['color'] for d in disease_list]
    y_pos = np.arange(len(disease_list))

    bars = ax.barh(y_pos, prob_vals, color=bar_colors, height=0.55, edgecolor='none', alpha=0.9)

    # Threshold markers (vertical dashed lines)
    for i, d in enumerate(disease_list):
        thr = thresholds.get(d, 0.5)
        ax.plot([thr, thr], [i - 0.3, i + 0.3], color='#7a8299', linewidth=1.5,
                linestyle='--', alpha=0.5)

    # Probability labels on bars
    for i, (d, val) in enumerate(zip(disease_list, prob_vals)):
        thr = thresholds.get(d, 0.5)
        detected = val >= thr
        label = f'{val:.1%}'
        if detected:
            label += '  DETECTED'
        ax.text(val + 0.015, i, label, va='center', ha='left',
                color='#c53030' if detected else '#7a8299',
                fontsize=12, fontweight='bold' if detected else 'normal')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(disease_list, color='#1a1d2e', fontsize=13, fontweight='600')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability', color='#7a8299', fontsize=12)
    ax.set_title('Disease Probabilities', color='#1a1d2e', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(axis='x', colors='#7a8299', labelsize=10)
    ax.tick_params(axis='y', colors='#1a1d2e', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('#e2e6ef')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.15, color='#ccc')

    # Overall status
    flagged = [d for d in disease_list if probs[d] >= thresholds.get(d, 0.5)]
    if flagged:
        status = f"{', '.join(flagged)} DETECTED"
        status_color = '#c53030'
    else:
        status = 'No disease detected above threshold'
        status_color = '#2f855a'

    fig.suptitle(status, color=status_color, fontsize=15, fontweight='bold', y=0.98)

    # Encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ═════════════════════════════════════════════════════════════════════════════
# EIGENGRADCAM HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

try:
    from pytorch_grad_cam import EigenGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_OK = True
except ImportError:
    _GRADCAM_OK = False


class _Wrapper(nn.Module):
    """Thin wrapper so pytorch-grad-cam gets a plain tensor output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def make_heatmap_b64(image_path: str, disease: str) -> str:
    """Generate EigenGradCAM heatmap for a specific disease. Returns base64 PNG."""
    from scipy.ndimage import gaussian_filter

    disease_idx = DISEASES.index(disease)
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    img_orig = np.array(img.resize((400, 400)))

    if _GRADCAM_OK:
        wrapped = _Wrapper(model).eval()

        # Target: the last feature layer of MobileNetV2 backbone
        # This is the shared feature map before disease-specific heads
        target_layers = [model.backbone.backbone[-1]]  # last InvertedResidual block

        targets = [ClassifierOutputTarget(disease_idx)]

        with EigenGradCAM(model=wrapped, target_layers=target_layers) as cam_fn:
            cam = cam_fn(input_tensor=tensor, targets=targets)[0]

        method_str = 'EigenGradCAM'
    else:
        # Fallback: simple activation heatmap from backbone output
        with torch.no_grad():
            feat = model.backbone.backbone(tensor)
            cam = feat.mean(dim=1).squeeze().cpu().numpy()
        method_str = 'Activation map (fallback)'

    # Smooth + normalize
    cam_smooth = gaussian_filter(cam.astype(np.float32), sigma=2)
    lo, hi = cam_smooth.min(), cam_smooth.max()
    cam_norm = (cam_smooth - lo) / (hi - lo + 1e-8)
    cam_400 = np.array(
        Image.fromarray((cam_norm * 255).astype(np.uint8)).resize((400, 400), Image.BILINEAR)
    ) / 255.0

    heatmap = plt.get_cmap('jet')(cam_400)[:, :, :3]
    overlay = np.clip(0.55 * img_orig / 255.0 + 0.45 * heatmap, 0, 1)

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor='white')
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.98, top=0.88, bottom=0.02)

    panels = [
        (img_orig / 255.0, 'Original'),
        (heatmap,          f'{method_str}'),
        (overlay,          'Overlay'),
    ]
    d_color = DISEASE_INFO[disease]['color']
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.axis('off')
        ax.set_title(title, color='#1a1d2e', fontsize=11, fontweight='bold', pad=6)

    fig.suptitle(f'{disease} Attention Map  ({method_str})',
                 color=d_color, fontsize=13, fontweight='bold', y=0.97)

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01, aspect=20)
    cbar.ax.tick_params(colors='#7a8299', labelsize=7)
    cbar.set_label('Attention', color='#7a8299', fontsize=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ═════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo_images/<path:filename>')
def serve_sample(filename):
    return send_from_directory('demo_images', filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = Path(file.filename).suffix.lower() or '.jpg'
    save_path = str(UPLOAD_FOLDER / f'upload_{int(time.time())}{ext}')
    file.save(save_path)

    try:
        t0 = time.time()
        probs, logits = run_inference(save_path)
        flagged = [d for d in DISEASES if probs[d] >= THRESHOLDS.get(d, 0.5)]

        # Bar plot (always generated — this is the main visual)
        bar_plot_b64 = make_bar_plot_b64(save_path, probs, THRESHOLDS)

        # Heatmaps for detected diseases (or top disease if none detected)
        heatmap_targets = flagged if flagged else [max(probs, key=probs.get)]
        heatmaps = {}
        for disease in heatmap_targets:
            try:
                heatmaps[disease] = make_heatmap_b64(save_path, disease)
            except Exception as e:
                print(f"Heatmap error for {disease}: {e}")

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            'success':   True,
            'probs':     probs,
            'flagged':   flagged,
            'thresholds': THRESHOLDS,
            'bar_plot':  bar_plot_b64,
            'heatmaps':  heatmaps,
            'elapsed':   elapsed,
            'device':    str(device),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass


if __name__ == '__main__':
    print(f"\n🚀 Demo running at http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)