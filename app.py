"""
app.py  —  GLAAM Fundus Disease Detection Web App
Run: python app.py
Open: http://localhost:5000
"""

import os
import io
import base64
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Flask
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from flask import Flask, request, jsonify, render_template, send_from_directory
from inference_local import (
    load_model, get_transform, temperature_scale,
    compute_eigengradcam, DISEASES, DEFAULT_CHECKPOINT
)

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

UPLOAD_FOLDER = Path('web_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

TEMPERATURES = {'Cataract': 1.286, 'DR': 1.000, 'Glaucoma': 1.213, 'Myopia': 0.471}
DISEASE_INFO = {
    'Cataract':  {'color': '#E74C3C', 'icon': '👁️',  'desc': 'Clouding of the eye lens, causing blurry vision'},
    'DR':        {'color': '#E67E22', 'icon': '🩸',  'desc': 'Diabetic Retinopathy — damage to retinal blood vessels'},
    'Glaucoma':  {'color': '#8E44AD', 'icon': '🔵',  'desc': 'Damage to the optic nerve, often from high eye pressure'},
    'Myopia':    {'color': '#2980B9', 'icon': '🔍',  'desc': 'Nearsightedness — difficulty seeing distant objects'},
}

# ── Load per-disease optimal thresholds (F1-optimised) ────────────────────────
THRESHOLD_PATH = Path(__file__).parent / 'configs' / 'optimal_thresholds.json'
if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH, 'r') as f:
        _thr_data = json.load(f)
    DISEASE_THRESHOLDS = {d: float(v['threshold']) for d, v in _thr_data.items()}
    print(f"Loaded per-disease thresholds: {DISEASE_THRESHOLDS}")
else:
    DISEASE_THRESHOLDS = {d: 0.5 for d in DISEASES}
    print("Warning: optimal_thresholds.json not found, using 0.5 for all diseases")

THRESHOLD = DISEASE_THRESHOLDS.get('Cataract', 0.5)  # legacy fallback

# ── Load model ONCE at startup ────────────────────────────────────────────────
print("Loading GLAAM model...")
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = load_model(DEFAULT_CHECKPOINT, device)
transform = get_transform(224)
print(f"Model ready on {device}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_temperature(logits_arr):
    """logits_arr: (4,) → calibrated probs (4,)"""
    probs = {}
    for i, d in enumerate(DISEASES):
        T = TEMPERATURES[d]
        probs[d] = float(1.0 / (1.0 + np.exp(-logits_arr[i] / T)))
    return probs


def run_inference(image_path: str):
    """Return probs dict + logits from a local image path."""
    with torch.no_grad():
        img    = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        out    = model(tensor)
        logits = out['logits'].cpu().numpy()[0]
    return apply_temperature(logits), logits


def make_heatmap_b64(image_path: str, disease: str) -> str:
    """Generate EigenGradCAM heatmap overlay, return as base64 PNG string."""
    cam_up, method_str = compute_eigengradcam(model, image_path, transform, device, disease)

    img_orig = np.array(Image.open(image_path).convert('RGB').resize((400, 400)))
    cam_384  = np.array(
        Image.fromarray((cam_up * 255).astype(np.uint8)).resize((400, 400), Image.BILINEAR)
    ) / 255.0

    heatmap = plt.get_cmap('viridis')(cam_384)[:, :, :3]
    overlay = np.clip(0.55 * img_orig / 255.0 + 0.45 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='#0d0d1a')
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.88, top=0.88, bottom=0.02)

    panels = [
        (img_orig / 255.0, 'Original'),
        (heatmap,           f'EigenGradCAM'),
        (overlay,           'Overlay'),
    ]
    d_color = DISEASE_INFO[disease]['color']
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.axis('off')
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)

    fig.suptitle(f'{disease} Attention Map  ({method_str})',
                 color=d_color, fontsize=13, fontweight='bold', y=0.97)

    sm   = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01, aspect=20)
    cbar.ax.tick_params(colors='white', labelsize=7)
    cbar.set_label('Attention', color='white', fontsize=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save upload
    ext       = Path(file.filename).suffix.lower() or '.jpg'
    save_path = str(UPLOAD_FOLDER / f'upload_{int(time.time())}{ext}')
    file.save(save_path)

    try:
        t0 = time.time()

        # Run model
        probs, logits = run_inference(save_path)
        flagged = [d for d, p in probs.items() if p >= DISEASE_THRESHOLDS.get(d, 0.5)]

        # Generate heatmaps for ALL diseases (or just flagged if none detected)
        heatmap_targets = flagged if flagged else [max(probs, key=probs.get)]
        heatmaps = {}
        for disease in heatmap_targets:
            heatmaps[disease] = make_heatmap_b64(save_path, disease)

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            'success':   True,
            'probs':     probs,
            'flagged':   flagged,
            'heatmaps':  heatmaps,
            'thresholds': DISEASE_THRESHOLDS,
            'elapsed':   elapsed,
            'device':    str(device),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up upload
        try:
            os.remove(save_path)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
