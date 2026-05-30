"""
generate_all_explanations.py
-----------------------------
Finds the best representative images from the test set and generates
EigenGradCAM explanation figures for:
  - Single-disease: best confident example per disease
  - Multi-label:    images where model detects 2+ diseases simultaneously

Run: python generate_all_explanations.py
"""

import pickle
import numpy as np
import sys
import io
from pathlib import Path

# ── Inline imports from inference_local (avoiding subprocess) ────────────────
# We import the module directly so there is no subprocess/encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from inference_local import (
    load_model, get_transform, temperature_scale,
    compute_eigengradcam, generate_explanation, DISEASES, DEFAULT_CHECKPOINT
)

# ── Config ───────────────────────────────────────────────────────────────────
PRED_PKL      = 'checkpoints_glaam/glaam_final_predictions.pkl'
LOCAL_IMG_DIR = Path('data/raw/odir/preprocessed_images')
OUT_BASE      = Path('explanations/showcase')
THRESHOLD     = 0.5
TEMPERATURES  = {'Cataract': 1.286, 'DR': 1.000, 'Glaucoma': 1.213, 'Myopia': 0.471}

# ── Load model ───────────────────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = load_model(DEFAULT_CHECKPOINT, device)
transform = get_transform(224)

# ── Load predictions ─────────────────────────────────────────────────────────
print("Loading predictions...")
d = pickle.load(open(PRED_PKL, 'rb'))
disease_names = d['disease_names']
logits        = d['logits']
labels        = d['labels']
paths_raw     = d['paths']

def remap(p):
    fname = Path(p).name
    local = LOCAL_IMG_DIR / fname
    return str(local) if local.exists() else None

paths = [remap(p) for p in paths_raw]

# Apply temperature scaling to get calibrated probs
calibrated = np.zeros_like(logits)
for i, dn in enumerate(disease_names):
    T = TEMPERATURES[dn]
    calibrated[:, i] = 1.0 / (1.0 + np.exp(-logits[:, i] / T))

exists = np.array([p is not None for p in paths])
print(f"Loaded {exists.sum()} locally-available images out of {len(paths)}")

# ── 1. Best SINGLE-DISEASE examples ──────────────────────────────────────────
print("\n" + "="*60)
print("SINGLE-DISEASE EXAMPLES")
print("="*60)

single_targets = {}
for di, disease in enumerate(disease_names):
    # Prefer pure single-disease cases
    pure = (labels[:, di] == 1) & (labels.sum(axis=1) == 1) & exists
    cands = np.where(pure)[0]
    if len(cands) == 0:
        cands = np.where((labels[:, di] == 1) & exists)[0]
    if len(cands) == 0:
        print(f"  {disease}: no local image found, skipping")
        continue
    # Pick highest calibrated probability
    best = cands[np.argsort(calibrated[cands, di])[::-1][0]]
    gt   = [disease_names[j] for j in range(4) if labels[best, j] == 1]
    print(f"  {disease:<12} {Path(paths[best]).name}  prob={calibrated[best,di]:.1%}  GT={gt}")
    single_targets[disease] = best

# ── 2. Best MULTI-LABEL examples ─────────────────────────────────────────────
print("\n" + "="*60)
print("MULTI-DISEASE EXAMPLES (model detects 2+ simultaneously)")
print("="*60)

detected_count = (calibrated >= THRESHOLD).sum(axis=1)
multi_mask     = (detected_count >= 2) & exists
multi_idx      = np.where(multi_mask)[0]
print(f"Found {len(multi_idx)} images with 2+ predicted diseases")

# Score by minimum detection confidence among flagged diseases
scored = []
for idx in multi_idx:
    flagged   = [j for j in range(4) if calibrated[idx, j] >= THRESHOLD]
    min_conf  = min(calibrated[idx, j] for j in flagged)
    diseases  = [disease_names[j] for j in flagged]
    gt        = [disease_names[j] for j in range(4) if labels[idx, j] == 1]
    scored.append((min_conf, idx, diseases, gt))

scored.sort(reverse=True)
top_multi = scored[:3]

for rank, (min_p, idx, detected, gt) in enumerate(top_multi):
    print(f"  [{rank+1}] {Path(paths[idx]).name}  detected={detected}  GT={gt}  min_conf={min_p:.1%}")

# ── 3. Generate ALL explanation figures ──────────────────────────────────────
print("\n" + "="*60)
print("GENERATING EXPLANATION FIGURES")
print("="*60)

generated = []

def make_result(idx):
    """Build result dict compatible with generate_explanation()"""
    probs    = {disease_names[j]: float(calibrated[idx, j]) for j in range(4)}
    flagged  = [disease_names[j] for j in range(4) if calibrated[idx, j] >= THRESHOLD]
    return {
        'path':     paths[idx],
        'diseases': probs,
        'flagged':  flagged,
        'logits':   logits[idx].tolist(),
    }

# --- Single-disease ---
for disease, idx in single_targets.items():
    out_dir = str(OUT_BASE / f'single_{disease}')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result  = make_result(idx)
    print(f"\n  [Single: {disease}]  {Path(paths[idx]).name}")
    probs_str = " | ".join(f"{dn}: {v:.1%}" for dn, v in result['diseases'].items())
    print(f"     Probs: {probs_str}")
    try:
        saved = generate_explanation(model, paths[idx], transform, device, result, out_dir=out_dir)
        print(f"     Saved: {saved}")
        generated.append(saved)
    except Exception as e:
        print(f"     FAILED: {e}")

# --- Multi-label ---
for rank, (min_p, idx, detected, gt) in enumerate(top_multi):
    label   = f'multi_{rank+1}_{"_".join(detected)}'
    out_dir = str(OUT_BASE / label)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result  = make_result(idx)
    print(f"\n  [Multi {rank+1}: {' + '.join(detected)}]  {Path(paths[idx]).name}")
    probs_str = " | ".join(f"{dn}: {v:.1%}" for dn, v in result['diseases'].items())
    print(f"     Probs: {probs_str}")
    try:
        saved = generate_explanation(model, paths[idx], transform, device, result, out_dir=out_dir)
        print(f"     Saved: {saved}")
        generated.append(saved)
    except Exception as e:
        print(f"     FAILED: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"DONE: {len(generated)} / {len(single_targets) + len(top_multi)} figures generated")
print(f"{'='*60}")
for f in generated:
    print(f"  {f}")
