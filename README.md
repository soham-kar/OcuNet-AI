# OcuNet-AI: GLAAM-4X — Multi-Disease Fundus Classification with Disease-Specific Attention Specialists

> **A unified deep learning framework for detecting Cataract, Diabetic Retinopathy (DR), Glaucoma, and Myopia from retinal fundus photographs using disease-specific attention mechanisms with SHAP/LIME explainability.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![Modal](https://img.shields.io/badge/Modal-serverless-green.svg)](https://modal.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

**GLAAM-4X** assigns each of four fundus diseases its own **attention specialist** while sharing a common MobileNetV2 backbone:

| Disease | Attention Head | Rationale |
|---------|---------------|-----------|
| **DR** 🩸 | MultiScaleGLAAM (3 scales) | Microaneurysms → hemorrhages (2–50+ px) |
| **Glaucoma** 🔵 | GLAAMBlock (reduction=8) | Optic disc localization |
| **Cataract** 👁️ | GLAAMBlock (reduction=16) | Diffuse lens opacity |
| **Myopia** 🔍 | GLAAMBlock (reduction=32) | Peripapillary atrophy |

**Key results** (3,480 test images across 16 sources):
- **Macro F1: 0.8091**
- Per-disease AUCs: **0.95–0.99**
- Disease-specific attention verified: DR head anti-correlated (−0.88) with others
- Lightweight shared-attention variant: 3.70M params at 0.8150 Macro F1 (70% fewer params)

---

## 🏗️ Project Structure

```
cataract_detection/
├── configs/                    # Hyperparameters & thresholds
│   ├── odir_glaam_final.json
│   ├── glaam_optuna_best.json
│   └── optimal_thresholds.json
├── models/
│   ├── glaam_4x.py            # GLAAM-4X architecture
│   ├── glaam_bayesian.py       # Bayesian variant
│   ├── hybrid_model.py         # Legacy YOLO-GLAAM
│   ├── attention/              # GLAAM, GLAAMBlock modules
│   ├── backbones/              # MobileNetV2 etc.
│   └── yolo/                   # Legacy YOLO detector
├── evaluation/                 # Grad-CAM, calibration, uncertainty
├── training/                   # Local training scripts (legacy)
├── modal_*.py                  # 🔥 **Modal scripts** (primary workflow)
├── app.py                      # 🌐 Flask web demo
├── inference_local.py          # 🖥️ Local inference with EigenGradCAM
├── GLAAM4X_PAPER.md            # 📄 Full manuscript draft
├── generate_attention_maps.py  # GLAAM intrinsic attention maps
├── generate_xai_figures.py     # SHAP/LIME local generation
├── notebooks/                  # Jupyter notebooks for analysis
├── xai_figures/                # Output figures
├── explanations/               # Generated explanations
├── checkpoints_glaam/          # Trained model weights
├── calibration_results/        # Temperature scaling
└── templates/
    └── index.html              # Web app frontend
```

---

## 🚀 Quick Start

### 1. Local Inference (Web Demo)

```bash
# Install dependencies
pip install -r requirements_local.txt

# Run the Flask web app
python app.py
# → Open http://localhost:5000
```

Upload a fundus photo and get per-disease probabilities with an EigenGradCAM heatmap overlay showing where the model detected pathology.

### 2. Local Inference (CLI)

```bash
python inference_local.py --image path/to/fundus.jpg
python inference_local.py --folder path/to/images/
```

### 3. Generate SHAP + LIME Explanations (Local)

```bash
python generate_xai_figures.py --method both --n-samples 4
```

### 4. Generate GLAAM Attention Maps

```bash
python generate_attention_maps.py --n-samples 10
```

---

## ☁️ Modal Training & Analysis

All training and analysis runs on **[Modal](https://modal.com)** serverless GPUs. Volumes:
- `cataract-data` — datasets and CSVs
- `cataract-checkpoints` — trained models and results

### Training

```bash
# Full GLAAM-4X training
modal run modal_train_glaam4x.py --epochs 40

# Unified training script
modal run modal_train_glaam4x_unified.py
```

### Publication Analysis (Ablation + Baselines + Significance)

```bash
# Run everything (7 ablation + 4 baseline models)
modal run modal_publication_analysis.py

# Analysis only (using saved results)
modal run modal_publication_analysis.py --analysis-only

# Skip baselines
modal run modal_publication_analysis.py --skip-baselines
```

### Explainability (SHAP + LIME)

```bash
# Generate both SHAP and LIME explanations
modal run modal_shap_lime.py

# LIME only
modal run modal_shap_lime.py --method lime --n-samples 4
```

### Evaluation

```bash
modal run modal_evaluate_glaam.py
modal run modal_eval_glaam4x.py
```

### Download Results

```bash
modal volume get cataract-checkpoints glaam4x_v6_winning_recipe/shap_lime/ ./shap_lime/ --recursive
```

---

## 📊 Key Results

### A. Per-Disease Performance

| Disease | AUC | F1 | Precision | Recall |
|---------|-----|----|-----------|--------|
| Cataract | 0.9927 | 0.8890 | 0.9070 | 0.8718 |
| DR | 0.9511 | 0.8579 | 0.8745 | 0.8420 |
| Glaucoma | 0.9706 | 0.8645 | 0.8910 | 0.8396 |
| Myopia | 0.9915 | 0.6248 | 0.9750 | 0.4596 |
| **Macro** | — | **0.8091** | — | — |

### B. Ablation Study

| Variant | Params (M) | Macro F1 | ∆ F1 |
|---------|-----------|----------|------|
| **A1 Full GLAAM-4X** | **12.52** | **0.8091** | 0.0 |
| A2 No MultiScale DR | 12.52 | 0.8082 | −0.0009 |
| A3 No Disease Gate | 12.52 | 0.8032 | −0.0059 |
| A4 No Attention | 12.52 | 0.7997 | −0.0094 |
| A5 Shared Attention | **3.70** | **0.8150** | **+0.0059** |
| A6 No Warmup | 12.52 | 0.8055 | −0.0036 |

### C. Dataset Composition

16 sources, **30,439 train** / 3,871 val / **3,480 test** images:

| Source | Samples | Diseases |
|--------|---------|----------|
| ODIR | 5,000 | Cataract, DR, Glaucoma, Myopia |
| ODIR-5K | 5,000 | 4-disease |
| DDR | 2,391 | DR |
| RFMiD | 1,920 | 4-disease |
| JSIEC | 500 | 4-disease |
| PALM | 400 | Myopia |
| PAPILA | 400 | Glaucoma |
| IDRiD | 254 | DR |
| ACRIMA | 353 | Glaucoma |
| RIM-ONE | 313 | Glaucoma |
| LAG | 2,699 | Glaucoma |
| *Synthetic* | ~12,000 | Cataract, quality augmentation |

### D. Significance (DR AUC)

| Comparison | ∆ DR AUC | p-value |
|-----------|----------|---------|
| A1 vs A2 (No MultiScale) | +0.0091 | **< 0.0001** |
| A1 vs A4 (No Attention) | +0.0071 | **< 0.0001** |
| A1 vs A5 (Shared) | +0.0102 | **< 0.0001** |
| A1 vs A3 (No Gate) | +0.0010 | 0.461 (n.s.) |

---

## 🔬 Explainability

GLAAM-4X offers **three complementary** explanation methods:

| Method | Type | What it shows |
|--------|------|---------------|
| **GLAAM Attention** | Intrinsic | Where each disease head looks (spatial attention activations) |
| **SHAP** | Post-hoc | Pixel-level feature attribution (red = pushes prediction up) |
| **LIME** | Post-hoc | Superpixel evidence (green = positive, red = negative) |

Quantitative attention verification (Table VIII in paper):
- DR head is anti-correlated (−0.88) with other disease heads
- Cataract, Glaucoma, Myopia heads show low inter-correlation (0.20–0.22)
- **Confirms each disease specialist learns a distinct spatial prior** without anatomical supervision

---

## 📦 Model Weights & Checkpoints

| Model | File | Params |
|-------|------|--------|
| GLAAM-4X (best) | `checkpoints_glaam/glaam_final_best.pth` | 12.52M |
| Shared-attention variant | From publication analysis | 3.70M |
| Multitask baseline | `checkpoints_modal/multitask_model.pth` | — |
| GLAAM calibrator | `calibration_results/glaam_calibrator.json` | — |

---

## 📝 Paper

The full manuscript is in **[GLAAM4X_PAPER.md](./GLAAM4X_PAPER.md)** with:
- Abstract, Introduction (Research Gaps & Contributions)
- Related Work, Methodology
- Experiments (Ablation, Baselines, Significance, Cross-Dataset, Failure, Efficiency)
- Interpretability (Disease-Specific Attention Verification)
- Discussion & Limitations
- 8 tables, references

---

## 📄 License

MIT
