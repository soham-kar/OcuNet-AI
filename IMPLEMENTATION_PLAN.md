# OcuNet-AI: Implementation Plan

> **Multi-Disease Fundus Classification with Explainable Attention Mechanisms**
>
> Last Updated: 2026-06-06 | Status: Phase 1 Complete ✅

---

## Table of Contents

1. [Project Vision & Goals](#1-project-vision--goals)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Foundation (COMPLETED ✅)](#3-phase-1-foundation-completed-)
4. [Phase 2: Optimization & Scaling (PLANNED)](#4-phase-2-optimization--scaling-planned)
5. [Phase 3: Clinical Validation (PLANNED)](#5-phase-3-clinical-validation-planned)
6. [Phase 4: Deployment & Publication (PLANNED)](#6-phase-4-deployment--publication-planned)
7. [Risk Register](#7-risk-register)
8. [Timeline](#8-timeline)

---

## 1. Project Vision & Goals

### 1.1 Problem Statement

Current AI-based fundus screening systems suffer from three critical limitations:

| Limitation | Impact | Our Solution |
|------------|--------|--------------|
| **Single-disease focus** | Misses comorbidities (e.g., DR + Glaucoma) | Multi-label classification for 4 diseases |
| **Black-box predictions** | Clinicians cannot trust or verify outputs | Disease-specific attention maps (GLAAM) |
| **Dataset fragmentation** | Models trained on single datasets don't generalize | Unified corpus: ODIR + JSIEC + RFMiD + PALM |

### 1.2 Target Diseases

| Disease | Abbreviation | Clinical Significance | Prevalence |
|---------|-------------|----------------------|------------|
| Diabetic Retinopathy | DR | Leading cause of blindness in working-age adults | 26.8% in dataset |
| Glaucoma | G | Silent thief of sight; irreversible damage | 9.9% in dataset |
| Cataract | C | Most common cause of reversible blindness globally | 3.7% in dataset |
| Myopia | M | Rapidly increasing; risk factor for retinal detachment | 8.9% in dataset |

### 1.3 Success Criteria

| Metric | Target | Current (Phase 1) | Status |
|--------|--------|-------------------|--------|
| Macro F1 (4 diseases) | ≥ 0.70 | 0.663 | 🔄 Needs improvement |
| DR AUC | ≥ 0.85 | 0.806 | 🔄 Needs improvement |
| Glaucoma AUC | ≥ 0.90 | 0.907 | ✅ Met |
| Cataract AUC | ≥ 0.90 | 0.951 | ✅ Met |
| Myopia AUC | ≥ 0.90 | 0.962 | ✅ Met |
| Inference time | < 100ms | TBD | ⏳ Not measured |
| Model size | < 50MB | TBD | ⏳ Not measured |

---

## 2. Architecture Overview

### 2.1 GLAAM-4X Model Architecture

```
                    ┌─────────────────────┐
                    │   Fundus Image       │
                    │   (384 × 384 × 3)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   MobileNetV2        │
                    │   (Feature Extractor)│
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼───────┐  ┌──────▼──────┐  ┌───────▼───────┐
    │ MultiScale    │  │  GLAAMBlock │  │  GLAAMBlock   │  ┌────────────┐
    │ GLAAM (DR)    │  │ (Glaucoma)  │  │  (Cataract)   │  │  Identity  │
    │               │  │             │  │               │  │  (Myopia)  │
    └───────┬───────┘  └──────┬──────┘  └───────┬───────┘  └─────┬──────┘
            │                  │                  │               │
    ┌───────▼───────┐  ┌──────▼──────┐  ┌───────▼───────┐  ┌────▼──────┐
    │  DR Head      │  │ Glaucoma    │  │  Cataract     │  │  Myopia   │
    │  (FC + Sigmoid)│  │ Head        │  │  Head         │  │  Head     │
    └───────────────┘  └─────────────┘  └───────────────┘  └───────────┘
            │                  │                  │               │
            └──────────────────┼──────────────────┘               │
                               │                                  │
                    ┌──────────▼──────────┐                       │
                    │  Concatenated Output │                       │
                    │  [DR, G, C, M]       │◄──────────────────────┘
                    └─────────────────────┘
```

### 2.2 Disease Order Mapping

Critical implementation detail — the model and dataset use different disease orderings:

```
Dataset CSV:  [Cataract, DR, Glaucoma, Myopia]  (alphabetical)
Model Output: [DR, Glaucoma, Cataract, Myopia]  (by attention complexity)
REORDER_IDX:  [2, 0, 1, 3]  ← maps model → dataset order
```

### 2.3 Data Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  ODIR-5K │    │  JSIEC   │    │  RFMiD   │    │   PALM   │
│  5,420   │    │   997    │    │  2,560   │    │   400    │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │               │
     └───────────────┼───────────────┼───────────────┘
                     │               │
            ┌────────▼───────────────▼────────┐
            │  create_unified_dataset.py      │
            │  → combined_fundus_dataset.csv  │
            │  → 9,377 images, 0 missing      │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  create_train_val_split.py      │
            │  → train_combined.csv (7,970)   │
            │  → val_combined.csv (1,407)     │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  UnifiedDataset + BalancedSampler│
            │  → Differential augmentation     │
            │  → Class-weighted focal loss     │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  GLAAM-4X Training              │
            │  → Modal T4 GPU (cloud)         │
            │  → AMP + torch.compile + TF32   │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  Evaluation + XAI               │
            │  → AUC, F1, Sensitivity         │
            │  → Grad-CAM++, Attention Maps   │
            └─────────────────────────────────┘
```

---

## 3. Phase 1: Foundation (COMPLETED ✅)

### 3.1 Dataset Construction

**Goal:** Create a unified, clean, multi-source dataset for multi-disease classification.

| Task | Script | Output | Status |
|------|--------|--------|--------|
| Download PALM dataset | `download_palm.py` | 400 glaucoma images | ✅ |
| Combine 4 datasets | `create_unified_dataset.py` | `combined_fundus_dataset.csv` (9,377 rows) | ✅ |
| Stratified split | `create_train_val_split.py` | `train_combined.csv` + `val_combined.csv` | ✅ |
| Upload to Modal cloud | `upload_data_to_modal.py` | Modal volume with data | ✅ |

**Verification Results:**
- ✅ 9,377 total images
- ✅ 0 missing image paths
- ✅ 0 duplicate paths
- ✅ 0 ODIR test set leakage
- ✅ Disease proportions preserved in train/val split

### 3.2 Model Architecture

**Goal:** Build GLAAM-4X with disease-specific attention heads.

| Component | File | Description | Status |
|-----------|------|-------------|--------|
| GLAAM Attention | `models/attention/glaam.py` | Guided Local Attention Augmentation Module | ✅ |
| GLAAI Attention | `models/attention/glaai.py` | Alternative attention variant | ✅ |
| MobileNetV2 Backbone | `models/backbones/backbone.py` | Feature extractor (pretrained) | ✅ |
| GLAAM-4X Classifier | `models/glaam_4x.py` | 4-head multi-disease classifier | ✅ |
| Bayesian Variant | `models/glaam_bayesian.py` | MC Dropout for uncertainty | ✅ |
| Hybrid Model | `models/hybrid_model.py` | CNN + Transformer | ✅ |
| YOLO Lens Detector | `models/yolo/lens_detector.py` | Lens ROI detection | ✅ |

### 3.3 Training Pipeline

**Goal:** Efficient training with class imbalance handling.

| Feature | Implementation | Benefit | Status |
|---------|---------------|---------|--------|
| Mixed Precision | `torch.cuda.amp` (GradScaler + autocast) | 2x faster, less memory | ✅ |
| Graph Compilation | `torch.compile` | 10-20% speedup | ✅ |
| TF32 | `torch.backends.cuda.matmul.allow_tf32` | Faster on Ampere+ GPUs | ✅ |
| SSD Data Copy | Copy to `/tmp` on Modal | Eliminates network I/O | ✅ |
| LR Schedule | LambdaLR: 5-epoch warmup + cosine decay | Stable convergence | ✅ |
| Class-Weighted Loss | `MultiLabelFocalLoss` (α=0.25, γ=2.0) | Handles 3.7% cataract prevalence | ✅ |
| Balanced Sampling | `BalancedBatchSampler` | Per-disease balanced batches | ✅ |
| Differential Augmentation | Per-disease augmentation strength | Prevents overfitting on rare classes | ✅ |
| Resume Support | Checkpoint save/load | Continue interrupted training | ✅ |

### 3.4 Training Results (v2_384)

**Configuration:**
```yaml
model: GLAAM-4X
backbone: MobileNetV2
image_size: 384×384
batch_size: 32
optimizer: AdamW (lr=1e-4, wd=1e-4)
scheduler: LambdaLR (5-epoch warmup + cosine)
loss: MultiLabelFocalLoss (α=0.25, γ=2.0)
epochs: 60
best_epoch: 40
```

**Performance:**

| Disease | AUC | F1 | Precision | Recall | Optimal Threshold |
|---------|-----|-----|-----------|--------|-------------------|
| Cataract | **0.951** | **0.720** | 0.720 | 0.720 | 0.46 |
| DR | 0.806 | 0.541 | 0.477 | 0.626 | 0.24 |
| Glaucoma | **0.907** | 0.574 | 0.574 | 0.574 | 0.26 |
| Myopia | **0.962** | **0.816** | 0.895 | 0.750 | 0.34 |
| **Macro Avg** | **0.907** | **0.663** | — | — | — |

### 3.5 Evaluation & XAI

| Tool | File | Purpose | Status |
|------|------|---------|--------|
| Benchmarking | `evaluation/benchmark.py` | AUC, F1, sensitivity, specificity | ✅ |
| Clinical Validation | `evaluation/clinical_validation.py` | Threshold optimization | ✅ |
| Grad-CAM++ | `evaluation/grad_cam.py` | Attention visualization | ✅ |
| Uncertainty | `evaluation/uncertainty_analysis.py` | MC Dropout uncertainty | ✅ |
| Calibration | `evaluation/temperature_scaling.py` | Model calibration | ✅ |
| Batch XAI | `generate_all_explanations.py` | Generate all explanations | ✅ |
| Publication Figures | `generate_xai_figures.py` | Publication-ready figures | ✅ |

### 3.6 Deployment

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Flask Web App | `app.py` | Browser-based inference | ✅ |
| Local Inference | `inference_local.py` | Command-line inference | ✅ |
| Modal Cloud | `modal_train_glaam4x_unified.py` | Serverless GPU training | ✅ |
| ONNX Export | `scripts/export_onnx.py` | Mobile deployment | ✅ |

---

## 4. Phase 2: Optimization & Scaling (PLANNED)

### 4.1 Model Improvements

| Task | Approach | Expected Impact | Priority |
|------|----------|-----------------|----------|
| **Larger Image Size** | Train at 512×512 | Better fine-detail detection (DR microaneurysms) | 🔴 High |
| **Ensemble Models** | Average 3-5 best checkpoints | +2-5% F1 improvement | 🔴 High |
| **Test-Time Augmentation** | 8-way TTA (flips + rotations) | +1-3% AUC improvement | 🟡 Medium |
| **Label Smoothing** | Soft labels (0.1 smoothing) | Better calibration | 🟡 Medium |
| **SAM Optimizer** | Sharpness-Aware Minimization | Better generalization | 🟢 Low |

### 4.2 Data Improvements

| Task | Approach | Expected Impact | Priority |
|------|----------|-----------------|----------|
| **Cataract Data Augmentation** | Add more cataract sources (e.g., CataractDataset) | Address 3.7% prevalence | 🔴 High |
| **Synthetic DR Lesions** | Use `augmentation/synthetic_dr_lesions.py` | More DR training samples | 🔴 High |
| **Hard Negative Mining** | Focus on misclassified samples | Reduce false positives | 🟡 Medium |
| **Cross-Validation** | 5-fold CV instead of single split | More reliable metrics | 🟡 Medium |

### 4.3 Training Improvements

| Task | Approach | Expected Impact | Priority |
|------|----------|-----------------|----------|
| **Longer Training** | 100-120 epochs with early stopping | Better convergence | 🔴 High |
| **Optuna Hyperparameter Search** | Bayesian optimization (use existing scripts) | Optimal LR, weight decay, dropout | 🔴 High |
| **Gradient Accumulation** | Effective batch_size=64 with accum=2 | Better batch statistics | 🟡 Medium |
| **EMA of Weights** | Exponential Moving Average | Smoother predictions | 🟢 Low |

---

## 5. Phase 3: Clinical Validation (PLANNED)

### 5.1 External Validation

| Task | Dataset | Purpose | Priority |
|------|---------|---------|----------|
| **Hospital Dataset Testing** | Partner hospital fundus images | Real-world performance | 🔴 High |
| **Ophthalmologist Review** | Expert annotation of attention maps | Clinical trustworthiness | 🔴 High |
| **Inter-rater Reliability** | Multiple ophthalmologists | Annotation quality | 🟡 Medium |
| **Sensitivity Analysis** | Subgroup analysis (age, gender, severity) | Fairness assessment | 🟡 Medium |

### 5.2 Explainability Validation

| Task | Method | Purpose | Priority |
|------|--------|---------|----------|
| **Expert IoU** | `evaluation/compute_expert_iou.py` | Attention map accuracy | 🔴 High |
| **Perturbation Analysis** | Remove attended regions, measure drop | Verify attention is meaningful | 🟡 Medium |
| **Counterfactual Explanations** | "What if" analysis | Clinical decision support | 🟢 Low |

---

## 6. Phase 4: Deployment & Publication (PLANNED)

### 6.1 Model Deployment

| Task | Target | Purpose | Priority |
|------|--------|---------|----------|
| **ONNX Optimization** | Quantize to INT8 | < 20MB model size | 🔴 High |
| **Mobile App** | Android/iOS with ONNX Runtime | Field screening | 🟡 Medium |
| **REST API** | FastAPI with Docker | Cloud inference service | 🟡 Medium |
| **Edge Deployment** | NVIDIA Jetson / Raspberry Pi | Low-resource settings | 🟢 Low |

### 6.2 Publication Plan

| Task | Target Venue | Content | Priority |
|------|-------------|---------|----------|
| **Paper Draft** | MICCAI / IEEE TMI | GLAAM-4X architecture + results | 🔴 High |
| **Ablation Study** | Paper section | Attention vs no-attention, per-disease | 🔴 High |
| **Comparison Table** | Paper section | vs ResNet, EfficientNet, ViT baselines | 🔴 High |
| **Figure Generation** | `generate_xai_figures.py` | Publication-ready figures | 🔴 High |
| **Supplementary Material** | Appendix | Full hyperparameters, per-fold results | 🟡 Medium |

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DR performance remains low | Medium | High | More DR data, synthetic lesions, higher resolution |
| Cataract class too small (3.7%) | High | Medium | External cataract datasets, heavy augmentation |
| Overfitting on ODIR (54% of data) | Medium | Medium | Domain adaptation, source-weighted sampling |
| Modal cloud costs escalate | Low | Medium | Local training fallback, spot instances |
| Attention maps not clinically meaningful | Low | High | Expert review in Phase 3, iterative refinement |
| Dataset licensing issues | Low | High | All datasets are publicly available for research |

---

## 8. Timeline

```
Phase 1: Foundation
├── Dataset Construction     ████████████  COMPLETE (May 2026)
├── Model Architecture       ████████████  COMPLETE (May 2026)
├── Training Pipeline        ████████████  COMPLETE (May 2026)
├── Evaluation & XAI         ████████████  COMPLETE (May 2026)
└── GitHub Release           ████████████  COMPLETE (May 2026)

Phase 2: Optimization        [ESTIMATED: 2-3 weeks]
├── Hyperparameter Tuning    ░░░░░░░░░░░░  PLANNED
├── Larger Image Size (512)  ░░░░░░░░░░░░  PLANNED
├── Ensemble Training        ░░░░░░░░░░░░  PLANNED
├── Data Augmentation        ░░░░░░░░░░░░  PLANNED
└── Cross-Validation         ░░░░░░░░░░░░  PLANNED

Phase 3: Clinical Validation [ESTIMATED: 3-4 weeks]
├── Hospital Data Collection ░░░░░░░░░░░░  PLANNED
├── Expert Annotation        ░░░░░░░░░░░░  PLANNED
├── Statistical Analysis     ░░░░░░░░░░░░  PLANNED
└── XAI Validation           ░░░░░░░░░░░░  PLANNED

Phase 4: Publication          [ESTIMATED: 4-6 weeks]
├── Paper Writing            ░░░░░░░░░░░░  PLANNED
├── Figure Generation        ░░░░░░░░░░░░  PLANNED
├── Model Deployment         ░░░░░░░░░░░░  PLANNED
└── Submission               ░░░░░░░░░░░░  PLANNED
```

---

## Appendix A: Key Files Reference

| Category | File | Lines | Purpose |
|----------|------|-------|---------|
| Dataset | `create_unified_dataset.py` | ~150 | Combine 4 datasets |
| Dataset | `create_train_val_split.py` | ~100 | Stratified split |
| Model | `models/glaam_4x.py` | ~300 | Core architecture |
| Model | `models/attention/glaam.py` | ~200 | GLAAM attention |
| Training | `modal_train_glaam4x_unified.py` | ~500 | Cloud training |
| Training | `train_glaam4x_unified.py` | ~400 | Local training |
| Utils | `utils/balanced_sampler.py` | ~390 | Balanced sampling |
| Utils | `utils/losses.py` | ~80 | Focal loss |
| Eval | `evaluation/benchmark.py` | ~200 | Metrics |
| Eval | `evaluation/grad_cam.py` | ~150 | XAI visualization |
| App | `app.py` | ~200 | Flask web app |

## Appendix B: Environment

```yaml
Python: 3.9+
PyTorch: 2.1.0
TorchVision: 0.16.0
CUDA: 12.1 (Modal T4)
GPU: NVIDIA T4 (16GB VRAM)
Cloud: Modal Labs (serverless)
OS: Linux (training), Cross-platform (inference)
```

## Appendix C: Git Commit History

```
f49864d docs(results): add training results and verification report
ac6b53d feat(app): add inference app, scripts, and notebooks
206753a feat(baseline): add baseline XAI models and augmentation
83963f9 feat(training): add legacy training and evaluation scripts
a02be59 feat(models): add additional model variants
0a1805b feat(configs): add model and training configurations
981f2c0 feat(utils): add dataset loaders, losses, and verification tools
3e8d53d feat(evaluation): add clinical evaluation and XAI pipeline
0a95171 feat(training): add GLAAM-4X unified training pipeline
f51ddb8 feat(models): add GLAAM-4X multi-disease classifier architecture
5902d0c feat(utils): add balanced sampler with differential augmentation
14ffdcd feat(dataset): add unified multi-dataset corpus creation
8e7384f feat(project): initialize OcuNet-AI repository
```

---

> **Next Action:** Begin Phase 2 — Hyperparameter optimization with Optuna and 512×512 training.
