# modal_publication_analysis.py
"""
Modal Script: Publication Analysis for GLAAM-4X
=================================================

Runs ALL critical experiments needed for a top-journal submission:
  1. Ablation Study (7 variants — A1 uses pre-trained demo model)
  2. Baseline Comparison (4 baselines — B5 uses pre-trained demo model)
  3. Statistical Significance (DeLong + Bootstrap CI + McNemar)
  4. FLOPs / Parameter Count
  5. Failure Case Analysis
  6. Cross-Dataset Generalization
  7. Publication-Ready Summary Tables & Figures

The pre-trained demo model (demo/model_weights.pth, 48MB) is automatically
uploaded and used as the reference (A1_Full_GLAAM4X and B5_GLAAM_4X_Ours).
This saves ~8-10 hours of training — only 7 ablation + 4 baseline models
need to be trained from scratch (11 models total instead of 13).

All models use the SAME augmentation pipeline (torchvision v2 with
ElasticTransform, GaussianNoise, RandomErasing, strong minority aug)
as the Colab training notebook, ensuring fair comparison.

Default: 40 epochs (close to the demo model's 48 epochs for fairness).
This takes ~80-100 hours total on T4 — split across multiple sessions.

Usage:
    # Run everything (ablation + baselines + analysis)
    modal run modal_publication_analysis.py

    # Run only ablation (skip baselines) — 7 models
    modal run modal_publication_analysis.py --skip-baselines

    # Run only baselines (skip ablation) — 4 models
    modal run modal_publication_analysis.py --skip-ablation

    # Run only the analysis parts (uses saved results from previous runs)
    modal run modal_publication_analysis.py --analysis-only

    # Custom epochs (default: 40 — matches demo model's training duration for fair comparison)
    modal run modal_publication_analysis.py --epochs 40

    # Use A10G GPU (faster, more expensive)
    modal run modal_publication_analysis.py --gpu A10G

Requirements:
    - Modal volume "cataract-data" with pre-parsed CSVs:
      /data/train_v4.csv, /data/val_tune_v4.csv, /data/test_v4.csv
      (columns: image_path, Cataract, DR, Glaucoma, Myopia, source)
    - Modal volume "cataract-checkpoints" for saving results
    - Local file: demo/model_weights.pth (best trained GLAAM-4X)
    - Local file: demo/thresholds.json (per-disease optimal thresholds)
"""

import modal
import json
import argparse
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Modal Image — all dependencies baked in
# ═══════════════════════════════════════════════════════════════
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "numpy<2.0",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "scikit-learn",
    "pandas",
    "tqdm",
    "pillow",
    "matplotlib",
    "scipy",
    "thop",           # FLOPs counting
    "opencv-python-headless",
    "safetensors",
])

app = modal.App("glaam-publication-analysis", image=image)

# Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
IMG_SIZE = 384
BATCH_SIZE = 64  # increased from 32 — T4 has 15.6GB VRAM, only 5.7GB used at bs=32
DEFAULT_EPOCHS = 40
WARMUP_EPOCHS = 10  # proportional to 40 total (same ratio as Colab notebook's 10/60)


# ═══════════════════════════════════════════════════════════════
# MAIN REMOTE FUNCTION — runs on Modal GPU
# ═══════════════════════════════════════════════════════════════
@app.function(
    gpu="T4",
    cpu=8,           # 8 CPUs to support 8 DataLoader workers for max data loading speed
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=86400,  # 24 hours
    memory=16384,   # 16GB RAM
)
def run_publication_analysis(
    epochs: int = DEFAULT_EPOCHS,
    skip_ablation: bool = False,
    skip_baselines: bool = False,
    analysis_only: bool = False,
):
    """
    Run all publication analysis experiments on Modal GPU.
    Results saved to /checkpoints/publication_analysis/
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import pickle
    import json
    import time
    import os
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torchvision import models, transforms
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        roc_curve, confusion_matrix, precision_recall_curve,
        average_precision_score, ConfusionMatrixDisplay
    )
    from scipy import stats
    from tqdm import tqdm
    from PIL import Image
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Speed optimizations for data loading ────────────────────────────────
    cv2.setNumThreads(0)  # avoid OpenCV thread contention with DataLoader workers
    torch.set_num_threads(8)  # match CPU count for 8 DataLoader workers
    Image.MAX_IMAGE_PIXELS = None  # suppress DecompressionBomb warnings

    # ── Export training graphs + inference package for the full model ──────
    # Mirrors modal_train_glaam4x.py Sections 12-13 so the retrained A1 gets
    # the same diagnostic graphs and a deployable inference package.
    def _export_full_model_artifacts(a1_result, out_dir):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, \
            average_precision_score, confusion_matrix, ConfusionMatrixDisplay

        graphs_dir = out_dir / "training_graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)

        hist = a1_result['history']
        epochs_hist = hist['epoch']
        best_epoch = a1_result.get('best_epoch', epochs_hist[-1] if epochs_hist else 0)
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_hist, hist['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best ({best_epoch})')
        ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss'); ax.set_title('Training Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "01_loss_curve.png")); plt.close()

        # Macro F1
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_hist, hist['val_macro_f1_opt'], 'green', linewidth=2, label='Val F1 (opt)')
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1'); ax.set_title('Macro F1 Progression')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "02_macro_f1.png")); plt.close()

        # Per-disease AUC
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, d in enumerate(DISEASE_NAMES):
            ax.plot(epochs_hist, hist['val_auc'][d], linewidth=2, color=colors[i], label=d)
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('AUC'); ax.set_title('Per-Disease Validation AUC')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "03_per_disease_auc.png")); plt.close()

        # Per-disease F1
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, d in enumerate(DISEASE_NAMES):
            ax.plot(epochs_hist, hist['val_f1'][d], linewidth=2, color=colors[i], label=d)
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel('F1'); ax.set_title('Per-Disease Validation F1')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "04_per_disease_f1.png")); plt.close()

        # ROC curves (test set)
        test_labels = np.array(a1_result['test_labels'])
        test_probs = np.array(a1_result['test_probs'])
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, d in enumerate(DISEASE_NAMES):
            y_true, y_prob = test_labels[:, i], test_probs[:, i]
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = roc_auc_score(y_true, y_prob)
                ax.plot(fpr, tpr, linewidth=2.5, color=colors[i], label=f'{d} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Chance')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('Test Set ROC Curves'); ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "05_roc_curves.png")); plt.close()

        # PR curves (test set)
        fig, ax = plt.subplots(figsize=(8, 7))
        for i, d in enumerate(DISEASE_NAMES):
            y_true, y_prob = test_labels[:, i], test_probs[:, i]
            if len(np.unique(y_true)) > 1:
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                ax.plot(rec, prec, linewidth=2.5, color=colors[i], label=f'{d} (AP={ap:.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Test Set Precision-Recall Curves'); ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(str(graphs_dir / "06_pr_curves.png")); plt.close()

        # Confusion matrices (test set)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, d in enumerate(DISEASE_NAMES):
            y_true = test_labels[:, i].astype(int)
            y_prob = test_probs[:, i]
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
            disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
            axes[i].set_title(d, fontsize=13)
        fig.suptitle('Test Set Confusion Matrices', fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(str(graphs_dir / "07_confusion_matrices.png")); plt.close()

        # ── Inference package ───────────────────────────────────────────────
        inf_dir = out_dir / "inference_package"
        inf_dir.mkdir(parents=True, exist_ok=True)
        # Copy the clean model weights
        import shutil
        shutil.copy(str(out_dir / "best_model.pth"), str(inf_dir / "model_weights.pth"))
        # Save thresholds
        with open(inf_dir / "thresholds.json", 'w') as f:
            json.dump(a1_result.get('best_thresholds', {d: 0.5 for d in DISEASE_NAMES}), f, indent=2)
        # Save model info
        model_info = {
            "model_name": "glaam4x_v6_winning_recipe", "architecture": "GLAAM-4X",
            "backbone": "MobileNetV2", "num_classes": 4,
            "disease_names": DISEASE_NAMES, "img_size": IMG_SIZE,
            "dropout_rate": 0.3, "best_epoch": best_epoch,
            "best_val_f1": a1_result.get('best_val_f1', 0.0),
            "myopia_attention": True, "recipe": "BCE + pos_weight + moderate augmentation",
            "test_macro_f1": a1_result.get('test_macro_f1', 0.0),
        }
        with open(inf_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"  ✅ Training graphs + inference package saved to {out_dir}/")

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT DIRECTORIES
    # ═══════════════════════════════════════════════════════════════
    RESULTS_DIR = Path("/checkpoints/publication_analysis_v2")
    ABLATION_DIR = RESULTS_DIR / "ablation"
    BASELINE_DIR = RESULTS_DIR / "baselines"
    SIGNIFICANCE_DIR = RESULTS_DIR / "significance"
    FIGURES_DIR = RESULTS_DIR / "figures"
    for d in [RESULTS_DIR, ABLATION_DIR, BASELINE_DIR, SIGNIFICANCE_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # DATA LOADING — use pre-parsed CSVs from the volume
    # The volume already has train_v4.csv, val_tune_v4.csv, test_v4.csv
    # with columns: image_path, Cataract, DR, Glaucoma, Myopia, source
    # Image paths are relative to /data/raw/ (e.g. "odir/preprocessed_images/xxx.jpg")
    # ═══════════════════════════════════════════════════════════════
    print("\n📁 Loading pre-parsed dataset CSVs from volume...")

    DATA_ROOT = Path("/data")
    RAW_ROOT = DATA_ROOT / "raw"  # image paths resolve relative to /data/raw/

    train_csv = DATA_ROOT / "train_v4.csv"
    val_csv = DATA_ROOT / "val_tune_v4.csv"
    test_csv = DATA_ROOT / "test_v4.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found!\n"
            f"Expected pre-parsed CSVs at /data/train_v4.csv, /data/val_tune_v4.csv, /data/test_v4.csv\n"
            f"Upload the unified dataset CSVs to the 'cataract-data' Modal volume."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Fix image paths: CSV paths are relative to /data/raw/
    # e.g. "odir/preprocessed_images/1282_right.jpg" → "/data/raw/odir/preprocessed_images/1282_right.jpg"
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        full = RAW_ROOT / p
        if full.exists():
            return str(full)
        # Try /data/ directly (some paths might be relative to /data/)
        full2 = DATA_ROOT / p
        if full2.exists():
            return str(full2)
        return str(full)  # return best guess even if not found

    for df_ in [train_df, val_df, test_df]:
        df_['image_path'] = df_['image_path'].apply(resolve_path)

    # Filter out images that don't exist on the volume (some sources like DDR may not be uploaded)
    print("   Filtering for existing images...")
    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df_)
        existing_mask = df_['image_path'].apply(os.path.exists)
        missing_sources = df_[~existing_mask]['source'].value_counts().to_dict() if 'source' in df_.columns else {}
        df_ = df_[existing_mask].reset_index(drop=True)
        if name == "train": train_df = df_
        elif name == "val": val_df = df_
        else: test_df = df_
        print(f"   {name}: {before} → {len(df_)} (removed {before - len(df_)})")
        if missing_sources:
            print(f"      Missing sources: {missing_sources}")

    if len(train_df) < 100:
        raise FileNotFoundError(
            f"Only {len(train_df)} training images found after filtering. "
            f"Check that image paths in the CSVs match the volume structure."
        )

    print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"   Disease distribution (train): {train_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"   Disease distribution (val):   {val_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"   Disease distribution (test):  {test_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"   Sources in train: {train_df['source'].value_counts().to_dict()}")
    print(f"   Sample path: {train_df['image_path'].iloc[0]}")

    # ═══════════════════════════════════════════════════════════════
    # SHARED INFRASTRUCTURE
    # ═══════════════════════════════════════════════════════════════

    # ── Dataset ─────────────────────────────────────────────────────
    # Uses the SAME torchvision v2 augmentation pipeline as the Colab training notebook
    # to ensure fair comparison with the demo model
    from torchvision.transforms import v2

    class GaussianNoiseTransform(nn.Module):
        """Custom Gaussian noise (torchvision v2 has no built-in)."""
        def __init__(self, std_range=(0.04, 0.2), p=0.3):
            super().__init__()
            self.std_range = std_range
            self.p = p
        def forward(self, img):
            if torch.rand(1).item() > self.p:
                return img
            std = torch.empty(1).uniform_(*self.std_range).item()
            return torch.clamp(img + torch.randn_like(img) * std, 0.0, 1.0)

    def _build_transforms(img_size, is_train, strong_aug=True):
        """Build transforms matching the Colab notebook exactly."""
        base_tail = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if not is_train:
            return v2.Compose([v2.Resize((img_size, img_size))] + base_tail), None

        # Base training transform (same as Colab notebook)
        train_t = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply([v2.RandomChoice([
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ])], p=0.3),
            v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            v2.RandomApply([v2.ColorJitter(hue=0.03, saturation=0.2)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        ] + base_tail + [
            GaussianNoiseTransform(std_range=(0.04, 0.2), p=0.3),
            v2.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
        ])

        # Strong augmentation for minority class (same as Colab notebook)
        strong_t = None
        if strong_aug:
            strong_t = v2.Compose([
                v2.Resize((img_size, img_size)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.3),
                v2.RandomApply([v2.RandomChoice([
                    v2.RandomRotation((90, 90)),
                    v2.RandomRotation((180, 180)),
                    v2.RandomRotation((270, 270)),
                ])], p=0.3),
                v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
                v2.RandomApply([v2.ColorJitter(hue=0.05, saturation=0.3)], p=0.4),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.3),
            ] + base_tail + [
                GaussianNoiseTransform(std_range=(0.08, 0.3), p=0.4),
                v2.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
            ])

        return train_t, strong_t

    class FundusDataset(Dataset):
        def __init__(self, df, disease_cols, img_size, is_train=False, strong_aug=True):
            self.df = df.reset_index(drop=True)
            self.disease_cols = disease_cols
            self.img_size = img_size
            self.is_train = is_train
            self.minority_aug_prob = 0.3 if (is_train and strong_aug) else 0.0
            self.transform, self.strong_transform = _build_transforms(img_size, is_train, strong_aug)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            labels = row[self.disease_cols].values.astype(np.float32)
            try:
                img = Image.open(row['image_path']).convert('RGB')
            except Exception:
                # Robustness: skip corrupted/unreadable images by returning a
                # zero image with the same labels (prevents a single bad file
                # from crashing the entire DataLoader).
                img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            if self.is_train and labels.sum() > 0 and np.random.rand() < self.minority_aug_prob:
                img = self.strong_transform(img)
            else:
                img = self.transform(img)
            return img, torch.tensor(labels)

    # ── Metrics ────────────────────────────────────────────────────
    def compute_metrics(logits, labels, thresholds=None):
        probs = 1 / (1 + np.exp(-logits))
        if thresholds is None:
            thresholds = {d: 0.5 for d in DISEASE_NAMES}
        metrics = {}
        for i, disease in enumerate(DISEASE_NAMES):
            y_true, y_prob = labels[:, i], probs[:, i]
            thr = thresholds.get(disease, 0.5)
            y_pred = (y_prob >= thr).astype(int)
            metrics[disease] = {
                'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
            }
        metrics['macro_f1'] = np.mean([m['f1'] for m in metrics.values()])
        return metrics, probs

    def find_optimal_thresholds(logits, labels):
        probs = 1 / (1 + np.exp(-logits))
        thresholds = {}
        for i, disease in enumerate(DISEASE_NAMES):
            y_true, y_prob = labels[:, i], probs[:, i]
            best_f1, best_thr = 0, 0.5
            for thr in np.arange(0.05, 0.95, 0.01):
                f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            thresholds[disease] = float(best_thr)
        return thresholds

    # ── Sampler ────────────────────────────────────────────────────
    def get_sample_weights(df, disease_cols):
        pos_counts = df[disease_cols].sum().values
        neg_counts = len(df) - pos_counts
        pos_w = np.sqrt(1.0 / (pos_counts + 1e-6)); pos_w = pos_w / pos_w.sum()
        neg_w = np.sqrt(1.0 / (neg_counts + 1e-6)); neg_w = neg_w / neg_w.sum()
        weights = np.zeros(len(df))
        for i, row in df.iterrows():
            w = sum(pos_w[j] if row[c] == 1 else neg_w[j] for j, c in enumerate(disease_cols))
            weights[i] = w / len(disease_cols)
        return weights

    # ── Training & evaluation ──────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def train_epoch(model, loader, criterion, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images); loss = criterion(logits, labels)
                if not torch.isfinite(loss): scaler.update(); continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(images); loss = criterion(logits, labels)
                if not torch.isfinite(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            total_loss += loss.item()
            all_logits.append(logits.detach().cpu()); all_labels.append(labels.cpu())
        scheduler.step()
        return total_loss / len(loader), torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            logits = model(images.to(device))
            all_logits.append(logits.cpu()); all_labels.append(labels)
        return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

    # ── Experiment runner ──────────────────────────────────────────
    def run_experiment(name, model, criterion, save_dir, strong_aug=True,
                      use_warmup=True, ep=epochs, extra_info=None):
        print(f"\n{'='*60}\n  EXPERIMENT: {name}\n{'='*60}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        model = model.to(device)

        # ── Per-epoch checkpoint path (for preemption recovery) ─────────────
        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        resume_ckpt = ckpt_dir / f"{name}_resume.pth"

        train_ds = FundusDataset(train_df, DISEASE_NAMES, IMG_SIZE, is_train=True, strong_aug=strong_aug)
        val_ds   = FundusDataset(val_df, DISEASE_NAMES, IMG_SIZE, is_train=False)
        test_ds  = FundusDataset(test_df, DISEASE_NAMES, IMG_SIZE, is_train=False)

        sw = get_sample_weights(train_df, DISEASE_NAMES)
        sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(train_df)*2, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8,
                                  pin_memory=True, drop_last=True, persistent_workers=True,
                                  prefetch_factor=4)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                 pin_memory=True, persistent_workers=True)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=5e-4)
        warmup = WARMUP_EPOCHS if use_warmup else 0
        def lr_lambda(epoch):
            if epoch < warmup: return float(epoch + 1) / float(max(1, warmup))
            return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup) / max(1, ep - warmup)))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # ── Resume from checkpoint if available ─────────────────────────────
        start_epoch = 1
        best_val_f1, best_epoch, best_thr = 0.0, 0, {d: 0.5 for d in DISEASE_NAMES}
        best_model_path = ckpt_dir / f"{name}_best.pth"
        history = {'epoch': [], 'train_loss': [], 'val_macro_f1_opt': [],
                   'val_auc': {d: [] for d in DISEASE_NAMES},
                   'val_f1': {d: [] for d in DISEASE_NAMES}}
        if resume_ckpt.exists():
            print(f"  📂 Found resume checkpoint — loading...")
            ckpt = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=ckpt['epoch'])
            start_epoch = ckpt['epoch'] + 1
            best_val_f1 = ckpt.get('best_val_f1', 0.0)
            best_epoch = ckpt.get('best_epoch', 0)
            best_thr = ckpt.get('best_thresholds', {d: 0.5 for d in DISEASE_NAMES})
            history = ckpt.get('history', history)
            print(f"  ✅ Resumed from epoch {start_epoch} | Best F1 so far: {best_val_f1:.4f}")
        else:
            print(f"  No resume checkpoint found — starting from epoch 1")

        for epoch in range(start_epoch, ep + 1):
            t_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, scheduler)
            v_logits, v_labels = evaluate(model, val_loader)
            opt_thr = find_optimal_thresholds(v_logits, v_labels)
            v_metrics_opt, _ = compute_metrics(v_logits, v_labels, opt_thr)
            if v_metrics_opt['macro_f1'] > best_val_f1:
                best_val_f1 = v_metrics_opt['macro_f1']; best_epoch = epoch; best_thr = opt_thr
                # Save the BEST model separately (for final evaluation)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_f1': best_val_f1,
                    'best_epoch': best_epoch,
                    'best_thresholds': best_thr,
                }, str(best_model_path))
            if epoch % 5 == 0 or epoch == ep:
                print(f"  Ep {epoch}/{ep} | Loss {t_loss:.4f} | Val F1 {v_metrics_opt['macro_f1']:.4f} (best {best_val_f1:.4f})")

            # ── Record history ─────────────────────────────────────────────
            history['epoch'].append(epoch)
            history['train_loss'].append(t_loss)
            history['val_macro_f1_opt'].append(v_metrics_opt['macro_f1'])
            for d in DISEASE_NAMES:
                history['val_auc'][d].append(v_metrics_opt[d]['auc'])
                history['val_f1'][d].append(v_metrics_opt[d]['f1'])

            # ── Save resume checkpoint every 2 epochs (preemption protection) ──
            if epoch % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'best_epoch': best_epoch,
                    'best_thresholds': best_thr,
                    'history': history,
                }, str(resume_ckpt))
                checkpoint_volume.commit()  # persist to volume

        # ── Final test evaluation (use the BEST model, not the last epoch) ──
        if best_model_path.exists():
            best_ckpt = torch.load(str(best_model_path), map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
            model.eval()
            print(f"  📂 Loaded best model (epoch {best_epoch}) for test evaluation")
        t_logits, t_labels = evaluate(model, test_loader)
        test_metrics, test_probs = compute_metrics(t_logits, t_labels, best_thr)
        print(f"\n  ✅ {name} | Best Val F1: {best_val_f1:.4f} (ep {best_epoch}) | Test Macro F1: {test_metrics['macro_f1']:.4f}")
        for d in DISEASE_NAMES:
            m = test_metrics[d]; print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f}")

        result = {
            'name': name, 'n_params': n_params,
            'best_val_f1': best_val_f1, 'best_epoch': best_epoch,
            'test_macro_f1': test_metrics['macro_f1'], 'test_metrics': test_metrics,
            'best_thresholds': best_thr,
            'test_logits': t_logits.tolist(), 'test_labels': t_labels.tolist(),
            'test_probs': test_probs.tolist(),
            'history': history,
            'extra_info': extra_info or {},
        }
        save_path = save_dir / f"{name}.json"
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {save_path}")

        # ── Save reusable model weights (unwrapped GLAAM_4X state_dict) ─────
        # The model is a GLAAM4XWrapper; save the inner backbone so it can be
        # loaded directly with GLAAM_4X(...) for future inference/XAI.
        try:
            inner = model.backbone if hasattr(model, 'backbone') else model
            model_ckpt = {
                'model_state_dict': inner.state_dict(),
                'best_val_f1': best_val_f1,
                'best_epoch': best_epoch,
                'best_thresholds': best_thr,
                'test_macro_f1': test_metrics['macro_f1'],
                'test_metrics': test_metrics,
                'recipe': 'BCE + pos_weight + moderate augmentation',
                'extra_info': extra_info or {},
            }
            model_weights_path = save_dir / f"{name}_model.pth"
            torch.save(model_ckpt, str(model_weights_path))
            print(f"  ✅ Saved model weights: {model_weights_path}")
        except Exception as e:
            print(f"  ⚠️ Could not save model weights: {e}")

        # ── Clean up resume checkpoint (experiment complete) ────────────────
        if resume_ckpt.exists():
            resume_ckpt.unlink()
            checkpoint_volume.commit()

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    # ═══════════════════════════════════════════════════════════════
    # MODEL DEFINITIONS (inline for Modal)
    # ═══════════════════════════════════════════════════════════════

    # ── GLAAM components ───────────────────────────────────────────
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
            gw = self.global_branch(x); lw = self.local_branch(x)
            combined = self.alpha * gw + (1 - self.alpha) * lw
            out = x * combined
            result = (x + out) if self.use_residual else out
            if return_attention:
                return result, combined
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
                return output, (fine + medium_up + coarse_up) / 3
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
    REORDER_IDX = [2, 0, 1, 3]  # [DR, Glaucoma, Cataract, Myopia] → [Cataract, DR, Glaucoma, Myopia]

    class GLAAM_4X(nn.Module):
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.backbone = mobilenet.features
            self.attention_heads = nn.ModuleDict({
                'DR': MultiScaleGLAAM(1280, reduction=4),
                'Glaucoma': GLAAMBlock(1280, reduction=8),
                'Cataract': GLAAMBlock(1280, reduction=16),
                'Myopia': GLAAMBlock(1280, reduction=32),  # FIXED: matches v5 trained model
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

    class GLAAM4XWrapper(nn.Module):
        def __init__(self, backbone_model):
            super().__init__()
            self.backbone = backbone_model
        def forward(self, x):
            return self.backbone(x)['logits'][:, REORDER_IDX]

    # ── Ablation variants ─────────────────────────────────────────
    class GLAAM_4X_NoMultiScale(GLAAM_4X):
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__(pretrained=pretrained, dropout_rate=dropout_rate)
            self.attention_heads['DR'] = GLAAMBlock(1280, reduction=8)

    class GLAAM_4X_NoGating(GLAAM_4X):
        def forward(self, x, return_attention=False):
            features = self.backbone(x)
            specialist_features = {}
            for disease in GLAAM_DISEASE_ORDER:
                if disease in self.attention_heads:
                    attended = self.attention_heads[disease](features)
                else:
                    attended = features
                specialist_features[disease] = F.adaptive_avg_pool2d(attended, 1).flatten(1)
            logits = [self.classifiers[d](specialist_features[d]).squeeze(-1) for d in GLAAM_DISEASE_ORDER]
            return {'logits': torch.stack(logits, dim=1), 'features': None, 'attention_maps': {}}

    class GLAAM_4X_NoAttention(GLAAM_4X):
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__(pretrained=pretrained, dropout_rate=dropout_rate)
            self.attention_heads = nn.ModuleDict()

    class GLAAM_4X_SharedAttention(GLAAM_4X):
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__(pretrained=pretrained, dropout_rate=dropout_rate)
            shared = GLAAMBlock(1280, reduction=16)
            self.attention_heads = nn.ModuleDict({
                'DR': shared, 'Glaucoma': shared, 'Cataract': shared, 'Myopia': shared,
            })

    # ── Baseline models ────────────────────────────────────────────
    class PlainMobileNetV2(nn.Module):
        def __init__(self, n_diseases=4, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), nn.Linear(256, n_diseases))
        def forward(self, x):
            return self.classifier(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels), nn.Sigmoid())
        def forward(self, x):
            b, c, _, _ = x.shape
            w = F.adaptive_avg_pool2d(x, 1).view(b, c)
            return x * self.fc(w).view(b, c, 1, 1)

    class MobileNetV2_SE(nn.Module):
        def __init__(self, n_diseases=4, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features
            self.se13 = SEBlock(96); self.se17 = SEBlock(320)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), nn.Linear(256, n_diseases))
        def forward(self, x):
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 13: x = self.se13(x)
                elif i == 17: x = self.se17(x)
            return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, bias=False))
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            avg = self.fc(F.adaptive_avg_pool2d(x, 1))
            max_ = self.fc(F.adaptive_max_pool2d(x, 1))
            return x * self.sigmoid(avg + max_)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            avg = torch.mean(x, dim=1, keepdim=True)
            max_, _ = torch.max(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

    class CBAMBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.ca = ChannelAttention(channels, reduction)
            self.sa = SpatialAttention()
        def forward(self, x):
            return self.sa(self.ca(x))

    class MobileNetV2_CBAM(nn.Module):
        def __init__(self, n_diseases=4, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features
            self.cbam13 = CBAMBlock(96); self.cbam17 = CBAMBlock(320)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), nn.Linear(256, n_diseases))
        def forward(self, x):
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 13: x = self.cbam13(x)
                elif i == 17: x = self.cbam17(x)
            return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    class ECABlock(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            b, c, _, _ = x.shape
            w = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)
            w = self.conv(w).view(b, c, 1, 1)
            return x * self.sigmoid(w)

    class MobileNetV2_ECA(nn.Module):
        def __init__(self, n_diseases=4, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=True)
            self.features = mobilenet.features
            self.eca13 = ECABlock(96); self.eca17 = ECABlock(320)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), nn.Linear(256, n_diseases))
        def forward(self, x):
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 13: x = self.eca13(x)
                elif i == 17: x = self.eca17(x)
            return self.classifier(F.adaptive_avg_pool2d(x, 1).flatten(1))

    # ── Loss functions ─────────────────────────────────────────────
    class AsymmetricLoss(nn.Module):
        def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8, pos_weight=None):
            super().__init__()
            self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos
            self.clip = clip; self.eps = eps
            self.pos_weight = pos_weight  # FIX: matches v5 training recipe
        def forward(self, logits, targets):
            xs_pos = logits; pt = torch.sigmoid(xs_pos).clamp(min=self.eps, max=1-self.eps)
            pw = self.pos_weight if self.pos_weight is not None else 1.0
            pos_loss = targets * pw * torch.pow(1 - pt, self.gamma_pos) * F.logsigmoid(xs_pos)
            xs_neg = -logits; p_neg = torch.sigmoid(xs_neg).clamp(min=self.eps, max=1-self.eps)
            neg_loss = (1 - targets) * torch.pow(1 - p_neg, self.gamma_neg) * F.logsigmoid(xs_neg)
            if self.clip > 0:
                probs = torch.sigmoid(logits)
                neg_loss = neg_loss * (~((targets == 0) & (probs < self.clip))).float()
            return (-pos_loss - neg_loss).mean()

    def bce():
        # Winning recipe: BCE with pos_weight for class imbalance
        pos_counts = train_df[DISEASE_NAMES].sum().values
        neg_counts = len(train_df) - pos_counts
        pw = torch.tensor(np.sqrt(neg_counts / (pos_counts + 1e-6)), dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Model factories ────────────────────────────────────────────
    def make_full():       return GLAAM4XWrapper(GLAAM_4X(pretrained=True, dropout_rate=0.3))
    def make_no_ms():      return GLAAM4XWrapper(GLAAM_4X_NoMultiScale(pretrained=True, dropout_rate=0.3))
    def make_no_gating():  return GLAAM4XWrapper(GLAAM_4X_NoGating(pretrained=True, dropout_rate=0.3))
    def make_no_attn():    return GLAAM4XWrapper(GLAAM_4X_NoAttention(pretrained=True, dropout_rate=0.3))
    def make_shared():     return GLAAM4XWrapper(GLAAM_4X_SharedAttention(pretrained=True, dropout_rate=0.3))
    def make_plain():      return PlainMobileNetV2(n_diseases=4, dropout_rate=0.3)
    def make_se():         return MobileNetV2_SE(n_diseases=4, dropout_rate=0.3)
    def make_cbam():       return MobileNetV2_CBAM(n_diseases=4, dropout_rate=0.3)
    def make_eca():        return MobileNetV2_ECA(n_diseases=4, dropout_rate=0.3)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: ABLATION STUDY
    # All variants trained with the WINNING RECIPE (BCE + pos_weight + moderate aug)
    # ═══════════════════════════════════════════════════════════════
    ablation_results = {}

    if not skip_ablation and not analysis_only:
        ABLATION_EXPERIMENTS = [
            ("A1_Full_GLAAM4X",      make_full,      bce, False, True,  {"removed": "nothing"}),
            ("A2_No_MultiScale",     make_no_ms,     bce, False, True,  {"removed": "MultiScaleGLAAM for DR"}),
            ("A3_No_Disease_Gating", make_no_gating, bce, False, True,  {"removed": "DiseaseGatingNetwork"}),
            ("A4_No_Attention",      make_no_attn,   bce, False, True,  {"removed": "all attention heads"}),
            ("A5_Shared_Attention",  make_shared,    bce, False, True,  {"removed": "disease-specific heads"}),
            ("A6_No_Warmup",         make_full,      bce, False, False, {"removed": "warmup phase"}),
        ]

        for name, model_fn, crit_fn, strong_aug, use_warmup, extra in ABLATION_EXPERIMENTS:
            save_path = ABLATION_DIR / f"{name}.json"
            if save_path.exists():
                print(f"\n✓ {name} already completed — loading")
                with open(save_path) as f:
                    ablation_results[name] = json.load(f)
                continue
            print(f"\n{'#'*60}\n# ABLATION: {name}\n{'#'*60}")
            result = run_experiment(name, model_fn(), crit_fn(), ABLATION_DIR,
                                   strong_aug=strong_aug, use_warmup=use_warmup, extra_info=extra)
            ablation_results[name] = result
            checkpoint_volume.commit()  # Save after each experiment
    else:
        # Load existing results
        for f in ABLATION_DIR.glob("*.json"):
            with open(f) as fh:
                ablation_results[f.name.replace('.json', '')] = json.load(fh)
        print(f"Loaded {len(ablation_results)} existing ablation results")

    # ── Save the retrained full model (A1) to a clean named checkpoint ─────
    # This gives a reusable, recognizable model for future inference/XAI.
    a1_weights = ABLATION_DIR / "A1_Full_GLAAM4X_model.pth"
    if a1_weights.exists():
        final_model_dir = Path("/checkpoints") / "glaam4x_v6_winning_recipe"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt_path = final_model_dir / "best_model.pth"
        # Copy the A1 weights to the named checkpoint
        import shutil
        shutil.copy(str(a1_weights), str(final_ckpt_path))
        # Also write a small config/readme
        with open(final_model_dir / "config.json", 'w') as f:
            json.dump({
                "model_name": "glaam4x_v6_winning_recipe",
                "recipe": "BCE + pos_weight + moderate augmentation",
                "source": "A1_Full_GLAAM4X (retrained full GLAAM-4X)",
                "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
                "disease_order": DISEASE_NAMES,
            }, f, indent=2)
        checkpoint_volume.commit()
        print(f"  ✅ Final model saved to {final_ckpt_path}")

        # ── Generate training graphs + inference package for the full model ──
        # This gives A1 the same "full treatment" as modal_train_glaam4x.py
        a1_result = ablation_results.get('A1_Full_GLAAM4X')
        if a1_result and a1_result.get('history'):
            try:
                _export_full_model_artifacts(a1_result, final_model_dir)
            except Exception as e:
                print(f"  ⚠️ Could not export full-model artifacts: {e}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: BASELINE COMPARISON
    # ═══════════════════════════════════════════════════════════════
    baseline_results = {}

    if not skip_baselines and not analysis_only:
        # All baselines trained with the WINNING RECIPE (BCE + pos_weight + moderate aug)
        BASELINE_EXPERIMENTS = [
            ("B1_Plain_MobileNetV2", make_plain, bce, {"attention": "none"}),
            ("B2_SE_Net",            make_se,    bce, {"attention": "SE"}),
            ("B3_CBAM",              make_cbam,  bce, {"attention": "CBAM"}),
            ("B4_ECA_Net",           make_eca,   bce, {"attention": "ECA"}),
        ]

        for name, model_fn, crit_fn, extra in BASELINE_EXPERIMENTS:
            save_path = BASELINE_DIR / f"{name}.json"
            if save_path.exists():
                print(f"\n✓ {name} already completed — loading")
                with open(save_path) as f:
                    baseline_results[name] = json.load(f)
                continue
            print(f"\n{'#'*60}\n# BASELINE: {name}\n{'#'*60}")
            result = run_experiment(name, model_fn(), crit_fn(), BASELINE_DIR,
                                    strong_aug=False, extra_info=extra)
            baseline_results[name] = result
            checkpoint_volume.commit()
    else:
        for f in BASELINE_DIR.glob("*.json"):
            with open(f) as fh:
                baseline_results[f.name.replace('.json', '')] = json.load(fh)
        print(f"Loaded {len(baseline_results)} existing baseline results")

    # B5 (Ours) = the full GLAAM-4X (A1) trained with the winning recipe
    if 'A1_Full_GLAAM4X' in ablation_results:
        b5_result = dict(ablation_results['A1_Full_GLAAM4X'])
        b5_result['name'] = 'B5_GLAAM_4X_Ours'
        b5_result['extra_info'] = {'attention': 'GLAAM-4X (disease-specific)', 'source': 'retrained winning recipe'}
        baseline_results['B5_GLAAM_4X_Ours'] = b5_result
        with open(BASELINE_DIR / "B5_GLAAM_4X_Ours.json", 'w') as f:
            json.dump(b5_result, f, indent=2, default=str)
        print(f"   ✅ B5 set to A1 (full GLAAM-4X, winning recipe)")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: STATISTICAL SIGNIFICANCE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  STATISTICAL SIGNIFICANCE TESTING\n{'='*60}")

    all_results = {**ablation_results, **baseline_results}
    ref_name = "B5_GLAAM_4X_Ours" if "B5_GLAAM_4X_Ours" in all_results else "A1_Full_GLAAM4X"

    if ref_name not in all_results:
        print("⚠️  No reference model found — skipping significance tests")
    else:
        ref = all_results[ref_name]
        ref_logits = np.array(ref['test_logits'])
        ref_labels = np.array(ref['test_labels'])
        ref_probs = 1 / (1 + np.exp(-ref_logits))
        ref_thr = ref['best_thresholds']

        # DeLong's test
        def delong_test(y_true, y_pred1, y_pred2):
            y_true = np.asarray(y_true).astype(int)
            y_pred1 = np.asarray(y_pred1, dtype=float)
            y_pred2 = np.asarray(y_pred2, dtype=float)
            n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
            if n_pos == 0 or n_neg == 0:
                return 0.5, 0.5, 0.0, 1.0
            pos1 = y_pred1[y_true==1]; neg1 = y_pred1[y_true==0]
            pos2 = y_pred2[y_true==1]; neg2 = y_pred2[y_true==0]
            v10_1 = np.array([np.sum(neg1 < pp) + 0.5*np.sum(neg1 == pp) for pp in pos1]) / n_neg
            v10_2 = np.array([np.sum(neg2 < pp) + 0.5*np.sum(neg2 == pp) for pp in pos2]) / n_neg
            v01_1 = np.array([np.sum(pos1 > nn) + 0.5*np.sum(pos1 == nn) for nn in neg1]) / n_pos
            v01_2 = np.array([np.sum(pos2 > nn) + 0.5*np.sum(pos2 == nn) for nn in neg2]) / n_pos
            auc1 = v10_1.mean(); auc2 = v10_2.mean()
            s_10 = np.var(v10_1, ddof=1) if n_pos > 1 else 0
            s_01 = np.var(v01_1, ddof=1) if n_neg > 1 else 0
            var1 = s_10/n_pos + s_01/n_neg
            s_10_2 = np.var(v10_2, ddof=1) if n_pos > 1 else 0
            s_01_2 = np.var(v01_2, ddof=1) if n_neg > 1 else 0
            var2 = s_10_2/n_pos + s_01_2/n_neg
            cov_10 = np.cov(v10_1, v10_2, ddof=1)[0,1] if n_pos > 1 else 0
            cov_01 = np.cov(v01_1, v01_2, ddof=1)[0,1] if n_neg > 1 else 0
            cov = cov_10/n_pos + cov_01/n_neg
            diff = auc1 - auc2
            se = np.sqrt(max(var1 + var2 - 2*cov, 1e-12))
            z = diff / se if se > 0 else 0
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            return auc1, auc2, z, p

        def bootstrap_f1_ci(y_true, y_pred, n_boot=2000):
            n = len(y_true); f1s = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                if len(np.unique(y_true[idx])) < 2: continue
                f1s.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
            f1s = np.array(f1s)
            return f1_score(y_true, y_pred, zero_division=0), np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)

        def mcnemar_test(y_true, p1, p2):
            c1 = (p1 == y_true); c2 = (p2 == y_true)
            b = np.sum(c1 & ~c2); c = np.sum(~c1 & c2)
            if b + c == 0: return 0.0, 1.0
            if b + c < 25:
                p = stats.binomtest(min(b,c), b+c, 0.5).pvalue
            else:
                chi2 = (abs(b-c)-1)**2 / (b+c)
                p = 1 - stats.chi2.cdf(chi2, 1)
            return (abs(b-c)-1)**2/(b+c), p

        np.random.seed(42)
        sig_rows = []
        for name, res in all_results.items():
            comp_logits = np.array(res['test_logits'])
            comp_probs = 1 / (1 + np.exp(-comp_logits))
            comp_thr = res['best_thresholds']
            for i, d in enumerate(DISEASE_NAMES):
                y_true = ref_labels[:, i]
                if len(np.unique(y_true)) < 2: continue
                auc1, auc2, z, p_d = delong_test(y_true, ref_probs[:, i], comp_probs[:, i])
                ref_pred = (ref_probs[:, i] >= ref_thr.get(d, 0.5)).astype(int)
                comp_pred = (comp_probs[:, i] >= comp_thr.get(d, 0.5)).astype(int)
                f1, ci_lo, ci_hi = bootstrap_f1_ci(y_true, ref_pred)
                _, p_mcn = mcnemar_test(y_true, ref_pred, comp_pred)
                sig_rows.append({
                    'Comparison': f'{ref_name} vs {name}', 'Disease': d,
                    'Ref AUC': round(auc1, 4), 'Comp AUC': round(auc2, 4),
                    'Δ AUC': round(auc1 - auc2, 4),
                    'DeLong z': round(z, 3),
                    'DeLong p': f'{p_d:.4f}' if p_d >= 0.0001 else '<0.0001',
                    'Ref F1': round(f1, 4),
                    'F1 95% CI': f'[{ci_lo:.4f}, {ci_hi:.4f}]',
                    'McNemar p': f'{p_mcn:.4f}' if p_mcn >= 0.0001 else '<0.0001',
                })
        sig_table = pd.DataFrame(sig_rows)
        sig_table.to_csv(SIGNIFICANCE_DIR / "significance_tests.csv", index=False)
        print(f"✅ Significance tests saved ({len(sig_table)} comparisons)")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: FLOPs & EFFICIENCY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  FLOPs & EFFICIENCY\n{'='*60}")
    from thop import profile, clever_format

    efficiency_models = {
        "B1_Plain": make_plain, "B2_SE": make_se, "B3_CBAM": make_cbam,
        "B4_ECA": make_eca, "B5_GLAAM4X": make_full,
    }
    eff_rows = []
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for name, fn in efficiency_models.items():
        m = fn().to(device).eval()
        params = sum(p.numel() for p in m.parameters())
        try:
            flops, _ = profile(m, inputs=(dummy,), verbose=False)
            flops_str = clever_format([flops], "%.2f")
        except:
            flops = 0; flops_str = "N/A"
        timings = []
        with torch.no_grad():
            for _ in range(10): _ = m(dummy)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            for _ in range(50):
                t0 = time.time(); _ = m(dummy)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                timings.append((time.time() - t0) * 1000)
        eff_rows.append({'Model': name, 'Params (M)': round(params/1e6, 2),
                         'FLOPs': flops_str, 'Mean Latency (ms)': round(np.mean(timings), 2),
                         'P95 Latency (ms)': round(np.percentile(timings, 95), 2)})
        print(f"  {name:15s} | {params/1e6:.2f}M | {flops_str} | {np.mean(timings):.1f}ms")
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    eff_table = pd.DataFrame(eff_rows)
    eff_table.to_csv(RESULTS_DIR / "efficiency_comparison.csv", index=False)
    print("✅ Efficiency comparison saved")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: FAILURE CASE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  FAILURE CASE ANALYSIS\n{'='*60}")
    if ref_name in all_results:
        test_paths = test_df['image_path'].values
        test_sources = test_df['source'].values if 'source' in test_df.columns else np.array(['unknown'] * len(test_df))
        failure_rows = []
        for i, d in enumerate(DISEASE_NAMES):
            y_true = ref_labels[:, i].astype(int)
            y_prob = ref_probs[:, i]
            thr = ref_thr.get(d, 0.5)
            y_pred = (y_prob >= thr).astype(int)
            fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
            fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
            for idx in fn_idx:
                failure_rows.append({'Disease': d, 'Type': 'False Negative',
                    'Source': test_sources[idx] if idx < len(test_sources) else 'unknown',
                    'Probability': round(float(y_prob[idx]), 4), 'Threshold': round(thr, 2)})
            for idx in fp_idx:
                failure_rows.append({'Disease': d, 'Type': 'False Positive',
                    'Source': test_sources[idx] if idx < len(test_sources) else 'unknown',
                    'Probability': round(float(y_prob[idx]), 4), 'Threshold': round(thr, 2)})
        failure_df = pd.DataFrame(failure_rows)
        failure_df.to_csv(RESULTS_DIR / "failure_cases.csv", index=False)
        print(f"  Total failures: {len(failure_df)}")
        print(f"  False Negatives: {(failure_df['Type'] == 'False Negative').sum()}")
        print(f"  False Positives: {(failure_df['Type'] == 'False Positive').sum()}")
        if 'Source' in failure_df.columns:
            print(f"  Per-source failures:")
            print(failure_df.groupby('Source').size().sort_values(ascending=False).to_string())
        print("✅ Failure cases saved")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 8: CROSS-DATASET GENERALIZATION (per-source)
    # ═══════════════════════════════════════════════════════════════
    # The test CSV has a 'source' column (odir, RFMiD, DDR, JSIEC, etc.)
    # We evaluate per-source to see if the model generalizes across datasets
    print(f"\n{'='*60}\n  CROSS-DATASET GENERALIZATION (per-source)\n{'='*60}")
    if ref_name in all_results and 'source' in test_df.columns:
        test_sources = test_df['source'].values
        unique_sources = np.unique(test_sources)
        print(f"  Test set sources: {unique_sources}")
        for src in unique_sources:
            count = (test_sources == src).sum()
            print(f"    {src}: {count} images")

        cross_rows = []
        for src in unique_sources:
            mask = test_sources == src
            if mask.sum() < 10:
                continue
            src_logits = ref_logits[mask]
            src_labels = ref_labels[mask]
            src_metrics, _ = compute_metrics(src_logits, src_labels, ref_thr)
            row = {'Source': src, 'N': int(mask.sum()), 'Macro F1': round(src_metrics['macro_f1'], 4)}
            for d in DISEASE_NAMES:
                row[f'{d} AUC'] = round(src_metrics[d]['auc'], 4)
            cross_rows.append(row)

        cross_table = pd.DataFrame(cross_rows)
        cross_table.to_csv(RESULTS_DIR / "cross_dataset_generalization.csv", index=False)
        print(f"\n{cross_table.to_string(index=False)}")
        print("✅ Cross-dataset analysis saved")
    else:
        print("⚠️  No 'source' column in test data — skipping cross-dataset analysis")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 9: SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PUBLICATION SUMMARY TABLES\n{'='*60}")

    # Table 1: Baselines
    if baseline_results:
        bl_rows = []
        for name, res in baseline_results.items():
            m = res['test_metrics']
            row = {'Model': name, 'Params (M)': round(res['n_params']/1e6, 2),
                   'Test Macro F1': round(res['test_macro_f1'], 4)}
            for d in DISEASE_NAMES:
                row[f'{d} AUC'] = round(m.get(d, {}).get('auc', 0), 4)
            bl_rows.append(row)
        bl_table = pd.DataFrame(bl_rows)
        bl_table.to_csv(RESULTS_DIR / "paper_table1_baselines.csv", index=False)
        print("\nTABLE 1: BASELINE COMPARISON")
        print(bl_table.to_string(index=False))

    # Table 2: Ablation
    if ablation_results:
        ab_rows = []
        ref_f1 = ablation_results.get('A1_Full_GLAAM4X', {}).get('test_macro_f1', 0)
        for name, res in ablation_results.items():
            m = res['test_metrics']
            row = {'Variant': name, 'Params (M)': round(res['n_params']/1e6, 2),
                   'Test Macro F1': round(res['test_macro_f1'], 4),
                   'Δ F1': round(res['test_macro_f1'] - ref_f1, 4)}
            for d in DISEASE_NAMES:
                row[f'{d} AUC'] = round(m.get(d, {}).get('auc', 0), 4)
            ab_rows.append(row)
        ab_table = pd.DataFrame(ab_rows)
        ab_table.to_csv(RESULTS_DIR / "paper_table2_ablation.csv", index=False)
        print("\nTABLE 2: ABLATION STUDY")
        print(ab_table.to_string(index=False))

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

    # Ablation bar chart
    if ablation_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(ablation_results.keys())
        f1s = [ablation_results[n]['test_macro_f1'] for n in names]
        colors = ['#2196F3'] + ['#FF5722'] * (len(names) - 1)
        ax.bar(range(len(names)), f1s, color=colors, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
        ax.set_ylabel('Test Macro F1')
        ax.set_title('Ablation Study: Test Macro F1 per Variant', fontweight='bold')
        for i, v in enumerate(f1s):
            ax.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        fig.savefig(FIGURES_DIR / "ablation_macro_f1.png")
        plt.close()

    # Baseline bar chart
    if baseline_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(baseline_results.keys())
        f1s = [baseline_results[n]['test_macro_f1'] for n in names]
        colors = ['#9E9E9E', '#FF9800', '#4CAF50', '#2196F3', '#E91E63']
        ax.bar(range(len(names)), f1s, color=colors[:len(names)], width=0.55)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
        ax.set_ylabel('Test Macro F1')
        ax.set_title('Baseline Comparison: Test Macro F1', fontweight='bold')
        for i, v in enumerate(f1s):
            ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        fig.savefig(FIGURES_DIR / "baseline_macro_f1.png")
        plt.close()

    # Efficiency scatter
    if eff_rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        for row in eff_rows:
            flops_g = float(row['FLOPs'].replace('G','').replace('M','').replace(' ','')) if 'G' in str(row['FLOPs']) else 0
            ax.scatter(flops_g, row['Params (M)'], s=150)
            ax.annotate(row['Model'], (flops_g, row['Params (M)']), fontsize=8, xytext=(5,5), textcoords='offset points')
        ax.set_xlabel('FLOPs (G)'); ax.set_ylabel('Parameters (M)')
        ax.set_title('Efficiency: Parameters vs FLOPs', fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / "efficiency_scatter.png")
        plt.close()

    # ── Per-disease AUC heatmap (ablation) ─────────────────────────────────
    if ablation_results:
        ab_names = list(ablation_results.keys())
        auc_data = np.array([[ablation_results[n]['test_metrics'].get(d, {}).get('auc', 0)
                              for d in DISEASE_NAMES] for n in ab_names])
        fig, ax = plt.subplots(figsize=(8, max(4, len(ab_names) * 0.5)))
        im = ax.imshow(auc_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(DISEASE_NAMES))); ax.set_xticklabels(DISEASE_NAMES, fontsize=10)
        ax.set_yticks(range(len(ab_names))); ax.set_yticklabels([n.replace('_', '\n') for n in ab_names], fontsize=7)
        for i in range(len(ab_names)):
            for j in range(len(DISEASE_NAMES)):
                ax.text(j, i, f'{auc_data[i,j]:.3f}', ha='center', va='center', fontsize=7,
                        color='white' if auc_data[i,j] > 0.8 else 'black')
        ax.set_title('Ablation: Per-Disease AUC Heatmap', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='AUC')
        fig.savefig(FIGURES_DIR / "ablation_auc_heatmap.png")
        plt.close()

    # ── Per-disease AUC grouped bars (baselines) ────────────────────────────
    if baseline_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(DISEASE_NAMES)); width = 0.15
        bl_names = list(baseline_results.keys())
        colors_bl = ['#9E9E9E', '#FF9800', '#4CAF50', '#2196F3', '#E91E63']
        for i, name in enumerate(bl_names):
            aucs = [baseline_results[name]['test_metrics'].get(d, {}).get('auc', 0) for d in DISEASE_NAMES]
            ax.bar(x + i * width, aucs, width, label=name.split('_')[0], color=colors_bl[i % len(colors_bl)])
        ax.set_xticks(x + width * 2); ax.set_xticklabels(DISEASE_NAMES, fontsize=11)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Baseline Comparison: Per-Disease AUC', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right'); ax.grid(axis='y', alpha=0.2); ax.set_ylim(0.5, 1.01)
        fig.savefig(FIGURES_DIR / "baseline_per_disease_auc.png")
        plt.close()

    # ── Significance p-value heatmap ───────────────────────────────────────
    if 'sig_table' in dir() and len(sig_table) > 0:
        ref_name_sig = "B5_GLAAM_4X_Ours" if "B5_GLAAM_4X_Ours" in all_results else "A1_Full_GLAAM4X"
        comp_names = [n for n in all_results.keys() if n != ref_name_sig]
        if comp_names:
            p_matrix = np.ones((len(comp_names), len(DISEASE_NAMES)))
            for i, name in enumerate(comp_names):
                for j, d in enumerate(DISEASE_NAMES):
                    row = sig_table[(sig_table['Comparison'] == f'{ref_name_sig} vs {name}') & (sig_table['Disease'] == d)]
                    if len(row) > 0:
                        p_str = row.iloc[0]['DeLong p']
                        if p_str != '—':
                            p_matrix[i, j] = float(p_str.replace('<', ''))
            fig, ax = plt.subplots(figsize=(8, max(4, len(comp_names) * 0.5)))
            log_p = -np.log10(p_matrix + 1e-10)
            im = ax.imshow(log_p, cmap='YlOrRd', aspect='auto', vmin=0, vmax=5)
            ax.set_xticks(range(len(DISEASE_NAMES))); ax.set_xticklabels(DISEASE_NAMES, fontsize=10)
            ax.set_yticks(range(len(comp_names)))
            ax.set_yticklabels([n.replace('_', '\n') for n in comp_names], fontsize=7)
            for i in range(len(comp_names)):
                for j in range(len(DISEASE_NAMES)):
                    p = p_matrix[i, j]
                    text = f'{p:.3f}' if p >= 0.001 else '<.001'
                    ax.text(j, i, text, ha='center', va='center', fontsize=7,
                            color='white' if log_p[i, j] > 2.5 else 'black')
            ax.set_title(f"DeLong's Test p-values\n({ref_name_sig} vs each model, -log10 scale)",
                         fontsize=11, fontweight='bold')
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='-log10(p)')
            fig.savefig(FIGURES_DIR / "significance_pvalue_heatmap.png")
            plt.close()

    # ── F1 with 95% CI forest plot ─────────────────────────────────────────
    if 'sig_table' in dir() and len(sig_table) > 0:
        ref_f1_rows = sig_table[sig_table['Comparison'].str.contains('vs itself')]
        if len(ref_f1_rows) == 0:
            # Try to get ref F1 CI from the reference comparison rows
            ref_f1_rows = sig_table[sig_table['Comparison'] == sig_table['Comparison'].iloc[0]]
        if len(ref_f1_rows) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            diseases_plot = ref_f1_rows['Disease'].tolist()
            f1s_plot = ref_f1_rows['Ref F1'].tolist()
            cis_plot = ref_f1_rows['F1 95% CI'].tolist()
            ci_los = [float(c.strip('[]').split(',')[0]) for c in cis_plot]
            ci_his = [float(c.strip('[]').split(',')[1]) for c in cis_plot]
            disease_colors = ['#E74C3C', '#E67E22', '#8E44AD', '#2980B9']
            y_pos = range(len(diseases_plot))
            ax.barh(list(y_pos), f1s_plot, color=disease_colors[:len(diseases_plot)], height=0.5, alpha=0.7)
            for i, (f1, lo, hi) in enumerate(zip(f1s_plot, ci_los, ci_his)):
                ax.plot([lo, hi], [i, i], color='black', linewidth=2)
                ax.text(hi + 0.01, i, f'{f1:.3f} [{lo:.3f}, {hi:.3f}]', va='center', fontsize=8)
            ax.set_yticks(list(y_pos)); ax.set_yticklabels(diseases_plot, fontsize=11)
            ax.set_xlabel('F1 Score with 95% Bootstrap CI', fontsize=11)
            ax.set_title(f'{ref_name}: Per-Disease F1 with 95% CI', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1.15); ax.grid(axis='x', alpha=0.2)
            fig.savefig(FIGURES_DIR / "f1_confidence_intervals.png")
            plt.close()

    # ── Failure case analysis figure ───────────────────────────────────────
    if 'failure_df' in dir() and len(failure_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Left: FN vs FP per disease
        fail_summary = failure_df.groupby(['Disease', 'Type']).size().unstack(fill_value=0)
        if 'False Negative' not in fail_summary.columns: fail_summary['False Negative'] = 0
        if 'False Positive' not in fail_summary.columns: fail_summary['False Positive'] = 0
        fail_summary = fail_summary[['False Negative', 'False Positive']]
        fail_summary.plot(kind='bar', stacked=True, ax=axes[0],
                          color=['#E74C3C', '#F39C12'], edgecolor='none')
        axes[0].set_title('Failure Cases per Disease', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count'); axes[0].set_xlabel('Disease')
        axes[0].legend(['False Negative', 'False Positive']); axes[0].tick_params(axis='x', rotation=0)
        # Right: Probability distribution of failures
        fn_probs = failure_df[failure_df['Type'] == 'False Negative']['Probability']
        fp_probs = failure_df[failure_df['Type'] == 'False Positive']['Probability']
        if len(fn_probs) > 0: axes[1].hist(fn_probs, bins=20, alpha=0.6, color='#E74C3C', label=f'False Neg (n={len(fn_probs)})')
        if len(fp_probs) > 0: axes[1].hist(fp_probs, bins=20, alpha=0.6, color='#F39C12', label=f'False Pos (n={len(fp_probs)})')
        axes[1].axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold ~0.5')
        axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Count')
        axes[1].set_title('Failure Probability Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        fig.suptitle(f'Failure Case Analysis — {ref_name}', fontsize=13, fontweight='bold', y=1.02)
        fig.savefig(FIGURES_DIR / "failure_analysis.png")
        plt.close()

    # ── Cross-dataset generalization bar chart ─────────────────────────────
    if 'cross_table' in dir() and len(cross_table) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sources = cross_table['Source'].tolist()
        f1s_cd = cross_table['Macro F1'].tolist()
        colors_cd = plt.cm.Set2(np.linspace(0, 1, len(sources)))
        bars = ax.bar(range(len(sources)), f1s_cd, color=colors_cd, width=0.6)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([s[:12] for s in sources], fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('Macro F1', fontsize=12)
        ax.set_title('Cross-Dataset Generalization: Test Macro F1 per Source', fontsize=12, fontweight='bold')
        for bar, val, n in zip(bars, f1s_cd, cross_table['N']):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}\n(n={n})',
                    ha='center', va='bottom', fontsize=7)
        ax.grid(axis='y', alpha=0.2)
        fig.savefig(FIGURES_DIR / "cross_dataset_generalization.png")
        plt.close()

    # ── HEAD-TO-HEAD: GLAAM-4X vs ALL others (the key "we're better" figure) ─
    # Combines baselines + ablations into one figure with GLAAM-4X highlighted
    all_compare = {}
    # Add baselines (B1-B4 are competitors, B5 is ours)
    for name, res in baseline_results.items():
        all_compare[name] = res
    # Add ablations (A1 is ours = same as B5, A2-A8 are degraded variants)
    for name, res in ablation_results.items():
        if name != 'A1_Full_GLAAM4X':  # skip duplicate of B5
            all_compare[name] = res

    if all_compare and 'B5_GLAAM_4X_Ours' in all_compare:
        ref_f1_val = all_compare['B5_GLAAM_4X_Ours']['test_macro_f1']

        # Sort: our model first, then baselines, then ablations (by F1 descending)
        ordered_names = ['B5_GLAAM_4X_Ours']
        for n in ['B1_Plain_MobileNetV2', 'B2_SE_Net', 'B3_CBAM', 'B4_ECA_Net']:
            if n in all_compare: ordered_names.append(n)
        for n in sorted(ablation_results.keys()):
            if n != 'A1_Full_GLAAM4X' and n in all_compare:
                ordered_names.append(n)

        # ── Figure 1: Macro F1 head-to-head ────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 7))
        f1_vals = [all_compare[n]['test_macro_f1'] for n in ordered_names]
        # Colors: gold for our model, gray for baselines, light red for ablations
        bar_colors = []
        for n in ordered_names:
            if n == 'B5_GLAAM_4X_Ours':
                bar_colors.append('#FFD700')  # gold
            elif n.startswith('B'):
                bar_colors.append('#607D8B')  # blue-gray (baselines)
            else:
                bar_colors.append('#FF8A80')  # light red (ablations)
        bars = ax.bar(range(len(ordered_names)), f1_vals, color=bar_colors,
                      edgecolor='black' if 'B5' in ordered_names[0] else 'none',
                      linewidth=2, width=0.6)
        # Highlight our model bar with thick border
        bars[0].set_edgecolor('black'); bars[0].set_linewidth(2.5)
        ax.set_xticks(range(len(ordered_names)))
        labels_display = []
        for n in ordered_names:
            if n == 'B5_GLAAM_4X_Ours': labels_display.append('GLAAM-4X\n(OURS)')
            elif n.startswith('B'): labels_display.append(n.split('_')[1])  # SE, CBAM, etc.
            else: labels_display.append(n.replace('_', '\n').replace('A', ''))
        ax.set_xticklabels(labels_display, fontsize=8)
        ax.set_ylabel('Test Macro F1', fontsize=13, fontweight='bold')
        ax.set_title('GLAAM-4X vs All Competitors: Test Macro F1', fontsize=15, fontweight='bold')
        # Add value labels on bars
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Add horizontal line at our model's F1
        ax.axhline(ref_f1_val, color='#FFD700', linestyle='--', alpha=0.5, linewidth=1.5)
        # Add Δ annotations for each competitor
        for i, (n, val) in enumerate(zip(ordered_names, f1_vals)):
            if i == 0: continue
            delta = ref_f1_val - val
            if delta > 0:
                ax.text(i, val - 0.015, f'-{delta:.3f}', ha='center', va='top',
                        fontsize=7, color='#E53935', fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        ax.set_ylim(min(f1_vals) - 0.05, max(f1_vals) + 0.04)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFD700', edgecolor='black', linewidth=1.5, label='GLAAM-4X (Ours)'),
            Patch(facecolor='#607D8B', label='Baseline Attention Methods'),
            Patch(facecolor='#FF8A80', label='Ablation Variants (degraded)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        fig.savefig(FIGURES_DIR / "head_to_head_macro_f1.png")
        plt.close()

        # ── Figure 2: Per-disease AUC head-to-head (radar/spider chart) ──────
        # Shows GLAAM-4X vs each baseline across all 4 diseases
        baseline_only = {n: all_compare[n] for n in ordered_names
                         if n.startswith('B') and n in baseline_results}
        if len(baseline_only) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            angles = np.linspace(0, 2 * np.pi, len(DISEASE_NAMES), endpoint=False).tolist()
            angles += angles[:1]  # close the polygon
            colors_radar = ['#FFD700', '#607D8B', '#FF9800', '#4CAF50', '#2196F3']
            for i, (name, res) in enumerate(baseline_only.items()):
                aucs = [res['test_metrics'].get(d, {}).get('auc', 0.5) for d in DISEASE_NAMES]
                aucs += aucs[:1]
                label = 'GLAAM-4X (OURS)' if name == 'B5_GLAAM_4X_Ours' else name.split('_')[1]
                ax.plot(angles, aucs, 'o-', linewidth=2.5 if i == 0 else 1.5,
                        label=label, color=colors_radar[i % len(colors_radar)],
                        markersize=8 if i == 0 else 5)
                ax.fill(angles, aucs, alpha=0.15 if i == 0 else 0.05,
                        color=colors_radar[i % len(colors_radar)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(DISEASE_NAMES, fontsize=12, fontweight='bold')
            ax.set_ylim(0.5, 1.0)
            ax.set_title('Per-Disease AUC: GLAAM-4X vs Baselines', fontsize=14,
                         fontweight='bold', pad=20)
            ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.savefig(FIGURES_DIR / "head_to_head_radar_auc.png")
            plt.close()

        # ── Figure 3: Improvement summary (Δ AUC per disease per competitor) ─
        if len(baseline_only) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))
            competitors = [n for n in baseline_only.keys() if n != 'B5_GLAAM_4X_Ours']
            ref_aucs = {d: baseline_only['B5_GLAAM_4X_Ours']['test_metrics'].get(d, {}).get('auc', 0)
                        for d in DISEASE_NAMES}
            x = np.arange(len(DISEASE_NAMES))
            width = 0.8 / max(len(competitors), 1)
            for i, comp_name in enumerate(competitors):
                deltas = []
                for d in DISEASE_NAMES:
                    comp_auc = baseline_only[comp_name]['test_metrics'].get(d, {}).get('auc', 0)
                    deltas.append(ref_aucs[d] - comp_auc)  # positive = our model is better
                offset = (i - len(competitors) / 2 + 0.5) * width
                bars = ax.bar(x + offset, deltas, width, label=comp_name.split('_')[1],
                              edgecolor='none', alpha=0.8)
                # Add value labels
                for bar, val in zip(bars, deltas):
                    y_pos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.005
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                            f'{val:+.3f}', ha='center',
                            va='bottom' if val >= 0 else 'top',
                            fontsize=7, fontweight='bold',
                            color='#2E7D32' if val > 0 else '#C62828')
            ax.set_xticks(x)
            ax.set_xticklabels(DISEASE_NAMES, fontsize=12, fontweight='bold')
            ax.set_ylabel('Δ AUC (GLAAM-4X − Competitor)', fontsize=12)
            ax.set_title('GLAAM-4X Improvement Over Baselines (per disease)\nPositive = GLAAM-4X is better',
                         fontsize=13, fontweight='bold')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.legend(fontsize=9, title='Competitor')
            ax.grid(axis='y', alpha=0.2)
            fig.savefig(FIGURES_DIR / "head_to_head_improvement_delta.png")
            plt.close()

    # ── Combined publication dashboard ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Baseline Macro F1
    ax1 = fig.add_subplot(gs[0, 0])
    if baseline_results:
        names = [n.replace('_', '\n') for n in baseline_results.keys()]
        f1s = [baseline_results[n]['test_macro_f1'] for n in baseline_results.keys()]
        colors = ['#9E9E9E', '#FF9800', '#4CAF50', '#2196F3', '#E91E63']
        ax1.bar(range(len(names)), f1s, color=colors[:len(names)], width=0.55)
        ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=7)
        ax1.set_title('Baseline Comparison', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Macro F1'); ax1.grid(axis='y', alpha=0.2)

    # Panel 2: Ablation Macro F1
    ax2 = fig.add_subplot(gs[0, 1])
    if ablation_results:
        names = [n.replace('_', '\n') for n in ablation_results.keys()]
        f1s = [ablation_results[n]['test_macro_f1'] for n in ablation_results.keys()]
        colors = ['#2196F3'] + ['#FF5722'] * (len(names) - 1)
        ax2.bar(range(len(names)), f1s, color=colors, width=0.6)
        ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=5)
        ax2.set_title('Ablation Study', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Macro F1'); ax2.grid(axis='y', alpha=0.2)

    # Panel 3: Efficiency scatter
    ax3 = fig.add_subplot(gs[0, 2])
    if eff_rows:
        for row in eff_rows:
            flops_g = float(row['FLOPs'].replace('G','').replace('M','').replace(' ','')) if 'G' in str(row['FLOPs']) else 0
            ax3.scatter(flops_g, row['Params (M)'], s=100)
            ax3.annotate(row['Model'], (flops_g, row['Params (M)']), fontsize=6, xytext=(3,3), textcoords='offset points')
        ax3.set_xlabel('FLOPs (G)'); ax3.set_ylabel('Params (M)')
        ax3.set_title('Efficiency', fontsize=10, fontweight='bold'); ax3.grid(True, alpha=0.3)

    # Panel 4: Per-disease AUC (baselines)
    ax4 = fig.add_subplot(gs[1, 0])
    if baseline_results:
        x = np.arange(len(DISEASE_NAMES)); width = 0.15
        for i, name in enumerate(baseline_results.keys()):
            aucs = [baseline_results[name]['test_metrics'].get(d, {}).get('auc', 0) for d in DISEASE_NAMES]
            ax4.bar(x + i * width, aucs, width, label=name.split('_')[0])
        ax4.set_xticks(x + width * 2); ax4.set_xticklabels(DISEASE_NAMES, fontsize=8)
        ax4.set_title('Per-Disease AUC (Baselines)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('AUC'); ax4.set_ylim(0.5, 1.0); ax4.legend(fontsize=5); ax4.grid(axis='y', alpha=0.2)

    # Panel 5: Cross-dataset
    ax5 = fig.add_subplot(gs[1, 1])
    if 'cross_table' in dir() and len(cross_table) > 0:
        sources = cross_table['Source'].tolist()
        f1s_cd = cross_table['Macro F1'].tolist()
        ax5.bar(range(len(sources)), f1s_cd, color=plt.cm.Set2(np.linspace(0, 1, len(sources))), width=0.6)
        ax5.set_xticks(range(len(sources))); ax5.set_xticklabels([s[:8] for s in sources], fontsize=6, rotation=30)
        ax5.set_title('Cross-Dataset Generalization', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Macro F1'); ax5.grid(axis='y', alpha=0.2)

    # Panel 6: Failure cases
    ax6 = fig.add_subplot(gs[1, 2])
    if 'failure_df' in dir() and len(failure_df) > 0:
        fail_summary = failure_df.groupby(['Disease', 'Type']).size().unstack(fill_value=0)
        if 'False Negative' not in fail_summary.columns: fail_summary['False Negative'] = 0
        if 'False Positive' not in fail_summary.columns: fail_summary['False Positive'] = 0
        fail_summary = fail_summary[['False Negative', 'False Positive']]
        fail_summary.plot(kind='bar', stacked=True, ax=ax6, color=['#E74C3C', '#F39C12'], edgecolor='none', legend=False)
        ax6.set_title('Failure Cases', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Count'); ax6.tick_params(axis='x', rotation=0)
    else:
        ax6.text(0.5, 0.5, 'No failure data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Failure Cases', fontsize=10, fontweight='bold')

    fig.suptitle('GLAAM-4X Publication Analysis Dashboard', fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(FIGURES_DIR / "publication_dashboard.png")
    plt.close()

    print(f"\n📊 Figures saved to {FIGURES_DIR}/")

    # ═══════════════════════════════════════════════════════════════
    # COMMIT VOLUMES
    # ═══════════════════════════════════════════════════════════════
    checkpoint_volume.commit()

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PUBLICATION ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Results saved to: /checkpoints/publication_analysis/")
    print(f"\n  Files generated:")
    for f in sorted(RESULTS_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.relative_to(RESULTS_DIR)}  ({size_kb:.1f} KB)")
    print(f"\n  Summary:")
    print(f"    Ablation variants: {len(ablation_results)}")
    print(f"    Baseline models:   {len(baseline_results)}")
    print(f"    Significance tests: {len(sig_rows) if 'sig_rows' in dir() else 0}")
    print(f"    Efficiency models: {len(eff_rows)}")
    print(f"    Failure cases:     {len(failure_df) if 'failure_df' in dir() else 0}")
    print(f"\n{'='*70}")

    return {
        'ablation_count': len(ablation_results),
        'baseline_count': len(baseline_results),
        'results_dir': str(RESULTS_DIR),
    }


# ═══════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main(
    epochs: int = DEFAULT_EPOCHS,
    skip_ablation: bool = False,
    skip_baselines: bool = False,
    analysis_only: bool = False,
    gpu: str = "T4",
):
    """
    Run publication analysis on Modal.

    Usage:
        # Run everything (ablation + baselines + analysis)
        modal run modal_publication_analysis.py

        # Run only ablation
        modal run modal_publication_analysis.py --skip-baselines

        # Run only baselines
        modal run modal_publication_analysis.py --skip-ablation

        # Run only analysis (uses saved results)
        modal run modal_publication_analysis.py --analysis-only

        # Custom epochs
        modal run modal_publication_analysis.py --epochs 15

        # Faster GPU
        modal run modal_publication_analysis.py --gpu A10G
    """
    print(f"🚀 Starting publication analysis on Modal ({gpu} GPU)")
    print(f"   Epochs: {epochs}")
    print(f"   Skip ablation: {skip_ablation}")
    print(f"   Skip baselines: {skip_baselines}")
    print(f"   Analysis only: {analysis_only}")
    print(f"   Winning recipe: BCE + pos_weight + moderate augmentation")

    result = run_publication_analysis.remote(
        epochs=epochs,
        skip_ablation=skip_ablation,
        skip_baselines=skip_baselines,
        analysis_only=analysis_only,
    )

    print(f"\n{'='*70}")
    print(f"🎉 PUBLICATION ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"  Ablation variants: {result['ablation_count']}")
    print(f"  Baseline models:   {result['baseline_count']}")
    print(f"  Results: {result['results_dir']}")
    print(f"\n💾 All results saved to Modal volume: cataract-checkpoints")
    print(f"   Access via: modal volume ls cataract-checkpoints publication_analysis_v2/")
    print(f"{'='*70}")