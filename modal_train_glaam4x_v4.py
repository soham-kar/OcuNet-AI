"""
Modal Training for GLAAM-4X on Unified Multi-Dataset Fundus Corpus (v4).

Run:
    modal run modal_train_glaam4x_v4.py

Features (v4 — Multi-Corpus + Synthetic Cataract):
  - 10+ datasets: ODIR, JSIEC, RFMiD, PALM, DDR, G1020, ORIGA, REFUGE + Synthetic
  - Asymmetric Loss (ASL) for multi-label imbalance (ICCV 2021)
  - Clean train/val_tune/test split (no data leakage)
  - Dual test evaluation (v3 comparison + extended publication set)
  - Balanced batch sampling with differential augmentation
  - Disease-specific attention heads (GLAAM-4X)
  - Per-disease threshold optimization on val_tune split
  - Auto-download results to local machine after training
  - LR warmup + cosine annealing
"""

import modal
import json
import os
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "numpy<2.0",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "scikit-learn",
    "pandas",
    "tqdm",
    "pillow",
    "matplotlib",
    "seaborn",
    "albumentations",
    "opencv-python-headless",
    "safetensors",  # Safe model format for publication
]).add_local_dir("./models", remote_path="/project/models").add_local_dir("./utils", remote_path="/project/utils").add_local_dir("./configs", remote_path="/project/configs")

app = modal.App("glaam4x-v4-training", image=image)

# Create Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']


# ═══════════════════════════════════════════════════════════════
@app.function(
    gpu="T4",  # T4 is cost-effective; model uses <5GB VRAM
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=86400,  # 24 hours
)
def train_glaam4x_unified(config: dict):
    """
    Train GLAAM-4X on unified multi-dataset fundus corpus.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    import torchvision.transforms as transforms
    from torchvision import models
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from tqdm import tqdm
    from PIL import Image
    import pandas as pd
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Validate config
    required_keys = ['model_name', 'num_classes', 'img_size', 'batch_size',
                     'dropout_rate', 'learning_rate', 'epochs', 'asl_gamma_neg', 'asl_gamma_pos']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    print("=" * 70)
    print(f"GLAAM-4X Unified Training: {config['model_name']}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TF32 only available on Ampere+ (A100, A10, RTX 30xx+). T4 is Turing, skip it.
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        torch.set_float32_matmul_precision('high')
        print("TF32 tensor cores enabled for float32 matmul")
    else:
        print("TF32 not available on this GPU (Turing or older)")
    
    # ========== COPY DATA TO LOCAL SSD ==========
    print("\nCopying dataset to local SSD (/tmp/data)...")
    import shutil
    import time
    t0 = time.time()
    shutil.copytree("/data", "/tmp/data", dirs_exist_ok=True)
    print(f"Dataset copied in {time.time() - t0:.1f}s")
    # Modal volume structure: /data/raw/ (images) and /data/data/ (CSVs)
    # After copytree: /tmp/data/raw/ and /tmp/data/data/
    # Synthetic images are at /tmp/data/synthetic/ (uploaded separately)
    csv_root = "/tmp/data/data" if Path("/tmp/data/data").exists() else "/tmp/data"
    img_root = "/tmp/data"  # Parent of both raw/ and synthetic/
    print(f"CSV root: {csv_root}")
    print(f"Image root: {img_root}")

    # ========== LOAD CSVs FROM LOCAL COPY ==========
    print("\nLoading v4 dataset CSV files...")
    # CSVs may be at /tmp/data/ root or /tmp/data/data/ subdirectory
    csv_root = "/tmp/data/data" if Path("/tmp/data/data").exists() else "/tmp/data"
    
    train_csv = Path("/tmp/data") / "train_v4.csv"
    val_tune_csv = Path("/tmp/data") / "val_tune_v4.csv"
    test_csv = Path("/tmp/data") / "test_v4.csv"
    test_extended_csv = Path("/tmp/data") / "test_v4_extended.csv"
    
    # Fallback: check csv_root if not at root
    if not train_csv.exists():
        train_csv = Path(csv_root) / "train_v4.csv"
        val_tune_csv = Path(csv_root) / "val_tune_v4.csv"
        test_csv = Path(csv_root) / "test_v4.csv"
        test_extended_csv = Path(csv_root) / "test_v4_extended.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_tune_csv.exists():
        raise FileNotFoundError(f"Val tune CSV not found: {val_tune_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    train_df = pd.read_csv(train_csv)
    val_tune_df = pd.read_csv(val_tune_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Train: {len(train_df)} images")
    print(f"Val tune: {len(val_tune_df)} images")
    print(f"Test (v3 comparison): {len(test_df)} images")
    print(f"Train disease distribution: {train_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"Val disease distribution: {val_tune_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"Test disease distribution: {test_df[DISEASE_NAMES].sum().to_dict()}")

    # ---------- Class weights for focal loss ----------
    pos_counts = train_df[DISEASE_NAMES].sum().values.astype(np.float32)
    class_weights = torch.tensor(len(train_df) / (pos_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 4  # normalize to mean=1
    class_weights = class_weights.to(device)
    print(f"Class weights: {dict(zip(DISEASE_NAMES, class_weights.cpu().tolist()))}")

    # ========== DATASET ==========
    def get_transforms(img_size, is_train, minority_aug_prob=0.7):
        base = [
            A.Resize(img_size, img_size),
            A.ToFloat(max_value=255),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        if not is_train:
            return A.Compose(base)

        strong = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(translate_percent={'x': 0.05, 'y': 0.05}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            # Heavy CPU transforms - reduced probability for speed
            A.ElasticTransform(alpha=1, sigma=50, p=0.1),
            A.GridDistortion(p=0.1),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
        ]
        return A.Compose(strong + base)

    class UnifiedDataset(Dataset):
        def __init__(self, df, img_dir, disease_cols, img_size, is_train=False, minority_aug_prob=0.7):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.disease_cols = disease_cols
            self.is_train = is_train
            self.minority_aug_prob = minority_aug_prob
            self.transform = get_transforms(img_size, is_train, minority_aug_prob)
            # Extra strong augmentation for minority-class samples
            self.strong_aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Affine(translate_percent={'x': 0.1, 'y': 0.1}, scale=(0.8, 1.2), rotate=(-30, 30), p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.4),
                A.GaussNoise(std_range=(0.08, 0.3), p=0.4),
                A.GaussianBlur(blur_limit=5, p=0.3),
                A.ElasticTransform(alpha=2, sigma=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.CoarseDropout(num_holes_range=(2, 6), hole_height_range=(16, 48), hole_width_range=(16, 48), p=0.3),
                A.Resize(img_size, img_size),
                A.ToFloat(max_value=255),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        def __len__(self):
            return len(self.df)

        def _load_image(self, path):
            if not os.path.isabs(path):
                # Try multiple base paths: img_root first, then img_root/raw
                candidates = [
                    os.path.join(self.img_dir, path),
                    os.path.join(self.img_dir, "raw", path),
                ]
                for p in candidates:
                    if os.path.exists(p):
                        path = p
                        break
                else:
                    path = candidates[0]  # Use first candidate, let cv2 raise error
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row['image_path']
            img = self._load_image(img_path)
            labels = row[self.disease_cols].values.astype(np.float32)
            # Differential augmentation: stronger aug for minority-class samples
            if self.is_train and labels.sum() > 0 and np.random.rand() < self.minority_aug_prob:
                augmented = self.strong_aug(image=img)
            else:
                augmented = self.transform(image=img)
            img = augmented['image']
            return img, torch.tensor(labels)

    train_dataset = UnifiedDataset(train_df, img_root, DISEASE_NAMES, config['img_size'], is_train=True, minority_aug_prob=0.7)
    val_tune_dataset = UnifiedDataset(val_tune_df, img_root, DISEASE_NAMES, config['img_size'], is_train=False)
    test_dataset = UnifiedDataset(test_df, img_root, DISEASE_NAMES, config['img_size'], is_train=False)

    # Balanced sampler
    def get_sample_weights(df, disease_cols):
        pos_counts = df[disease_cols].sum().values
        neg_counts = len(df) - pos_counts
        pos_weights = np.sqrt(1.0 / (pos_counts + 1e-6))
        neg_weights = np.sqrt(1.0 / (neg_counts + 1e-6))
        pos_weights = pos_weights / pos_weights.sum()
        neg_weights = neg_weights / neg_weights.sum()
        weights = np.zeros(len(df))
        for i, row in df.iterrows():
            w = 0.0
            for j, col in enumerate(disease_cols):
                w += pos_weights[j] if row[col] == 1 else neg_weights[j]
            weights[i] = w / len(disease_cols)
        return weights

    sample_weights = get_sample_weights(train_df, DISEASE_NAMES)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df) * 2,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )
    val_tune_loader = DataLoader(
        val_tune_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ========== MODEL ==========
    print("\nBuilding GLAAM-4X model...")
    import sys
    sys.path.insert(0, "/project")
    from models.glaam_4x import GLAAM_4X

    class GLAAM4XClassifier(nn.Module):
        REORDER_IDX = [2, 0, 1, 3]  # [DR, Glaucoma, Cataract, Myopia] -> [Cataract, DR, Glaucoma, Myopia]
        def __init__(self, num_classes=4, dropout_rate=0.3, pretrained=True):
            super().__init__()
            self.backbone = GLAAM_4X(pretrained=pretrained, dropout_rate=dropout_rate)
        def forward(self, x):
            out = self.backbone(x)
            logits = out['logits']
            return logits[:, self.REORDER_IDX]

    model = GLAAM4XClassifier(num_classes=4, dropout_rate=config['dropout_rate'], pretrained=True)
    model = model.to(device)
    
    # Sanity check: verify disease order alignment
    with torch.no_grad():
        dummy = torch.randn(2, 3, config['img_size'], config['img_size']).to(device)
        out_logits = model(dummy)
        print(f"Logits shape: {out_logits.shape} (should be [2, 4])")
        print("Expected disease order after reorder:", DISEASE_NAMES)
    
    # PyTorch 2.0+ optimization for ~20-40% speedup
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("torch.compile() not available (PyTorch < 2.0)")

    # ========== LOSS & OPTIMIZER ==========
    # ASL (Asymmetric Loss) — ICCV 2021, proven for multi-label imbalance
    from utils.losses import AsymmetricLossOptimized
    
    criterion = AsymmetricLossOptimized(
        gamma_neg=config.get('asl_gamma_neg', 4.0),
        gamma_pos=config.get('asl_gamma_pos', 0.0),
        clip=config.get('asl_clip', 0.05),
    )
    print(f"Using AsymmetricLoss (γ_neg={config.get('asl_gamma_neg', 4.0)}, "
          f"γ_pos={config.get('asl_gamma_pos', 0.0)}, clip={config.get('asl_clip', 0.05)})")
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    
    # Combined warmup + cosine annealing scheduler using LambdaLR
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config['epochs']
    def lr_lambda(epoch):
        # Epoch is 0-indexed in LambdaLR
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)  # linearly from 0 to 1
        else:
            # Cosine annealing from 1 to 0 over remaining epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    print(f"Using warmup ({warmup_epochs} epochs) + cosine annealing scheduler.")

    # ========== TRAINING LOOP ==========
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def train_epoch(model, loader, criterion, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Automatic Mixed Precision (AMP)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                
                # Safety check: detect NaN/Inf loss before backward
                if not torch.isfinite(loss):
                    print(f"\n🚨 WARNING: Non-finite loss ({loss.item():.2f}) at batch {batch_idx}")
                    print("  Skipping batch...")
                    scaler.update()
                    continue
                
                scaler.scale(loss).backward()
                # Gradient clipping to prevent explosion from ASL
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                
                # Safety check: detect NaN/Inf loss before backward
                if not torch.isfinite(loss):
                    print(f"\n🚨 WARNING: Non-finite loss ({loss.item():.2f}) at batch {batch_idx}")
                    print("  Skipping batch...")
                    continue
                
                loss.backward()
                # Gradient clipping to prevent explosion from ASL
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())
        scheduler.step()  # step after each epoch
        avg_loss = total_loss / len(loader)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return avg_loss, all_logits, all_labels

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
        return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

    def compute_metrics(logits, labels, thresholds=None):
        probs = 1 / (1 + np.exp(-logits))
        if thresholds is None:
            thresholds = {d: 0.5 for d in DISEASE_NAMES}
        metrics = {}
        for i, disease in enumerate(DISEASE_NAMES):
            y_true = labels[:, i]
            y_prob = probs[:, i]
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
            y_true = labels[:, i]
            y_prob = probs[:, i]
            best_f1, best_thr = 0, 0.5
            for thr in np.arange(0.05, 0.95, 0.01):
                y_pred = (y_prob >= thr).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            thresholds[disease] = float(best_thr)
        return thresholds

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    run_dir = f"/checkpoints/{config['model_name']}"
    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ---------- Resume support ----------
    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    latest_ckpt_path = os.path.join(run_dir, "latest_model.pth")
    if os.path.exists(latest_ckpt_path):
        print(f"\nFound checkpoint {latest_ckpt_path}, resuming...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        # Re-initialize scheduler at correct epoch
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch - 1)
        print(f"Resumed at epoch {start_epoch} | Best val F1 so far: {best_val_f1:.4f}")
    else:
        print("No checkpoint found, starting from scratch.")

    print("\nStarting training...")
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"\n{'='*60}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{total_epochs} | LR: {current_lr:.2e}")
        print(f"{'='*60}")

        train_loss, train_logits, train_labels = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        train_metrics, _ = compute_metrics(train_logits, train_labels)

        # Validation on tuning set
        val_logits, val_labels = evaluate(model, val_tune_loader)
        val_metrics, _ = compute_metrics(val_logits, val_labels)
        optimal_thresholds = find_optimal_thresholds(val_logits, val_labels)
        val_metrics_opt, _ = compute_metrics(val_logits, val_labels, optimal_thresholds)

        print(f"Train Loss: {train_loss:.4f} | Val Tune Macro F1: {val_metrics['macro_f1']:.4f} (thr=0.5)")
        print(f"Val Tune Macro F1 (optimal thr): {val_metrics_opt['macro_f1']:.4f}")
        for d in DISEASE_NAMES:
            m = val_metrics_opt[d]
            print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} thr={optimal_thresholds[d]:.2f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics_opt': val_metrics_opt,
            'optimal_thresholds': optimal_thresholds,
            'config': config,
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
        }

        if val_metrics_opt['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics_opt['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(checkpoint, f"{run_dir}/best_model.pth")
            # Also export safetensors (safe format for publication/sharing)
            from safetensors.torch import save_file
            save_file(model.state_dict(), f"{run_dir}/best_model.safetensors")
            print(f"New best model saved (tune macro F1 = {best_val_f1:.4f})")
        else:
            patience_counter += 1

        torch.save(checkpoint, f"{run_dir}/latest_model.pth")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

    # ========== FINAL EVALUATION ON TEST SETS ==========
    print("\nLoading best model for test evaluation...")
    best_checkpoint = torch.load(f"{run_dir}/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    final_thresholds = best_checkpoint['optimal_thresholds']

    # Evaluate on both test sets
    test_configs = [
        ("test_v4", test_csv, test_loader),
    ]

    # Add extended test set if available
    if test_extended_csv.exists():
        test_extended_df = pd.read_csv(test_extended_csv)
        test_extended_dataset = UnifiedDataset(test_extended_df, img_root, DISEASE_NAMES, config['img_size'], is_train=False)
        test_extended_loader = DataLoader(
            test_extended_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_configs.append(("test_v4_extended", test_extended_csv, test_extended_loader))
        print(f"Extended test set: {len(test_extended_df)} images")
    else:
        print(f"Extended test CSV not found: {test_extended_csv}, skipping")

    all_test_results = {}

    for test_name, test_csv_path, test_loader_obj in test_configs:
        print(f"\n{'─'*40}")
        print(f"  {test_name}")
        print(f"{'─'*40}")

        test_logits, test_labels = evaluate(model, test_loader_obj)
        test_metrics, test_probs = compute_metrics(test_logits, test_labels, final_thresholds)

        print(f"\n  {test_name} — {len(pd.read_csv(test_csv_path))} images")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        for d in DISEASE_NAMES:
            m = test_metrics[d]
            print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

        all_test_results[test_name] = {
            'num_images': len(pd.read_csv(test_csv_path)),
            'metrics': test_metrics,
        }

    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_tune_f1': best_val_f1,
        'optimal_thresholds': final_thresholds,
        'test_results': all_test_results,
        'config': config,
    }
    with open(f"{run_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTraining complete. Results saved to {run_dir}/")
    return results


@app.local_entrypoint()
def main():
    import subprocess
    from datetime import datetime

    config = {
        "model_name": "glaam4x_unified_v4_asl_384",
        "num_classes": 4,
        "img_size": 384,
        "batch_size": 32,
        "dropout_rate": 0.3,
        "learning_rate": 2e-5,
        "weight_decay": 5e-4,
        "epochs": 60,
        "warmup_epochs": 10,
        # ASL parameters (replaces focal_alpha/focal_gamma)
        "asl_gamma_neg": 4.0,   # Aggressively suppress easy negatives
        "asl_gamma_pos": 0.0,   # Don't suppress positives
        "asl_clip": 0.05,       # Hard-threshold negatives below 5% probability
    }

    model_name = config["model_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_results_dir = f"training_results/{model_name}_{timestamp}"

    print("=" * 60)
    print(f"GLAAM-4X v4 — Multi-Corpus Training Pipeline")
    print(f"Model: {model_name}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Step 1: Train on Modal cloud
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  STEP 1/2: Starting training on Modal T4 GPU...")
    print(f"{'='*60}")
    results = train_glaam4x_unified.remote(config)
    print(f"\n✅ Training complete!")

    # ═══════════════════════════════════════════════════════
    # Step 2: Download results to local machine
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  STEP 2/2: Downloading results to local machine...")
    print(f"{'='*60}")

    os.makedirs(local_results_dir, exist_ok=True)

    remote_checkpoint_dir = model_name
    print(f"\n  Downloading checkpoints from volume...")
    subprocess.run([
        "modal", "volume", "get",
        "cataract-checkpoints",
        remote_checkpoint_dir,
        local_results_dir,
    ], check=True)

    print(f"\n{'='*60}")
    print(f"  ✅ ALL DONE!")
    print(f"  Results saved to: {local_results_dir}/")
    print(f"     ├── best_model.pth          (full checkpoint, ~20MB)")
    print(f"     ├── best_model.safetensors  (safe format, ~14MB)")
    print(f"     ├── latest_model.pth")
    print(f"     ├── results.json")
    print(f"     └── config.json")
    print(f"{'='*60}")

    # Print summary
    results_path = os.path.join(local_results_dir, remote_checkpoint_dir, "results.json")
    if os.path.exists(results_path):
        import json as _json
        with open(results_path) as f:
            final_results = _json.load(f)
        print(f"\n📊 Final Results Summary:")
        for test_name, test_data in final_results.get("test_results", {}).items():
            print(f"  {test_name} ({test_data['num_images']} images):")
            m = test_data["metrics"]
            print(f"    Macro F1: {m['macro_f1']:.4f}")
            for d in DISEASE_NAMES:
                dm = m[d]
                print(f"    {d:12s} AUC={dm['auc']:.4f} F1={dm['f1']:.4f}")

    # Print summary
    results_path = os.path.join(local_results_dir, remote_checkpoint_dir, "results.json")
    if os.path.exists(results_path):
        import json as _json
        with open(results_path) as f:
            final_results = _json.load(f)
        print(f"\n📊 Final Results Summary:")
        for test_name, test_data in final_results.get("test_results", {}).items():
            print(f"  {test_name} ({test_data['num_images']} images):")
            m = test_data["metrics"]
            print(f"    Macro F1: {m['macro_f1']:.4f}")
            for d in DISEASE_NAMES:
                dm = m[d]
                print(f"    {d:12s} AUC={dm['auc']:.4f} F1={dm['f1']:.4f}")
