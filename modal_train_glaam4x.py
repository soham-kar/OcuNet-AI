# modal_train_glaam4x.py
"""
Modal Training Script for GLAAM-4X
==================================

Trains GLAAM-4X from scratch on Modal GPU with:
  - Disease-specific attention heads (including Myopia — fixed!)
  - Asymmetric Loss (ASL)
  - Multi-source data from the pre-parsed "cataract-data" Modal volume
  - Per-epoch checkpointing with resume support
  - Publication-ready training graphs
  - Inference package export

Usage:
    # Train on T4 (default)
    modal run modal_train_glaam4x.py

    # Train on A10G (faster)
    modal run modal_train_glaam4x.py --gpu A10G

    # Custom epochs
    modal run modal_train_glaam4x.py --epochs 60

    # Resume from checkpoint (automatic if checkpoint exists)
    modal run modal_train_glaam4x.py

Requirements:
    - Modal volume "cataract-data" with pre-parsed CSVs:
        /data/train_v4.csv, /data/val_tune_v4.csv, /data/test_v4.csv
      and image folders under /data/raw/
    - Modal volume "cataract-checkpoints" (for saving model)
"""

import modal
import json
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Modal Image
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
])

app = modal.App("glaam4x-training", image=image)

# Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
IMG_SIZE = 384
BATCH_SIZE = 64      # Match baseline batch size for fair comparison
DEFAULT_EPOCHS = 40  # Match baseline epochs
WARMUP_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.3
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 0.0
ASL_CLIP = 0.05
PATIENCE = 10  # early stopping patience


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════
@app.function(
    gpu="T4",
    cpu=8,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=86400,
    memory=16384,
)
def train_glaam4x(epochs: int = DEFAULT_EPOCHS, gpu: str = "T4"):
    """
    Train GLAAM-4X from scratch on Modal GPU.
    Saves checkpoints, graphs, and inference package to Modal volume.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import json
    import os
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torchvision import models
    from torchvision.transforms import v2
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        roc_curve, precision_recall_curve, average_precision_score,
        confusion_matrix, ConfusionMatrixDisplay
    )
    from tqdm import tqdm
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import multiprocessing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1+2+3: LOAD PRE-PARSED DATASET CSVS FROM VOLUME
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 1: LOAD DATASET CSVS FROM cataract-data VOLUME")
    print("=" * 60)

    DATA_ROOT = Path("/data")
    RAW_ROOT = DATA_ROOT / "raw"

    train_csv = DATA_ROOT / "train_v4.csv"
    val_csv = DATA_ROOT / "val_tune_v4.csv"
    test_csv = DATA_ROOT / "test_v4.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found!\n"
                f"Expected pre-parsed CSVs at /data/train_v4.csv, /data/val_tune_v4.csv, /data/test_v4.csv\n"
                f"Upload the unified dataset CSVs to the 'cataract-data' Modal volume."
            )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Resolve image paths relative to /data/raw/
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        full = RAW_ROOT / p
        if full.exists():
            return str(full)
        full2 = DATA_ROOT / p
        if full2.exists():
            return str(full2)
        return str(full)

    for df_ in [train_df, val_df, test_df]:
        df_['image_path'] = df_['image_path'].apply(resolve_path)

    # Filter out missing images
    print("  Filtering for existing images...")
    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df_)
        existing_mask = df_['image_path'].apply(os.path.exists)
        df_ = df_[existing_mask].reset_index(drop=True)
        if name == "train": train_df = df_
        elif name == "val": val_df = df_
        else: test_df = df_
        print(f"  {name}: {before} → {len(df_)} (removed {before - len(df_)})")

    if len(train_df) < 100:
        raise FileNotFoundError(
            f"Only {len(train_df)} training images found after filtering. "
            f"Check that image paths in the CSVs match the volume structure."
        )

    print(f"\n  TOTAL: {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")
    print(f"  Disease counts (train): {train_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"  Disease counts (val):   {val_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"  Disease counts (test):  {test_df[DISEASE_NAMES].sum().to_dict()}")
    print(f"  Sources in train: {train_df['source'].value_counts().to_dict()}")
    print(f"  Sample path: {train_df['image_path'].iloc[0]}")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: AUGMENTATION + DATASET
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 4: AUGMENTATION + DATASET")
    print("=" * 60)

    torch.set_num_threads(8)

    class GaussianNoise(nn.Module):
        def __init__(self, std_range=(0.04, 0.2), p=0.3):
            super().__init__()
            self.std_range = std_range
            self.p = p
        def forward(self, img):
            if torch.rand(1).item() > self.p:
                return img
            std = torch.empty(1).uniform_(*self.std_range).item()
            return torch.clamp(img + torch.randn_like(img) * std, 0.0, 1.0)

    def get_transforms(img_size, is_train, minority_aug_prob=0.3): # FIX 3: Lowered from 0.7
        base_tail = [
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if not is_train:
            return v2.Compose([v2.Resize((img_size, img_size))] + base_tail)

        # FIX 3: Removed ElasticTransform entirely from both pipelines
        train_t = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply([v2.RandomChoice([
                v2.RandomRotation((90, 90)), v2.RandomRotation((180, 180)), v2.RandomRotation((270, 270)),
            ])], p=0.3),
            v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            v2.RandomApply([v2.ColorJitter(hue=0.03, saturation=0.2)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        ] + base_tail + [
            GaussianNoise(std_range=(0.04, 0.2), p=0.3),
            v2.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0)),
        ])

        strong_t = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.3),
            v2.RandomApply([v2.RandomChoice([
                v2.RandomRotation((90, 90)), v2.RandomRotation((180, 180)), v2.RandomRotation((270, 270)),
            ])], p=0.3),
            v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            v2.RandomApply([v2.ColorJitter(hue=0.05, saturation=0.3)], p=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.3),
        ] + base_tail + [
            GaussianNoise(std_range=(0.08, 0.3), p=0.4),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ])

        return train_t, strong_t

    class FundusDataset(Dataset):
        def __init__(self, df, disease_cols, img_size, is_train=False, minority_aug_prob=0.3):
            self.df = df.reset_index(drop=True)
            self.disease_cols = disease_cols
            self.img_size = img_size
            self.is_train = is_train
            self.minority_aug_prob = minority_aug_prob
            transforms = get_transforms(img_size, is_train, minority_aug_prob)
            if is_train:
                self.transform, self.strong_aug = transforms
            else:
                self.transform = transforms
                self.strong_aug = None

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            labels = row[self.disease_cols].values.astype(np.float32)
            if self.is_train and labels.sum() > 0 and np.random.rand() < self.minority_aug_prob:
                img = self.strong_aug(img)
            else:
                img = self.transform(img)
            return img, torch.tensor(labels)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: DATA LOADERS
    # ═══════════════════════════════════════════════════════════════
    NUM_WORKERS = min(8, multiprocessing.cpu_count() - 1)

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

    train_dataset = FundusDataset(train_df, DISEASE_NAMES, IMG_SIZE, is_train=True)
    val_dataset = FundusDataset(val_df, DISEASE_NAMES, IMG_SIZE, is_train=False)
    test_dataset = FundusDataset(test_df, DISEASE_NAMES, IMG_SIZE, is_train=False)

    sample_weights = get_sample_weights(train_df, DISEASE_NAMES)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_df) * 2, replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              prefetch_factor=4, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: GLAAM-4X MODEL (WITH MYOPIA ATTENTION FIX)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 6: GLAAM-4X MODEL (WITH MYOPIA FIX)")
    print("=" * 60)

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

    class GLAAM_4X(nn.Module):
        """
        GLAAM-4X: Four Disease-Specific Attention Specialists
        WITH MYOPIA ATTENTION HEAD (FIXED)
        """
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.backbone = mobilenet.features

            # 🔥 DISEASE-SPECIFIC ATTENTION HEADS (ALL 4 DISEASES)
            self.attention_heads = nn.ModuleDict({
                'DR': MultiScaleGLAAM(1280, reduction=4),
                'Glaucoma': GLAAMBlock(1280, reduction=8),
                'Cataract': GLAAMBlock(1280, reduction=16),
                'Myopia': GLAAMBlock(1280, reduction=32),  # ✅ FIXED: was Identity
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
            gate_weights = self.disease_gate(features)
            specialist_features = {}
            attention_maps = {}
            for disease in DISEASE_NAMES:
                if disease in self.attention_heads:
                    if return_attention:
                        attended, attn = self.attention_heads[disease](features, return_attention=True)
                        attention_maps[disease] = attn
                    else:
                        attended = self.attention_heads[disease](features)
                else:
                    attended = features
                pooled = F.adaptive_avg_pool2d(attended, 1).flatten(1)
                specialist_features[disease] = pooled
            logits = []
            for disease in DISEASE_NAMES:
                logits.append(self.classifiers[disease](specialist_features[disease]).squeeze(-1))
            output_logits = torch.stack(logits, dim=1)
            if return_attention:
                return {'logits': output_logits, 'features': specialist_features,
                        'attention_maps': attention_maps, 'gate_weights': gate_weights}
            return {'logits': output_logits, 'features': None, 'attention_maps': {}}

    model = GLAAM_4X(pretrained=True, dropout_rate=DROPOUT_RATE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ GLAAM-4X created | Params: {n_params:,}")
    print(f"  ✅ Myopia attention head: ENABLED (reduction=32)")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: LOSS + OPTIMIZER + SCHEDULER
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 7: LOSS + OPTIMIZER")
    print("=" * 60)

    class AsymmetricLoss(nn.Module):
        def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8, pos_weight=None):
            super().__init__()
            self.gamma_neg = gamma_neg
            self.gamma_pos = gamma_pos
            self.clip = clip
            self.eps = eps
            self.pos_weight = pos_weight  # FIX 4: Added pos_weight

        def forward(self, logits, targets):
            xs_pos = logits
            pt = torch.sigmoid(xs_pos).clamp(min=self.eps, max=1-self.eps)

            # FIX 4: Apply pos_weight to the positive loss
            pw = self.pos_weight if self.pos_weight is not None else 1.0
            pos_loss = targets * pw * torch.pow(1 - pt, self.gamma_pos) * F.logsigmoid(xs_pos)

            xs_neg = -logits
            p_neg = torch.sigmoid(xs_neg).clamp(min=self.eps, max=1-self.eps)
            neg_loss = (1 - targets) * torch.pow(1 - p_neg, self.gamma_neg) * F.logsigmoid(xs_neg)

            if self.clip > 0:
                probs = torch.sigmoid(logits)
                neg_loss = neg_loss * (~((targets == 0) & (probs < self.clip))).float()
            return (-pos_loss - neg_loss).mean()

    # FIX 4: Calculate pos_weights based on training data frequency
    pos_counts = train_df[DISEASE_NAMES].sum().values
    neg_counts = len(train_df) - pos_counts
    # Use sqrt to soften the weights, standard practice for ASL
    pos_weights = torch.tensor(np.sqrt(neg_counts / (pos_counts + 1e-6)), dtype=torch.float32).to(device)
    print(f"  Calculated Pos Weights: {pos_weights.cpu().numpy()}")

    criterion = AsymmetricLoss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP, pos_weight=pos_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"  ASL: γ_neg={ASL_GAMMA_NEG}, γ_pos={ASL_GAMMA_POS}, clip={ASL_CLIP}")
    print(f"  AdamW: lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")
    print(f"  Schedule: {WARMUP_EPOCHS} warmup + cosine over {epochs} epochs")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 8: TRAINING + EVALUATION FUNCTIONS
    # ═══════════════════════════════════════════════════════════════
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

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

    def train_epoch(model, loader, criterion, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Train", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(images)['logits']
                    loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    scaler.update(); continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(images)['logits']
                loss = criterion(logits, labels)
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
            logits = model(images.to(device))['logits']
            all_logits.append(logits.cpu()); all_labels.append(labels)
        return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 9: OUTPUT DIR + RESUME
    # ═══════════════════════════════════════════════════════════════
    MODEL_NAME = "glaam4x_v5_myopia_fix"
    RUN_DIR = Path("/checkpoints") / MODEL_NAME
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR = RUN_DIR / "training_graphs"
    GRAPHS_DIR.mkdir(exist_ok=True)

    # Save config
    with open(RUN_DIR / "config.json", 'w') as f:
        json.dump({
            "model_name": MODEL_NAME, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "epochs": epochs,
            "warmup_epochs": WARMUP_EPOCHS, "asl_gamma_neg": ASL_GAMMA_NEG,
            "asl_gamma_pos": ASL_GAMMA_POS, "asl_clip": ASL_CLIP,
            "myopia_attention": True, "myopia_reduction": 32,
        }, f, indent=2)

    # Resume support
    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch = 0
    best_opt_thr = None
    patience_counter = 0
    latest_ckpt = RUN_DIR / "latest_model.pth"

    history = {
        'epoch': [], 'lr': [], 'train_loss': [], 'train_macro_f1': [],
        'val_loss': [], 'val_macro_f1': [], 'val_macro_f1_opt': [],
        'val_auc': {d: [] for d in DISEASE_NAMES},
        'val_f1': {d: [] for d in DISEASE_NAMES},
        'val_precision': {d: [] for d in DISEASE_NAMES},
        'val_recall': {d: [] for d in DISEASE_NAMES},
    }

    if latest_ckpt.exists():
        print(f"\n  📂 Found checkpoint, resuming...")
        ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        best_opt_thr = ckpt.get('best_opt_thr', None)
        patience_counter = ckpt.get('patience_counter', 0)
        if 'history' in ckpt:
            history = ckpt['history']
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch - 1)
        print(f"  ✅ Resumed at epoch {start_epoch} | Best F1: {best_val_f1:.4f}")
    else:
        print(f"  No checkpoint found, starting from scratch")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 10: TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 10: TRAINING LOOP")
    print("=" * 60)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        train_loss, train_logits, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, scheduler)
        train_metrics, _ = compute_metrics(train_logits, train_labels)

        val_logits, val_labels = evaluate(model, val_loader)
        opt_thr = find_optimal_thresholds(val_logits, val_labels)
        val_metrics, _ = compute_metrics(val_logits, val_labels)
        val_metrics_opt, _ = compute_metrics(val_logits, val_labels, opt_thr)
        val_loss = criterion(torch.tensor(val_logits, dtype=torch.float32).to(device),
                             torch.tensor(val_labels, dtype=torch.float32).to(device)).item()

        print(f"Train Loss: {train_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val F1 (opt): {val_metrics_opt['macro_f1']:.4f}")
        for d in DISEASE_NAMES:
            m = val_metrics_opt[d]
            print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

        # Record history
        history['epoch'].append(epoch)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['train_loss'].append(train_loss)
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_macro_f1_opt'].append(val_metrics_opt['macro_f1'])
        for d in DISEASE_NAMES:
            history['val_auc'][d].append(val_metrics_opt[d]['auc'])
            history['val_f1'][d].append(val_metrics_opt[d]['f1'])
            history['val_precision'][d].append(val_metrics_opt[d]['precision'])
            history['val_recall'][d].append(val_metrics_opt[d]['recall'])

        # Save checkpoint
        ckpt = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_f1': best_val_f1, 'best_opt_thr': best_opt_thr,
            'patience_counter': patience_counter,
            'history': history,
        }

        if val_metrics_opt['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics_opt['macro_f1']
            best_epoch = epoch
            best_opt_thr = opt_thr
            patience_counter = 0
            torch.save(ckpt, str(RUN_DIR / "best_model.pth"))
            print(f"  ✅ New best model (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        torch.save(ckpt, str(latest_ckpt))
        checkpoint_volume.commit()

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping after {PATIENCE} epochs without improvement")
            break

    print(f"\n  Training complete. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Save history
    with open(RUN_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 11: TEST EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 11: TEST EVALUATION")
    print("=" * 60)

    best_ckpt = torch.load(str(RUN_DIR / "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    test_logits, test_labels = evaluate(model, test_loader)
    # Use the optimized thresholds found during validation (ASL shifts raw probs,
    # so 0.5 is a poor default threshold). Fall back to 0.5 if unavailable.
    test_metrics, test_probs = compute_metrics(test_logits, test_labels,
                                               thresholds=best_opt_thr)

    print(f"\n  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    for d in DISEASE_NAMES:
        m = test_metrics[d]
        print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f}")

    # Save results
    results = {
        'best_epoch': best_epoch, 'best_val_f1': best_val_f1,
        'best_opt_thr': best_opt_thr,
        'test_metrics': test_metrics,
    }
    with open(RUN_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 12: TRAINING GRAPHS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 12: TRAINING GRAPHS")
    print("=" * 60)

    plt.rcParams.update({'font.size': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    epochs_hist = history['epoch']

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_hist, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs_hist, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best ({best_epoch})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ASL Loss'); ax.set_title('Training & Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "01_loss_curve.png")); plt.close()

    # Macro F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_hist, history['val_macro_f1_opt'], 'green', linewidth=2, label='Val F1 (opt)')
    ax.plot(epochs_hist, history['train_macro_f1'], 'blue', linewidth=1, alpha=0.5, label='Train F1')
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro F1'); ax.set_title('Macro F1 Progression')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "02_macro_f1.png")); plt.close()

    # Per-disease AUC
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, d in enumerate(DISEASE_NAMES):
        ax.plot(epochs_hist, history['val_auc'][d], linewidth=2, color=colors[i], label=d)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC'); ax.set_title('Per-Disease Validation AUC')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "03_per_disease_auc.png")); plt.close()

    # Per-disease F1
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, d in enumerate(DISEASE_NAMES):
        ax.plot(epochs_hist, history['val_f1'][d], linewidth=2, color=colors[i], label=d)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1'); ax.set_title('Per-Disease Validation F1')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "04_per_disease_f1.png")); plt.close()

    # ROC curves (test set)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, d in enumerate(DISEASE_NAMES):
        y_true = test_labels[:, i]
        y_prob = test_probs[:, i]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, linewidth=2.5, color=colors[i], label=f'{d} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Chance')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Test Set ROC Curves'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "05_roc_curves.png")); plt.close()

    # PR curves (test set)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, d in enumerate(DISEASE_NAMES):
        y_true = test_labels[:, i]
        y_prob = test_probs[:, i]
        if len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, linewidth=2.5, color=colors[i], label=f'{d} (AP={ap:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Test Set Precision-Recall Curves'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(GRAPHS_DIR / "06_pr_curves.png")); plt.close()

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
    fig.savefig(str(GRAPHS_DIR / "07_confusion_matrices.png")); plt.close()

    print(f"  ✅ Graphs saved to {GRAPHS_DIR}/")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 13: EXPORT INFERENCE PACKAGE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 13: EXPORT INFERENCE PACKAGE")
    print("=" * 60)

    INFERENCE_DIR = Path("/checkpoints") / f"inference_package_{MODEL_NAME}"
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save clean weights
    torch.save(model.state_dict(), str(INFERENCE_DIR / "model_weights.pth"))

    # Save thresholds
    with open(INFERENCE_DIR / "thresholds.json", 'w') as f:
        json.dump({d: 0.5 for d in DISEASE_NAMES}, f, indent=2)

    # Save model info
    model_info = {
        "model_name": MODEL_NAME, "architecture": "GLAAM-4X",
        "backbone": "MobileNetV2", "num_classes": 4,
        "disease_names": DISEASE_NAMES, "img_size": IMG_SIZE,
        "dropout_rate": DROPOUT_RATE, "best_epoch": best_epoch,
        "best_val_f1": best_val_f1, "myopia_attention": True,
        "test_macro_f1": test_metrics['macro_f1'],
    }
    with open(INFERENCE_DIR / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"  ✅ Inference package saved to {INFERENCE_DIR}/")

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    for d in DISEASE_NAMES:
        m = test_metrics[d]
        print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f}")
    print(f"\n  Saved to: {RUN_DIR}/")
    print(f"  Graphs: {GRAPHS_DIR}/")
    print(f"  Inference: {INFERENCE_DIR}/")
    print("=" * 60)

    checkpoint_volume.commit()

    # Return a plain JSON string to avoid numpy pickling issues on the local client
    return json.dumps({
        'best_epoch': int(best_epoch),
        'best_val_f1': float(best_val_f1),
        'test_macro_f1': float(test_metrics['macro_f1']),
        'test_metrics': {d: {k: float(v) for k, v in m.items()} for d, m in test_metrics.items()},
    })


# ═══════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main(epochs: int = DEFAULT_EPOCHS, gpu: str = "T4"):
    """
    Train GLAAM-4X from scratch on Modal.

    Usage:
        modal run modal_train_glaam4x.py
        modal run modal_train_glaam4x.py --epochs 60
        modal run modal_train_glaam4x.py --gpu A10G
    """
    print(f"🚀 Training GLAAM-4X on Modal ({gpu} GPU)")
    print(f"   Epochs: {epochs}")
    print(f"   Myopia attention: FIXED (reduction=32)")

    result = json.loads(train_glaam4x.remote(epochs=epochs, gpu=gpu))

    print(f"\n{'='*60}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val F1: {result['best_val_f1']:.4f}")
    print(f"  Test Macro F1: {result['test_macro_f1']:.4f}")
    for d in DISEASE_NAMES:
        m = result['test_metrics'][d]
        print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f}")
    print(f"\n💾 Saved to Modal volume: cataract-checkpoints")
    print(f"{'='*60}")