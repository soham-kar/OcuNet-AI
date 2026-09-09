# modal_evaluate_thresholds.py
"""
Modal Threshold Evaluation for GLAAM-4X (v5_myopia_fix)
========================================================

Compares threshold-optimization strategies for clinical screening:

  1. Youden's J statistic (Sensitivity + Specificity - 1)
     - The standard medical-AI method for finding the optimal cut-point
       on an ROC curve. Robust to class imbalance; prioritizes balanced
       sensitivity/specificity.
  2. F1-optimized thresholds (the previous approach)

Loads the trained best_model.pth, computes thresholds on the VALIDATION set,
then reports test-set metrics under each strategy.

IMPORTANT: The GLAAM_4X architecture is embedded inline (NOT imported from
models/glaam_4x.py) because the trained checkpoint uses the FIXED Myopia
attention head (reduction=32), whereas models/glaam_4x.py still has Myopia
as Identity. Importing the wrong architecture would fail to load the weights.

Usage:
    modal run modal_evaluate_thresholds.py
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

app = modal.App("glaam4x-thresholds", image=image)

# Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════
# Constants (must match training)
# ═══════════════════════════════════════════════════════════════
DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
IMG_SIZE = 384
BATCH_SIZE = 64
MODEL_NAME = "glaam4x_v5_myopia_fix"


@app.function(
    gpu="T4",
    cpu=8,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=3600,
    memory=16384,
)
def evaluate_thresholds():
    """
    Load best_model.pth, compute Youden's-J and F1-optimal thresholds on val,
    then evaluate test under both strategies.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import json
    import os
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from torchvision.transforms import v2
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score, roc_curve,
    )
    from tqdm import tqdm
    from PIL import Image
    import multiprocessing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ═══════════════════════════════════════════════════════════════
    # LOAD DATASET CSVS
    # ═══════════════════════════════════════════════════════════════
    DATA_ROOT = Path("/data")
    RAW_ROOT = DATA_ROOT / "raw"

    train_csv = DATA_ROOT / "train_v4.csv"
    val_csv = DATA_ROOT / "val_tune_v4.csv"
    test_csv = DATA_ROOT / "test_v4.csv"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

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

    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        before = len(df_)
        existing_mask = df_['image_path'].apply(os.path.exists)
        df_ = df_[existing_mask].reset_index(drop=True)
        if name == "train": train_df = df_
        elif name == "val": val_df = df_
        else: test_df = df_
        print(f"  {name}: {before} → {len(df_)} (removed {before - len(df_)})")

    print(f"\n  TOTAL: {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")
    print(f"  Test disease prevalence: {test_df[DISEASE_NAMES].mean().round(3).to_dict()}")

    # ═══════════════════════════════════════════════════════════════
    # TRANSFORMS + DATASET (eval only — no augmentation)
    # ═══════════════════════════════════════════════════════════════
    torch.set_num_threads(8)

    def get_eval_transform(img_size):
        return v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    class FundusDataset(Dataset):
        def __init__(self, df, disease_cols, img_size):
            self.df = df.reset_index(drop=True)
            self.disease_cols = disease_cols
            self.transform = get_eval_transform(img_size)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            labels = row[self.disease_cols].values.astype(np.float32)
            return self.transform(img), torch.tensor(labels)

    val_dataset = FundusDataset(val_df, DISEASE_NAMES, IMG_SIZE)
    test_dataset = FundusDataset(test_df, DISEASE_NAMES, IMG_SIZE)

    NUM_WORKERS = min(8, multiprocessing.cpu_count() - 1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # ═══════════════════════════════════════════════════════════════
    # GLAAM-4X MODEL (inline — matches trained architecture with Myopia fix)
    # ═══════════════════════════════════════════════════════════════
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
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.backbone = mobilenet.features
            self.attention_heads = nn.ModuleDict({
                'DR': MultiScaleGLAAM(1280, reduction=4),
                'Glaucoma': GLAAMBlock(1280, reduction=8),
                'Cataract': GLAAMBlock(1280, reduction=16),
                'Myopia': GLAAMBlock(1280, reduction=32),  # FIXED
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

    model = GLAAM_4X(pretrained=True, dropout_rate=0.3).to(device)

    # ═══════════════════════════════════════════════════════════════
    # LOAD BEST CHECKPOINT
    # ═══════════════════════════════════════════════════════════════
    RUN_DIR = Path("/checkpoints") / MODEL_NAME
    best_ckpt_path = RUN_DIR / "best_model.pth"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"{best_ckpt_path} not found on volume!")
    best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    print(f"  ✅ Loaded best_model.pth (epoch {best_ckpt.get('epoch', '?')})")

    # ═══════════════════════════════════════════════════════════════
    # EVALUATION FUNCTIONS
    # ═══════════════════════════════════════════════════════════════
    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_logits, all_labels = [], []
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            logits = model(images.to(device))['logits']
            all_logits.append(logits.cpu()); all_labels.append(labels)
        return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

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
                'specificity': recall_score(y_true, 1 - y_pred, zero_division=0),
            }
        metrics['macro_f1'] = np.mean([m['f1'] for m in metrics.values()])
        return metrics, probs

    def youden_thresholds(logits, labels):
        """Youden's J = Sensitivity + Specificity - 1, maximized over ROC curve."""
        probs = 1 / (1 + np.exp(-logits))
        thresholds = {}
        for i, disease in enumerate(DISEASE_NAMES):
            y_true, y_prob = labels[:, i], probs[:, i]
            fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            thresholds[disease] = float(roc_thr[best_idx])
        return thresholds

    def f1_thresholds(logits, labels):
        """F1-optimized thresholds (previous approach)."""
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

    # ═══════════════════════════════════════════════════════════════
    # RUN INFERENCE ON VAL + TEST
    # ═══════════════════════════════════════════════════════════════
    print("\n  Running inference on validation set...")
    val_logits, val_labels = evaluate(model, val_loader)
    print("  Running inference on test set...")
    test_logits, test_labels = evaluate(model, test_loader)

    # ═══════════════════════════════════════════════════════════════
    # COMPUTE THRESHOLDS ON VALIDATION
    # ═══════════════════════════════════════════════════════════════
    youden_thr = youden_thresholds(val_logits, val_labels)
    f1_thr = f1_thresholds(val_logits, val_labels)

    print("\n" + "=" * 70)
    print("  THRESHOLDS (computed on VALIDATION set)")
    print("=" * 70)
    print(f"  {'Disease':12s} | {'Youden J':>10s} | {'F1-opt':>10s}")
    print("-" * 70)
    for d in DISEASE_NAMES:
        print(f"  {d:12s} | {youden_thr[d]:10.4f} | {f1_thr[d]:10.4f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST METRICS UNDER EACH STRATEGY
    # ═══════════════════════════════════════════════════════════════
    def report(title, thresholds):
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
        metrics, _ = compute_metrics(test_logits, test_labels, thresholds)
        print(f"  {'Disease':12s} | {'AUC':>6s} | {'F1':>6s} | {'P':>6s} | {'R':>6s} | {'Spec':>6s}")
        print("-" * 70)
        for d in DISEASE_NAMES:
            m = metrics[d]
            print(f"  {d:12s} | {m['auc']:6.4f} | {m['f1']:6.4f} | {m['precision']:6.4f} | "
                  f"{m['recall']:6.4f} | {m['specificity']:6.4f}")
        print(f"  {'MACRO':12s} | {'':6s} | {metrics['macro_f1']:6.4f}")
        return metrics

    youden_test = report("TEST METRICS — YOUDEN'S J THRESHOLDS (Clinical Standard)", youden_thr)
    f1_test = report("TEST METRICS — F1-OPTIMIZED THRESHOLDS", f1_thr)

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    results = {
        'best_epoch': best_ckpt.get('epoch', None),
        'youden_thr': youden_thr,
        'f1_thr': f1_thr,
        'youden_test': youden_test,
        'f1_test': f1_test,
        'test_prevalence': test_df[DISEASE_NAMES].mean().round(3).to_dict(),
        'note': ('Youden J maximizes Sensitivity+Specificity-1 (clinical screening). '
                 'F1-opt maximizes F1. Both computed on validation, applied to test.'),
    }
    with open(RUN_DIR / "threshold_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✅ Saved results to {RUN_DIR / 'threshold_results.json'}")

    checkpoint_volume.commit()

    # Return a plain JSON string to avoid numpy pickling issues locally
    return json.dumps({
        'youden_thr': {d: float(v) for d, v in youden_thr.items()},
        'f1_thr': {d: float(v) for d, v in f1_thr.items()},
        'youden_test_macro_f1': float(youden_test['macro_f1']),
        'f1_test_macro_f1': float(f1_test['macro_f1']),
        'youden_test': {d: {k: float(v) for k, v in m.items()}
                        for d, m in youden_test.items() if isinstance(m, dict)},
        'f1_test': {d: {k: float(v) for k, v in m.items()}
                    for d, m in f1_test.items() if isinstance(m, dict)},
    })


# ═══════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main():
    print(f"🚀 Evaluating thresholds for GLAAM-4X ({MODEL_NAME}) on Modal")
    result = json.loads(evaluate_thresholds.remote())

    print(f"\n{'='*70}")
    print(f"🎉 THRESHOLD EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Youden's J thresholds: {result['youden_thr']}")
    print(f"  F1-opt thresholds:     {result['f1_thr']}")
    print(f"\n  Test Macro F1 (Youden J): {result['youden_test_macro_f1']:.4f}")
    print(f"  Test Macro F1 (F1-opt):   {result['f1_test_macro_f1']:.4f}")
    print(f"\n  Per-disease (Youden J):")
    for d, m in result['youden_test'].items():
        print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} Spec={m['specificity']:.4f}")
