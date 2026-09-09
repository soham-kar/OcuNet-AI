# modal_eval_glaam4x.py
"""
Modal Evaluation Script for GLAAM-4X (v5_myopia_fix)
=====================================================

Loads the already-trained best_model.pth from the Modal volume and
re-evaluates on the test set using the OPTIMAL thresholds found on the
validation set. Because GLAAM-4X uses Asymmetric Loss, raw probabilities
are shifted, so the default 0.5 threshold under-reports F1. This script
fixes that and produces publication-ready Test F1 scores.

Usage:
    modal run modal_eval_glaam4x.py
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

app = modal.App("glaam4x-eval", image=image)

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
def eval_glaam4x():
    """
    Load best_model.pth, recompute optimal thresholds on val, evaluate on test.
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
        roc_auc_score, f1_score, precision_score, recall_score,
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
    # GLAAM-4X MODEL (must match training architecture)
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
                'Myopia': GLAAMBlock(1280, reduction=32),
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

    # ═══════════════════════════════════════════════════════════════
    # VALIDATION: find optimal thresholds
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  VALIDATION: FINDING OPTIMAL THRESHOLDS")
    print("=" * 60)
    val_logits, val_labels = evaluate(model, val_loader)
    opt_thr = find_optimal_thresholds(val_logits, val_labels)
    val_metrics_opt, _ = compute_metrics(val_logits, val_labels, opt_thr)
    print(f"  Optimal thresholds: {opt_thr}")
    print(f"  Val Macro F1 (opt): {val_metrics_opt['macro_f1']:.4f}")
    for d in DISEASE_NAMES:
        m = val_metrics_opt[d]
        print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST: evaluate with optimal thresholds
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST EVALUATION (OPTIMAL THRESHOLDS)")
    print("=" * 60)
    test_logits, test_labels = evaluate(model, test_loader)
    test_metrics, test_probs = compute_metrics(test_logits, test_labels, opt_thr)

    print(f"\n  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    for d in DISEASE_NAMES:
        m = test_metrics[d]
        print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    results = {
        'best_epoch': best_ckpt.get('epoch', None),
        'best_val_f1': best_ckpt.get('best_val_f1', None),
        'best_opt_thr': opt_thr,
        'test_metrics': test_metrics,
        'test_macro_f1': float(test_metrics['macro_f1']),
        'note': 'Test F1 computed with optimal thresholds from validation set (ASL shifts raw probs).',
    }
    with open(RUN_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✅ Saved results to {RUN_DIR / 'results.json'}")

    # Save test probabilities for downstream analysis (ROC/PR curves)
    np.savez(RUN_DIR / "test_probs.npz",
             logits=test_logits, labels=test_labels, probs=test_probs)
    print(f"  ✅ Saved test probs to {RUN_DIR / 'test_probs.npz'}")

    checkpoint_volume.commit()

    # Return a plain JSON string to avoid numpy pickling issues locally
    return json.dumps({
        'best_epoch': int(best_ckpt.get('epoch', 0)),
        'best_val_f1': float(best_ckpt.get('best_val_f1', 0.0)),
        'test_macro_f1': float(test_metrics['macro_f1']),
        'test_metrics': {d: {k: float(v) for k, v in m.items()}
                         for d, m in test_metrics.items() if isinstance(m, dict)},
    })


# ═══════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
@app.local_entrypoint()
def main():
    print(f"🚀 Evaluating GLAAM-4X ({MODEL_NAME}) on Modal")
    result = json.loads(eval_glaam4x.remote())

    print(f"\n{'='*60}")
    print(f"🎉 EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val F1: {result['best_val_f1']:.4f}")
    print(f"  Test Macro F1 (opt thr): {result['test_macro_f1']:.4f}")
    print(f"\n  Per-disease test metrics:")
    for d, m in result['test_metrics'].items():
        print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")
