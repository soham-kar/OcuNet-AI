"""
End-to-end verification of the unified training pipeline.
Runs a mini training session on a small subset to catch any issues.

Usage:
    python verify_training_pipeline.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.balanced_sampler import BalancedODIRDataset, get_sample_weights
from train_glaam4x_unified import GLAAM4XClassifier, MultiLabelFocalLoss

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']


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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / len(loader)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return avg_loss, all_logits, all_labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def main():
    print("=" * 70)
    print("END-TO-END TRAINING PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv("data/train_combined.csv")
    val_df = pd.read_csv("data/val_combined.csv")

    # Use small subset for quick verification
    train_df = train_df.head(64)
    val_df = val_df.head(16)

    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")

    # 2. Create datasets (use data/raw/ prefix for local testing with relative paths)
    print("\n[2/6] Creating datasets...")
    train_ds = BalancedODIRDataset(train_df, "data/raw", DISEASE_NAMES, 224, is_train=True, minority_aug_prob=0.7)
    val_ds = BalancedODIRDataset(val_df, "data/raw", DISEASE_NAMES, 224, is_train=False)
    print("  Datasets created successfully")

    # 3. Create loaders
    print("\n[3/6] Creating data loaders...")
    weights = get_sample_weights(train_df, DISEASE_NAMES)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df) * 2, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 4. Create model
    print("\n[4/6] Creating model...")
    device = torch.device("cpu")
    model = GLAAM4XClassifier(num_classes=4, dropout_rate=0.3, pretrained=False)
    model = model.to(device)
    print(f"  Model created on {device}")

    # 5. Train for 2 epochs
    print("\n[5/6] Training for 2 epochs...")
    criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(1, 3):
        print(f"\n  Epoch {epoch}/2")
        train_loss, train_logits, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics, _ = compute_metrics(train_logits, train_labels)

        val_logits, val_labels = evaluate(model, val_loader, device)
        val_metrics, _ = compute_metrics(val_logits, val_labels)
        val_thresholds = find_optimal_thresholds(val_logits, val_labels)
        val_metrics_opt, _ = compute_metrics(val_logits, val_labels, val_thresholds)

        print(f"    Train Loss: {train_loss:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"    Val Macro F1 (optimal): {val_metrics_opt['macro_f1']:.4f}")
        for d in DISEASE_NAMES:
            m = val_metrics_opt[d]
            print(f"      {d:12s} F1={m['f1']:.4f} thr={val_thresholds[d]:.2f}")

    # 6. Save and load checkpoint
    print("\n[6/6] Testing checkpoint save/load...")
    checkpoint = {
        'epoch': 2,
        'model_state_dict': model.state_dict(),
        'optimal_thresholds': val_thresholds,
    }
    torch.save(checkpoint, "/tmp/test_checkpoint.pth")
    loaded = torch.load("/tmp/test_checkpoint.pth", map_location=device)
    model.load_state_dict(loaded['model_state_dict'])
    print("  Checkpoint save/load successful")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED! Pipeline is ready for full training.")
    print("=" * 70)


if __name__ == '__main__':
    main()
