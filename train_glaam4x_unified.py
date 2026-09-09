"""
Train GLAAM-4X on the unified multi-dataset fundus corpus.

Features:
  - Unified dataset (ODIR + JSIEC + RFMiD + PALM)
  - Balanced batch sampling (oversamples rare diseases)
  - Differential augmentation (strong for minorities)
  - Focal loss for hard example mining
  - Disease-specific attention heads (GLAAM-4X)
  - Per-disease threshold optimization on validation set

Usage:
    python train_glaam4x_unified.py --epochs 60 --batch_size 32 --lr 1e-4
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.balanced_sampler import BalancedODIRDataset, get_sample_weights


# ═══════════════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════

class MultiLabelFocalLoss(nn.Module):
    """
    Focal loss for multi-label classification.
    Down-weights easy negatives and focuses on hard examples.
    Supports per-class weighting to address severe imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights  # Tensor of shape (num_classes,)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits
            targets: (B, C) binary labels {0, 1}
        """
        probs = torch.sigmoid(logits)
        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Focal weighting
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Apply class weights if provided
        if self.class_weights is not None:
            w = self.class_weights.to(targets.device).view(1, -1)
            alpha_t = alpha_t * w
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL: GLAAM-4X WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class GLAAM4XClassifier(nn.Module):
    """
    GLAAM-4X with disease-specific attention heads.
    Wraps the existing glaam_4x.py model for multi-label classification.
    
    Note: The underlying model outputs diseases in order [DR, Glaucoma, Cataract, Myopia].
    We reorder to match our target order [Cataract, DR, Glaucoma, Myopia].
    """
    # Model order: DR=0, Glaucoma=1, Cataract=2, Myopia=3
    # Target order: Cataract=0, DR=1, Glaucoma=2, Myopia=3
    REORDER_IDX = [2, 0, 1, 3]  # maps model output index -> target index
    
    def __init__(self, num_classes=4, dropout_rate=0.3, pretrained=True):
        super().__init__()
        from models.glaam_4x import GLAAM_4X as _GLAAM_4X
        self.backbone = _GLAAM_4X(pretrained=pretrained, dropout_rate=dropout_rate)

    def forward(self, x):
        out = self.backbone(x)
        # Reorder logits from [DR, Glaucoma, Cataract, Myopia] to [Cataract, DR, Glaucoma, Myopia]
        logits = out['logits']
        reordered = logits[:, self.REORDER_IDX]
        return reordered


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_logits, all_labels


def compute_metrics(logits, labels, disease_names, thresholds=None):
    """Compute per-disease metrics."""
    probs = 1 / (1 + np.exp(-logits))
    if thresholds is None:
        thresholds = {d: 0.5 for d in disease_names}

    metrics = {}
    for i, disease in enumerate(disease_names):
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

    # Macro F1
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    metrics['macro_f1'] = macro_f1

    return metrics, probs


def find_optimal_thresholds(logits, labels, disease_names):
    """Find per-disease thresholds that maximize F1 on validation set."""
    probs = 1 / (1 + np.exp(-logits))
    thresholds = {}

    for i, disease in enumerate(disease_names):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        best_f1 = 0
        best_thr = 0.5
        for thr in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[disease] = float(best_thr)

    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train GLAAM-4X on unified fundus dataset')
    parser.add_argument('--train_csv', type=str, default='data/train_v4.csv', help='Path to train split CSV')
    parser.add_argument('--val_csv', type=str, default='data/val_tune_v4.csv', help='Path to validation split CSV')
    parser.add_argument('--test_csv', type=str, default='data/test_v4.csv', help='Path to test split CSV (v3 comparison)')
    parser.add_argument('--test_csv_extended', type=str, default='data/test_v4_extended.csv', help='Path to extended test CSV (publication)')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_glaam4x')
    parser.add_argument('--resume', type=str, default=None,help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device)
    print(f"Device: {device}")

    # Load pre-split datasets
    print(f"Loading train from {args.train_csv}...")
    print(f"Loading val from {args.val_csv}...")
    train_df = pd.read_csv(args.train_csv)
    val_tune_df = pd.read_csv(args.val_csv)
    disease_names = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
    print(f"Train images: {len(train_df)}")
    print(f"Val tune images: {len(val_tune_df)}")
    print(f"Train disease distribution: {train_df[disease_names].sum().to_dict()}")
    print(f"Val tune disease distribution: {val_tune_df[disease_names].sum().to_dict()}")

    # Create datasets and loaders
    train_dataset = BalancedODIRDataset(
        df=train_df,
        img_dir='.',
        disease_cols=disease_names,
        img_size=args.img_size,
        is_train=True,
        minority_aug_prob=0.7,
    )

    val_tune_dataset = BalancedODIRDataset(
        df=val_tune_df,
        img_dir='.',
        disease_cols=disease_names,
        img_size=args.img_size,
        is_train=False,
    )

    print(f"Val tune: {len(val_tune_df)} images")

    # Balanced sampler for training
    sample_weights = get_sample_weights(train_df, disease_names)
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df) * 2,
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
    )
    val_tune_loader = DataLoader(
        val_tune_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    print("Building GLAAM-4X model...")
    model = GLAAM4XClassifier(num_classes=4, dropout_rate=args.dropout, pretrained=True)
    model = model.to(device)

    # Sanity check: verify disease order alignment
    with torch.no_grad():
        dummy = torch.randn(2, 3, args.img_size, args.img_size).to(device)
        out_logits = model(dummy)
        print(f"Logits shape: {out_logits.shape} (should be [2, 4])")
        print("Expected disease order after reorder:", disease_names)

    # Class weights for focal loss (inverse frequency, normalized)
    pos_counts = train_df[disease_names].sum().values.astype(np.float32)
    class_weights = torch.tensor(len(train_df) / (pos_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 4  # normalize to mean=1
    class_weights = class_weights.to(device)
    print(f"Class weights: {dict(zip(disease_names, class_weights.cpu().tolist()))}")

    # Loss and optimizer
    criterion = MultiLabelFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        class_weights=class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine annealing scheduler
    warmup_epochs = 5
    total_epochs = args.epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.checkpoint_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Resume support
    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        # Re-initialize scheduler at correct epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch - 1)
        print(f"Resumed at epoch {start_epoch} | Best val F1 so far: {best_val_f1:.4f}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_logits, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        train_metrics, _ = compute_metrics(train_logits, train_labels, disease_names)

        # Validate on tuning set
        val_logits, val_labels = evaluate(model, val_tune_loader, device)
        val_metrics, val_probs = compute_metrics(val_logits, val_labels, disease_names)

        # Find optimal thresholds on tuning set
        optimal_thresholds = find_optimal_thresholds(val_logits, val_labels, disease_names)
        val_metrics_opt, _ = compute_metrics(val_logits, val_labels, disease_names, optimal_thresholds)

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Val Tune Macro F1: {val_metrics['macro_f1']:.4f} (thr=0.5)")
        print(f"Val Tune Macro F1 (optimal thr): {val_metrics_opt['macro_f1']:.4f}")
        print("Per-disease (optimal thresholds):")
        for d in disease_names:
            m = val_metrics_opt[d]
            print(f"  {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} thr={optimal_thresholds[d]:.2f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics_opt': val_metrics_opt,
            'optimal_thresholds': optimal_thresholds,
            'config': vars(args),
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
        }

        # Save best model
        if val_metrics_opt['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics_opt['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(checkpoint, os.path.join(run_dir, 'best_model.pth'))
            print(f"✅ New best model saved (tune macro F1 = {best_val_f1:.4f})")
        else:
            patience_counter += 1

        # Save latest
        torch.save(checkpoint, os.path.join(run_dir, 'latest_model.pth'))

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

        scheduler.step()

    # Final evaluation on test sets
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    # Load best model
    best_ckpt = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    final_thresholds = best_ckpt['optimal_thresholds']

    # Evaluate on each test set
    test_csvs = {
        'test_v4': args.test_csv,
        'test_v4_extended': args.test_csv_extended,
    }

    all_results = {}

    for test_name, test_csv in test_csvs.items():
        if not os.path.exists(test_csv):
            print(f"\n⚠️  {test_csv} not found, skipping {test_name}")
            continue

        print(f"\n{'─'*40}")
        print(f"  {test_name}: {test_csv}")
        print(f"{'─'*40}")

        test_df = pd.read_csv(test_csv)
        test_dataset = BalancedODIRDataset(
            df=test_df,
            img_dir='.',
            disease_cols=disease_names,
            img_size=args.img_size,
            is_train=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_logits, test_labels = evaluate(model, test_loader, device)
        test_metrics, _ = compute_metrics(test_logits, test_labels, disease_names, final_thresholds)

        print(f"\n  {test_name} — {len(test_df)} images")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Per-disease:")
        for d in disease_names:
            m = test_metrics[d]
            print(f"    {d:12s} AUC={m['auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} thr={final_thresholds[d]:.2f}")

        all_results[test_name] = {
            'num_images': len(test_df),
            'metrics': test_metrics,
            'thresholds': final_thresholds,
        }

    # Save final results
    results = {
        'best_epoch': best_epoch,
        'best_val_tune_f1': best_val_f1,
        'optimal_thresholds': final_thresholds,
        'test_results': all_results,
        'config': vars(args),
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {run_dir}/")
    print(f"Best model: {run_dir}/best_model.pth")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
