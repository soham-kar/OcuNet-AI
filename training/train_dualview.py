"""
Training script for Dual-View Bayesian GLAAM model.
Week 5 of 10-week roadmap: Multi-view fusion + uncertainty quantification.

Usage:
    python training/train_dualview.py --data_root data/raw/slitlamp --epochs 100
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.glaam_bayesian import DualViewGLAAM_Bayesian
from models.hybrid_model import MultiTaskLoss
from utils.dataset_dualview import create_dualview_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with dual-view inputs."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images_45 = batch['image_45'].to(device)
        images_135 = batch['image_135'].to(device)
        labels = batch['severity_label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with both views
        logits = model(images_45, images_135)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images_45.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images_45.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total:.3f}"
        })
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def validate_with_uncertainty(model, val_loader, criterion, device, n_samples=10):
    """Validate with uncertainty estimation."""
    model.train()  # Keep dropout ON for MC sampling
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images_45 = batch['image_45'].to(device)
            images_135 = batch['image_135'].to(device)
            labels = batch['severity_label'].to(device)
            
            # Uncertainty prediction
            result = model.predict_with_uncertainty(images_45, images_135, n_samples=n_samples)
            
            # Use mean probabilities for prediction
            pred = result['prediction']
            mean_probs = result['mean_probs']
            
            # Compute loss on mean prediction
            loss = criterion(torch.log(mean_probs + 1e-8), labels)  # NLL loss
            
            total_loss += loss.item() * images_45.size(0)
            correct += (pred == labels).sum().item()
            total += images_45.size(0)
            
            all_uncertainties.extend(result['uncertainty_score'].cpu().numpy())
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'mean_uncertainty': np.mean(all_uncertainties),
        'high_uncertainty_rate': np.mean(np.array(all_uncertainties) > 0.1)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dual-View Bayesian GLAAM")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add', 'attention'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--mc_samples', type=int, default=10, help='MC Dropout samples for uncertainty')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataloaders
    print(f"Loading dual-view data from {args.data_root}...")
    train_loader, val_loader, test_loader = create_dualview_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=384
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Model
    model = DualViewGLAAM_Bayesian(
        num_classes=7,
        dropout_rate=args.dropout,
        fusion_method=args.fusion,
        pretrained=True
    )
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,} (~{params/1e6:.1f}M)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/dualview_{timestamp}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate with uncertainty
        val_metrics = validate_with_uncertainty(
            model, val_loader, criterion, device, n_samples=args.mc_samples
        )
        
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val   - Mean Uncertainty: {val_metrics['mean_uncertainty']:.4f}")
        print(f"  Val   - High Uncertainty Rate: {val_metrics['high_uncertainty_rate']:.2%}")
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Uncertainty/mean', val_metrics['mean_uncertainty'], epoch)
        
        # Save best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': {
                    'fusion': args.fusion,
                    'dropout': args.dropout
                }
            }, checkpoint_dir / 'best_dualview.pth')
            print(f"  Saved best model with val_acc: {best_val_acc:.4f}")
    
    writer.close()
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
