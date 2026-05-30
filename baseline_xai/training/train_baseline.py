"""
Training script for MobileNetV2 Baseline Model.

Simple training without YOLO or dual-view complexity.
Focus on getting baseline metrics for comparison with GLAAM.

Usage:
    python baseline_xai/training/train_baseline.py --data_root data/raw/slitlamp --epochs 50
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.mobilenet_baseline import MobileNetBaseline, SimpleLoss


def get_dataloaders(data_root, batch_size=32, image_size=224, num_workers=4):
    """Create train/val/test dataloaders."""
    # Import from main utils
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.dataset import CataractDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = CataractDataset(data_root, split="train", transform=train_transform)
    val_dataset = CataractDataset(data_root, split="val", transform=val_transform)
    test_dataset = CataractDataset(data_root, split="test", transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(train_loader, num_classes=7, device='cuda'):
    """Compute class weights for imbalanced data."""
    try:
        from sklearn.utils.class_weight import compute_class_weight
        
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['severity_label'].numpy().tolist())
        
        all_labels = np.array(all_labels)
        unique_classes = np.unique(all_labels)
        
        weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
        
        weight_tensor = torch.ones(num_classes, dtype=torch.float32)
        for i, cls in enumerate(unique_classes):
            weight_tensor[int(cls)] = weights[i]
        
        return weight_tensor.to(device)
        
    except Exception as e:
        print(f"Warning: Could not compute class weights: {e}")
        return None


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    binary_correct = 0
    severity_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['image'].to(device)
        binary_labels = batch['binary_label'].to(device)
        severity_labels = batch['severity_label'].to(device)
        
        optimizer.zero_grad()
        
        output = model(images)
        
        losses = criterion(
            output['binary_logits'],
            output['severity_logits'],
            binary_labels,
            severity_labels
        )
        
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += losses['loss'].item() * images.size(0)
        binary_pred = output['binary_logits'].argmax(dim=1)
        severity_pred = output['severity_logits'].argmax(dim=1)
        binary_correct += (binary_pred == binary_labels).sum().item()
        severity_correct += (severity_pred == severity_labels).sum().item()
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'bin_acc': f"{binary_correct/total_samples:.3f}",
            'sev_acc': f"{severity_correct/total_samples:.3f}"
        })
    
    return {
        'loss': total_loss / total_samples,
        'binary_acc': binary_correct / total_samples,
        'severity_acc': severity_correct / total_samples
    }


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    binary_correct = 0
    severity_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            binary_labels = batch['binary_label'].to(device)
            severity_labels = batch['severity_label'].to(device)
            
            output = model(images)
            
            losses = criterion(
                output['binary_logits'],
                output['severity_logits'],
                binary_labels,
                severity_labels
            )
            
            total_loss += losses['loss'].item() * images.size(0)
            binary_pred = output['binary_logits'].argmax(dim=1)
            severity_pred = output['severity_logits'].argmax(dim=1)
            binary_correct += (binary_pred == binary_labels).sum().item()
            severity_correct += (severity_pred == severity_labels).sum().item()
            total_samples += images.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'binary_acc': binary_correct / total_samples,
        'severity_acc': severity_correct / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Train MobileNet Baseline")
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='baseline_xai/checkpoints')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data
    print(f"Loading data from {args.data_root}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, 
        batch_size=args.batch_size,
        image_size=224
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Class weights
    print("Computing class weights...")
    severity_weights = compute_class_weights(train_loader, device=device)
    
    # Model
    print("Creating MobileNetBaseline...")
    model = MobileNetBaseline(pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = SimpleLoss(severity_class_weights=severity_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"baseline_xai/logs/{timestamp}")
    
    # Training
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Binary: {train_metrics['binary_acc']:.4f}, Severity: {train_metrics['severity_acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Binary: {val_metrics['binary_acc']:.4f}, Severity: {val_metrics['severity_acc']:.4f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train_binary', train_metrics['binary_acc'], epoch)
        writer.add_scalar('Accuracy/val_binary', val_metrics['binary_acc'], epoch)
        writer.add_scalar('Accuracy/train_severity', train_metrics['severity_acc'], epoch)
        writer.add_scalar('Accuracy/val_severity', val_metrics['severity_acc'], epoch)
        
        # Save best
        val_acc = (val_metrics['binary_acc'] + val_metrics['severity_acc']) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'metrics': val_metrics
            }, checkpoint_dir / 'best_baseline.pth')
            print(f"  ✓ Saved best model (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    writer.close()
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test - Binary: {test_metrics['binary_acc']:.4f}, Severity: {test_metrics['severity_acc']:.4f}")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'test_binary_acc': test_metrics['binary_acc'],
        'test_severity_acc': test_metrics['severity_acc'],
        'total_params': total_params
    }
    
    torch.save(results, checkpoint_dir / 'baseline_results.pth')
    print(f"\n✅ Baseline training complete!")
    print(f"Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
