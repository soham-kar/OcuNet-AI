"""
Ablation Study Script.
Week 7 of 10-week roadmap: Prove each component adds value.

Experiments:
1. No YOLO (full image) - baseline
2. YOLO + single view
3. YOLO + dual view (no attention)
4. YOLO + GLAAM + single view
5. YOLO + GLAAM + dual view (full model)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hybrid_model import HybridCataractModel
from models.glaam_bayesian import GLAAM_Bayesian, DualViewGLAAM_Bayesian
from models.backbones.backbone import MobileNetV2WithAttention
from utils.dataset import create_dataloaders


class BaselineMobileNet(nn.Module):
    """MobileNetV2 without attention (baseline)."""
    
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        from torchvision import models
        
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def run_experiment(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-4
):
    """Run training experiment and return metrics."""
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        correct = 0
        total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['severity_label'].to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'attention'):
                # HybridCataractModel returns dict
                output = model(images)
                logits = output['severity_logits'] if isinstance(output, dict) else output
            else:
                logits = model(images)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
        
        train_acc = correct / total
        train_accs.append(train_acc)
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['severity_label'].to(device)
                
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'attention'):
                    output = model(images)
                    logits = output['severity_logits'] if isinstance(output, dict) else output
                else:
                    logits = model(images)
                
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += images.size(0)
        
        val_acc = correct / total
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={train_acc:.4f}, Val={val_acc:.4f}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    
    return {
        'best_val_acc': best_val_acc,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'parameters': params,
        'train_history': train_accs,
        'val_history': val_accs
    }


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default='outputs/ablation_results.csv')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, _ = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=384
    )
    
    # Define experiments
    experiments = {
        'MobileNetV2 (no attention)': lambda: BaselineMobileNet(num_classes=7, pretrained=True),
        
        'MobileNetV2 + GLAAM': lambda: HybridCataractModel(
            backbone='mobilenetv2',
            attention_type='glaam',
            pretrained=True,
            use_yolo=False
        ),
        
        'MobileNetV2 + GLAAI': lambda: HybridCataractModel(
            backbone='mobilenetv2',
            attention_type='glaai',
            pretrained=True,
            use_yolo=False
        ),
        
        'GLAAM + Bayesian (MC Dropout)': lambda: GLAAM_Bayesian(
            num_classes=7,
            dropout_rate=0.2,
            pretrained=True
        ),
    }
    
    # Run experiments
    results = []
    
    for name, model_fn in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        
        model = model_fn()
        metrics = run_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs
        )
        
        results.append({
            'Model': name,
            'Best Val Acc': f"{metrics['best_val_acc']:.4f}",
            'Final Val Acc': f"{metrics['final_val_acc']:.4f}",
            'Parameters': f"{metrics['parameters']:,}",
            'Params (M)': f"{metrics['parameters']/1e6:.2f}M"
        })
        
        print(f"\nResult: {metrics['best_val_acc']:.4f} accuracy")
        
        # Cleanup
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print('='*60)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
