"""
Modal Training for MobileNet Baseline on ODIR-5K Fundus Dataset.

This uses REAL labels from the ODIR-5K dataset with proper train/val/test splits.

Usage:
    # 1. Upload ODIR data
    modal volume put cataract-data ./data/raw/odir /odir
    
    # 2. Train
    modal run baseline_xai/modal_odir.py::train_odir --epochs=50
    
    # 3. Download results
    modal volume get cataract-checkpoints /odir_baseline ./checkpoints_modal/odir
"""

import modal
import os
import sys
from pathlib import Path

# Volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "numpy<2.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "opencv-python-headless",
        "scikit-learn",
        "pandas",
        "tqdm",
        "pyyaml",
        "tensorboard",
        "pillow"
    ])
)

app = modal.App("cataract-odir-baseline", image=image)


@app.function(
    gpu="T4",  # Cheaper for baseline
    timeout=3600,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def train_odir(epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """
    Train MobileNet baseline on ODIR-5K with REAL labels.
    
    Expected metrics:
    - Loss should decrease (NOT stay at 0.0000)
    - Accuracy should gradually improve (start ~50%, end ~85%)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    print("=" * 60)
    print("MobileNet Baseline on ODIR-5K (REAL LABELS)")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========== LOAD ODIR DATA ==========
    print("\n[1/4] Loading ODIR-5K data with REAL labels...")
    
    csv_path = "/data/odir/full_df.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        print("Upload with: modal volume put cataract-data ./data/raw/odir /odir")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Total records in CSV: {len(df)}")
    
    # Find image directory
    img_dirs = [
        "/data/odir/ODIR-5K/ODIR-5K/Training Images",
        "/data/odir/ODIR-5K/Training Images",
        "/data/odir/preprocessed_images"
    ]
    
    img_dir = None
    for d in img_dirs:
        if os.path.exists(d):
            img_dir = d
            print(f"Found images at: {img_dir}")
            break
    
    if img_dir is None:
        print("ERROR: Could not find image directory!")
        print("Available directories:")
        for root, dirs, files in os.walk("/data/odir"):
            print(f"  {root}: {len(files)} files")
            if len(files) > 0:
                print(f"    Sample: {files[:3]}")
        return
    
    # Parse labels - extract cataract from diagnostic keywords
    def get_cataract_label(keywords):
        if pd.isna(keywords):
            return 0
        keywords_lower = str(keywords).lower()
        if 'cataract' in keywords_lower:
            return 1
        return 0
    
    # Create dataset records with REAL labels
    records = []
    
    for _, row in df.iterrows():
        patient_id = row['ID']
        
        # Left eye
        left_img = row.get('Left-Fundus', '')
        left_keywords = row.get('Left-Diagnostic Keywords', '')
        if pd.notna(left_img) and left_img:
            left_path = os.path.join(img_dir, str(left_img))
            if os.path.exists(left_path):
                label = get_cataract_label(left_keywords)
                records.append({
                    'path': left_path,
                    'binary_label': label,
                    'severity_label': label * 3,  # 0 or 3 (simplified severity)
                    'patient_id': patient_id
                })
        
        # Right eye
        right_img = row.get('Right-Fundus', '')
        right_keywords = row.get('Right-Diagnostic Keywords', '')
        if pd.notna(right_img) and right_img:
            right_path = os.path.join(img_dir, str(right_img))
            if os.path.exists(right_path):
                label = get_cataract_label(right_keywords)
                records.append({
                    'path': right_path,
                    'binary_label': label,
                    'severity_label': label * 3,
                    'patient_id': patient_id
                })
    
    print(f"Total images with labels: {len(records)}")
    
    # Check label distribution
    labels = [r['binary_label'] for r in records]
    n_cataract = sum(labels)
    n_normal = len(labels) - n_cataract
    print(f"Label distribution: Normal={n_normal}, Cataract={n_cataract}")
    
    if n_cataract == 0:
        print("WARNING: No cataract images found! Check label parsing.")
    
    # Split by PATIENT ID (prevent data leakage)
    patient_ids = list(set(r['patient_id'] for r in records))
    np.random.seed(42)
    np.random.shuffle(patient_ids)
    
    n_train = int(0.7 * len(patient_ids))
    n_val = int(0.15 * len(patient_ids))
    
    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train+n_val])
    test_patients = set(patient_ids[n_train+n_val:])
    
    train_records = [r for r in records if r['patient_id'] in train_patients]
    val_records = [r for r in records if r['patient_id'] in val_patients]
    test_records = [r for r in records if r['patient_id'] in test_patients]
    
    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    
    # Dataset class
    class ODIRDataset(Dataset):
        def __init__(self, records, transform):
            self.records = records
            self.transform = transform
        
        def __len__(self):
            return len(self.records)
        
        def __getitem__(self, idx):
            r = self.records[idx]
            img = Image.open(r['path']).convert('RGB')
            img = self.transform(img)
            
            return {
                'image': img,
                'binary_label': torch.tensor(r['binary_label'], dtype=torch.long),
                'severity_label': torch.tensor(r['severity_label'], dtype=torch.long)
            }
    
    # Transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = ODIRDataset(train_records, train_transform)
    val_dataset = ODIRDataset(val_records, val_transform)
    test_dataset = ODIRDataset(test_records, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # ========== MODEL ==========
    print("\n[2/4] Creating model...")
    
    class MobileNetBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()
            
            self.binary_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
            self.severity_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Linear(256, 7)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return {
                'binary_logits': self.binary_head(features),
                'severity_logits': self.severity_head(features),
                'features': features
            }
    
    model = MobileNetBaseline().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights for imbalanced data
    if n_cataract > 0 and n_normal > 0:
        weight_normal = len(labels) / (2 * n_normal)
        weight_cataract = len(labels) / (2 * n_cataract)
        class_weights = torch.tensor([weight_normal, weight_cataract]).to(device)
        print(f"Class weights: Normal={weight_normal:.2f}, Cataract={weight_cataract:.2f}")
    else:
        class_weights = None
    
    binary_criterion = nn.CrossEntropyLoss(weight=class_weights)
    severity_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # ========== TRAINING ==========
    print("\n[3/4] Training...")
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(device)
            binary_labels = batch['binary_label'].to(device)
            severity_labels = batch['severity_label'].to(device)
            
            optimizer.zero_grad()
            output = model(images)
            
            loss = binary_criterion(output['binary_logits'], binary_labels) + \
                   0.5 * severity_criterion(output['severity_logits'], severity_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = output['binary_logits'].argmax(dim=1)
            train_correct += (pred == binary_labels).sum().item()
            train_total += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.3f}'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                binary_labels = batch['binary_label'].to(device)
                output = model(images)
                
                loss = binary_criterion(output['binary_logits'], binary_labels)
                val_loss += loss.item()
                
                pred = output['binary_logits'].argmax(dim=1)
                val_correct += (pred == binary_labels).sum().item()
                val_total += images.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_correct/train_total:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs("/checkpoints/odir_baseline", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc
            }, "/checkpoints/odir_baseline/best_model.pth")
            print(f"  ✓ Saved best model (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # ========== TEST ==========
    print("\n[4/4] Testing...")
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            binary_labels = batch['binary_label'].to(device)
            output = model(images)
            pred = output['binary_logits'].argmax(dim=1)
            test_correct += (pred == binary_labels).sum().item()
            test_total += images.size(0)
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'n_train': len(train_records),
        'n_val': len(val_records),
        'n_test': len(test_records),
        'n_cataract': n_cataract,
        'n_normal': n_normal
    }
    
    torch.save(results, "/checkpoints/odir_baseline/results.pth")
    checkpoint_volume.commit()
    
    print("\n" + "=" * 60)
    print("✅ ODIR BASELINE TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 60)
    print("\nDownload model with:")
    print("  modal volume get cataract-checkpoints /odir_baseline ./checkpoints_modal/odir")
    
    return results


@app.local_entrypoint()
def main():
    print("ODIR-5K Baseline Training")
    print("Usage: modal run baseline_xai/modal_odir.py::train_odir --epochs=50")
