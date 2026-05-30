"""
Modal Training for Multi-Disease Multi-Label Model (CORRECTED).

Fixes from review:
1. Single multi-label head instead of separate heads
2. BCEWithLogitsLoss instead of CrossEntropyLoss  
3. pos_weight for class imbalance
4. Per-disease AUC evaluation
5. MC Dropout for uncertainty

Usage:
    modal run baseline_xai/modal_multitask.py::train_multitask --epochs=50
"""

import modal
import os
from pathlib import Path

# Volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "numpy<2.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "scikit-learn",
        "pandas",
        "tqdm",
        "pillow"
    ])
)

app = modal.App("cataract-multitask-v2", image=image)


@app.function(
    gpu="T4",
    timeout=7200,  # 2 hours
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def train_multitask(epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """
    Train Multi-Label model for 7-disease prediction.
    
    Correct architecture:
    - Single multi-label head (not 7 separate heads)
    - BCEWithLogitsLoss with pos_weight
    - Per-disease AUC evaluation
    - MC Dropout for uncertainty
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from PIL import Image
    
    DISEASE_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 
                     'amd', 'hypertension', 'myopia', 'others']
    
    print("=" * 60)
    print("Multi-Label 7-Disease Training (CORRECTED)")
    print("=" * 60)
    print(f"Epochs: {epochs}, Batch: {batch_size}")
    print(f"Diseases: {', '.join(DISEASE_NAMES)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ========== DATA ==========
    print("\n[1/4] Loading ODIR-5K with multi-disease labels...")
    
    csv_path = "/data/odir/full_df.csv"
    df = pd.read_csv(csv_path)
    print(f"Records: {len(df)}")
    
    # Verify columns
    required_cols = ['ID', 'C', 'D', 'G', 'A', 'H', 'M', 'O']
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        use_direct_cols = False
    else:
        use_direct_cols = True
        print("✓ Using direct disease columns (C, D, G, A, H, M, O)")
        
        # Validate label columns
        print("\nLabel column validation:")
        for col in ['C', 'D', 'G', 'A', 'H', 'M', 'O']:
            non_null = df[col].notna().sum()
            unique = sorted(df[col].dropna().unique())
            print(f"  {col}: {non_null} values, unique: {unique[:5]}")
    
    # Find images
    img_dir = "/data/odir/preprocessed_images"
    if not os.path.exists(img_dir):
        img_dir = "/data/odir/ODIR-5K/ODIR-5K/Training Images"
    print(f"Image directory: {img_dir}")
    
    # Parse labels
    records = []
    parsing_errors = 0
    
    for idx, row in df.iterrows():
        patient_id = row['ID']
        
        if use_direct_cols:
            try:
                labels = {
                    'C': int(row.get('C', 0)) if pd.notna(row.get('C')) else 0,
                    'D': int(row.get('D', 0)) if pd.notna(row.get('D')) else 0,
                    'G': int(row.get('G', 0)) if pd.notna(row.get('G')) else 0,
                    'A': int(row.get('A', 0)) if pd.notna(row.get('A')) else 0,
                    'H': int(row.get('H', 0)) if pd.notna(row.get('H')) else 0,
                    'M': int(row.get('M', 0)) if pd.notna(row.get('M')) else 0,
                    'O': int(row.get('O', 0)) if pd.notna(row.get('O')) else 0
                }
            except:
                parsing_errors += 1
                labels = {'C': 0, 'D': 0, 'G': 0, 'A': 0, 'H': 0, 'M': 0, 'O': 0}
        else:
            # Fallback: parse target string
            target_str = row.get('target', '')
            try:
                import ast
                target_list = ast.literal_eval(str(target_str))
                labels = {
                    'C': target_list[3] if len(target_list) > 3 else 0,
                    'D': target_list[1] if len(target_list) > 1 else 0,
                    'G': target_list[2] if len(target_list) > 2 else 0,
                    'A': target_list[4] if len(target_list) > 4 else 0,
                    'H': target_list[5] if len(target_list) > 5 else 0,
                    'M': target_list[6] if len(target_list) > 6 else 0,
                    'O': target_list[7] if len(target_list) > 7 else 0
                }
            except:
                labels = {'C': 0, 'D': 0, 'G': 0, 'A': 0, 'H': 0, 'M': 0, 'O': 0}
                parsing_errors += 1
        
        # Create records for both eyes
        for col in ['Left-Fundus', 'Right-Fundus']:
            img_name = str(row.get(col, ''))
            if img_name and img_name != 'nan':
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    records.append({
                        'path': img_path,
                        'patient_id': patient_id,
                        'labels': [labels['C'], labels['D'], labels['G'], 
                                  labels['A'], labels['H'], labels['M'], labels['O']]
                    })
    
    if len(records) == 0:
        raise ValueError("CRITICAL: No valid records found!")
    
    print(f"\nTotal images: {len(records)}")
    
    # Disease distribution
    print("\nDisease distribution:")
    disease_counts = [0] * 7
    for r in records:
        for i, val in enumerate(r['labels']):
            disease_counts[i] += val
    
    for i, disease in enumerate(DISEASE_NAMES):
        pct = 100 * disease_counts[i] / len(records)
        print(f"  {disease}: {disease_counts[i]} ({pct:.1f}%)")
    
    if sum(disease_counts) == 0:
        raise ValueError("CRITICAL: ALL LABELS ARE ZERO!")
    
    # ========== DEBUG: LABEL PARSING CHECK ==========
    print("\n=== DEBUG: LABEL PARSING CHECK ===")
    print("Label parsing sample (first 5 records):")
    for i, r in enumerate(records[:5]):
        print(f"  Record {i}: {dict(zip(DISEASE_NAMES, r['labels']))}")
    
    # ========== ROBUST PATIENT-LEVEL SPLIT ==========
    from sklearn.model_selection import train_test_split
    
    # First, group records by patient
    patient_data = {}
    for r in records:
        pid = r['patient_id']
        if pid not in patient_data:
            patient_data[pid] = {'paths': [], 'labels': []}
        patient_data[pid]['paths'].append(r['path'])
        patient_data[pid]['labels'].append(r['labels'])
    
    # Create patient-level aggregated labels (max across eyes/records)
    patient_ids = list(patient_data.keys())
    patient_labels = np.array([np.max(patient_data[pid]['labels'], axis=0) for pid in patient_ids])
    
    # Stratify on ANY disease presence for balanced splits
    any_disease = (patient_labels.sum(axis=1) > 0).astype(int)
    
    print(f"\nPatient-level stats:")
    print(f"  Total patients: {len(patient_ids)}")
    print(f"  Patients with any disease: {sum(any_disease)} ({100*sum(any_disease)/len(any_disease):.1f}%)")
    
    # First split: train vs temp (val+test)
    train_idx, temp_idx = train_test_split(
        range(len(patient_ids)),
        test_size=0.3,
        stratify=any_disease,
        random_state=42
    )
    
    # Second split: val vs test
    temp_any_disease = any_disease[temp_idx]
    val_idx_rel, test_idx_rel = train_test_split(
        range(len(temp_idx)),
        test_size=0.5,
        stratify=temp_any_disease,
        random_state=42
    )
    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]
    
    # Reconstruct image-level records from patient splits
    train_records = []
    val_records = []
    test_records = []
    
    for idx in train_idx:
        pid = patient_ids[idx]
        for path, labels in zip(patient_data[pid]['paths'], patient_data[pid]['labels']):
            train_records.append({'path': path, 'patient_id': pid, 'labels': labels})
    
    for idx in val_idx:
        pid = patient_ids[idx]
        for path, labels in zip(patient_data[pid]['paths'], patient_data[pid]['labels']):
            val_records.append({'path': path, 'patient_id': pid, 'labels': labels})
    
    for idx in test_idx:
        pid = patient_ids[idx]
        for path, labels in zip(patient_data[pid]['paths'], patient_data[pid]['labels']):
            test_records.append({'path': path, 'patient_id': pid, 'labels': labels})
    
    print("✓ Patient-level split complete (no data leakage)")
    
    # ========== DEBUG: VERIFY NO PATIENT OVERLAP ==========
    train_pids = set(r['patient_id'] for r in train_records)
    val_pids = set(r['patient_id'] for r in val_records)
    test_pids = set(r['patient_id'] for r in test_records)
    
    print(f"\n=== PATIENT OVERLAP VERIFICATION ===")
    print(f"Train patients: {len(train_pids)}")
    print(f"Val patients: {len(val_pids)}")
    print(f"Test patients: {len(test_pids)}")
    print(f"Train-Val overlap: {len(train_pids & val_pids)}")
    print(f"Train-Test overlap: {len(train_pids & test_pids)}")
    print(f"Val-Test overlap: {len(val_pids & test_pids)}")
    
    # This should be 0 for all overlaps
    if any(len(overlap) > 0 for overlap in [train_pids & val_pids, train_pids & test_pids, val_pids & test_pids]):
        raise ValueError("CRITICAL: Patient overlap detected between splits!")
    
    # ========== DEBUG: LABEL DISTRIBUTION PER SPLIT ==========
    print(f"\n=== LABEL DISTRIBUTION PER SPLIT ===")
    for split_name, split_records in [("Train", train_records), ("Val", val_records), ("Test", test_records)]:
        print(f"\n{split_name} ({len(split_records)} images):")
        for i, disease in enumerate(DISEASE_NAMES):
            pos_count = sum(r['labels'][i] for r in split_records)
            print(f"  {disease}: {pos_count} positive ({100*pos_count/len(split_records):.1f}%)")
    
    print(f"\nTrain: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    
    # Calculate pos_weights for imbalanced classes (from training split only)
    pos_counts = np.array([sum(r['labels'][i] for r in train_records) for i in range(7)])
    neg_counts = np.array([len(train_records) - c for c in pos_counts])
    
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)
    pos_weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    
    print("\nClass weights (pos_weight):")
    for i, disease in enumerate(DISEASE_NAMES):
        print(f"  {disease}: {pos_weights[i].item():.2f}")
    
    # ========== DATASET ==========
    class MultiLabelDataset(Dataset):
        def __init__(self, records, transform):
            self.records = records
            self.transform = transform
        
        def __len__(self):
            return len(self.records)
        
        def __getitem__(self, idx):
            r = self.records[idx]
            img = Image.open(r['path']).convert('RGB')
            img = self.transform(img)
            
            # Multi-label tensor [7 diseases]
            labels = torch.tensor(r['labels'], dtype=torch.float32)
            return {'image': img, 'labels': labels}
    
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
    
    train_dataset = MultiLabelDataset(train_records, train_transform)
    val_dataset = MultiLabelDataset(val_records, val_transform)
    test_dataset = MultiLabelDataset(test_records, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # ========== MODEL ==========
    print("\n[2/4] Creating Multi-Label model (single head)...")
    
    class MobileNetMultiLabel(nn.Module):
        """
        Single multi-label head (correct architecture).
        Includes MC Dropout for uncertainty estimation.
        """
        def __init__(self, n_diseases=7, dropout_rate=0.3):
            super().__init__()
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(dropout_rate)
            
            # Single multi-label classifier (not 7 separate heads)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n_diseases)  # 7 outputs
            )
        
        def forward(self, x, n_samples=1, return_logits=False):
            features = self.features(x)
            pooled = self.pool(features).flatten(1)
            
            if n_samples > 1:
                # MC Dropout: sample multiple times for uncertainty
                predictions = []
                for _ in range(n_samples):
                    drop_feat = self.dropout(pooled)
                    logits = self.classifier(drop_feat)
                    predictions.append(torch.sigmoid(logits))
                
                predictions = torch.stack(predictions)  # [n_samples, B, 7]
                mean_pred = predictions.mean(dim=0)
                uncertainty = predictions.var(dim=0).mean(dim=1)  # Per-sample
                return mean_pred, uncertainty
            
            # Single forward pass
            drop_feat = self.dropout(pooled)
            logits = self.classifier(drop_feat)
            
            if return_logits:  # Use during training
                return logits
            else:  # Use during inference
                return torch.sigmoid(logits)
    
    model = MobileNetMultiLabel(n_diseases=7).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss: BCEWithLogitsLoss with pos_weight (correct for multi-label)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # ========== TRAINING ==========
    print("\n[3/4] Training...")
    
    best_val_auc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)  # [B, 7]
            
            optimizer.zero_grad()
            logits = model(images, n_samples=1, return_logits=True)  # Explicit logits for training
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Validation with AUC
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['labels']
                
                probs = model(images, n_samples=1)  # Already returns probabilities
                
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Per-disease AUC
        auc_scores = {}
        for i, disease in enumerate(DISEASE_NAMES):
            # Only compute AUC if both classes exist
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            else:
                auc = 0.5  # Random guess if only one class
            auc_scores[disease] = auc
        
        avg_auc = np.mean(list(auc_scores.values()))
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val AUC (avg): {avg_auc:.4f}")
        print("  Per-disease AUC:", end=" ")
        for d, auc in auc_scores.items():
            print(f"{d[:3]}:{auc:.2f}", end=" ")
        print()
        
        # Save best
        if avg_auc > best_val_auc:
            best_val_auc = avg_auc
            patience_counter = 0
            
            os.makedirs("/checkpoints/multitask_v2", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'per_disease_auc': auc_scores
            }, "/checkpoints/multitask_v2/best_model.pth")
            checkpoint_volume.commit()  # Ensure persistence immediately
            print(f"  ✓ Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # ========== CALIBRATE UNCERTAINTY THRESHOLD ==========
    print("\n[4/5] Calibrating uncertainty threshold on validation...")
    
    model.train()  # Keep dropout ON for MC sampling
    val_uncertainties = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            _, uncertainty = model(images, n_samples=10)
            val_uncertainties.extend(uncertainty.cpu().numpy())
    
    # Set threshold at 95th percentile (flag top 5% uncertain)
    uncertainty_threshold = np.percentile(val_uncertainties, 95)
    print(f"  Calibrated uncertainty threshold: {uncertainty_threshold:.4f}")
    print(f"  Val uncertainty mean: {np.mean(val_uncertainties):.4f}")
    
    # ========== TEST ==========
    print("\n[5/5] Testing with calibrated uncertainty...")
    
    # Load best model
    checkpoint = torch.load("/checkpoints/multitask_v2/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # MC Dropout inference
    model.train()  # Keep dropout active for MC sampling
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['labels']
            
            probs, uncertainty = model(images, n_samples=10)  # MC Dropout
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_uncertainties = np.concatenate(all_uncertainties)
    
    # Per-disease test AUC
    print("\nTest Results per Disease:")
    test_auc_scores = {}
    for i, disease in enumerate(DISEASE_NAMES):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        else:
            auc = 0.5
        test_auc_scores[disease] = auc
        print(f"  {disease}: AUC = {auc:.4f}")
    
    avg_test_auc = np.mean(list(test_auc_scores.values()))
    print(f"\nAverage Test AUC: {avg_test_auc:.4f}")
    
    # Uncertainty analysis with calibrated threshold
    uncertain_count = (all_uncertainties > uncertainty_threshold).sum()
    print(f"\nUncertainty Analysis (threshold={uncertainty_threshold:.4f}):")
    print(f"  High uncertainty cases: {uncertain_count} ({100*uncertain_count/len(all_uncertainties):.1f}%)")
    print(f"  Mean uncertainty: {all_uncertainties.mean():.4f}")
    print(f"  Max uncertainty: {all_uncertainties.max():.4f}")
    
    # Save results
    results = {
        'best_val_auc': best_val_auc,
        'avg_test_auc': avg_test_auc,
        'per_disease_test_auc': test_auc_scores,
        'uncertainty_threshold': float(uncertainty_threshold),
        'mean_uncertainty': float(all_uncertainties.mean()),
        'high_uncertainty_pct': float(100*uncertain_count/len(all_uncertainties))
    }
    torch.save(results, "/checkpoints/multitask_v2/results.pth")
    checkpoint_volume.commit()
    
    print("\n" + "=" * 60)
    print("✅ MULTI-LABEL TRAINING COMPLETE!")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {avg_test_auc:.4f}")
    print("=" * 60)
    print("\nDownload with:")
    print("  modal volume get cataract-checkpoints /multitask_v2 ./checkpoints_modal/multitask")
    
    return results


@app.local_entrypoint()
def main():
    print("Multi-Label Training (Corrected Architecture)")
    print("Usage: modal run baseline_xai/modal_multitask.py::train_multitask --epochs=50")
