# modal_train_glaam_odir.py (FINAL ENHANCED VERSION)
"""
Modal Training for GLAAM-Enhanced Multi-Disease Multi-Label Model.

Run:
    # Baseline
    modal run modal_train_glaam_odir.py --config configs/odir_baseline_config.json
    
    # GLAAM
    modal run modal_train_glaam_odir.py --config configs/odir_glaam_config.json
    
    # Bayesian GLAAM (with uncertainty)
    modal run modal_train_glaam_odir.py --config configs/odir_bayesian_config.json
"""

import modal
import json
import argparse
import os
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "numpy<2.0",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "scikit-learn",
    "pandas",
    "tqdm",
    "pillow",
    "matplotlib",
    "seaborn"
])

app = modal.App("glaam-odir-training", image=image)

# Create Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'AMD', 'Hypertension', 'Myopia', 'Others']


@app.function(
    gpu="T4",  # T4 for cost efficiency
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume
    },
    timeout=86400,  # 24 hours
)
def train_glaam_odir(config: dict):
    """
    Train GLAAM-enhanced model on ODIR-5K with uncertainty quantification.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torchvision.transforms as transforms
    from torchvision import models
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from PIL import Image
    import pandas as pd
    import pickle
    
    # Config is passed as dict from local entrypoint
    # Validate config
    required_keys = ['model_name', 'model_type', 'num_classes', 'img_size', 'batch_size', 
                     'attention_stages', 'dropout_rate', 'learning_rate', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"❌ Missing required config key: {key}")
    
    print("=" * 70)
    print(f"🎯 GLAAM-ODIR Training: {config['model_name']}")
    print(f"📊 Model Type: {config['model_type']}")
    print(f"🔧 Attention Stages: {config['attention_stages']}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    
    # ========== DATA LOADING ==========
    print("\n📁 Loading ODIR-5K dataset...")
    # Volume cataract-data mounted at /data, ODIR data is at /data/odir/
    odir_img_dir = Path("/data/odir/preprocessed_images")
    odir_csv = Path("/data/odir/full_df.csv")
    
    if not odir_csv.exists():
        raise FileNotFoundError(f"❌ ODIR CSV not found: {odir_csv}")
    if not odir_img_dir.exists():
        raise FileNotFoundError(f"❌ ODIR image directory not found: {odir_img_dir}")
    
    df = pd.read_csv(odir_csv)
    
    # Map disease columns (your existing mapping)
    disease_cols = {
        'C': 'Cataract', 'D': 'DR', 'G': 'Glaucoma', 
        'A': 'AMD', 'H': 'Hypertension', 'M': 'Myopia', 'O': 'Others'
    }
    
    records = []
    missing_images = 0
    
    for _, row in df.iterrows():
        patient_id = str(row['ID'])
        
        for eye in ['Left', 'Right']:
            filename_col = f'{eye}-Fundus'
            if filename_col not in row or pd.isna(row[filename_col]):
                continue
            
            filename = row[filename_col]
            img_path = odir_img_dir / filename
            
            if not img_path.exists():
                missing_images += 1
                continue
            
            # Extract labels
            labels = []
            for col, disease in disease_cols.items():
                disease_col = f'label_{col}'
                if disease_col in row:
                    labels.append(1 if row[disease_col] == 1 else 0)
                else:
                    diag = str(row.get(f'{eye}-Diagnostic Keywords', '')).lower()
                    labels.append(1 if disease.lower() in diag else 0)
            
            records.append({
                'patient_id': patient_id,
                'path': str(img_path),
                'labels': labels
            })
    
    print(f"✅ Loaded {len(records)} images ({missing_images} missing)")
    
    # ========== FIX 1: LABEL DISTRIBUTION CHECK ==========
    print("\n📊 Label Distribution (BEFORE filtering):")
    all_pos_counts = np.array([sum(r['labels'][i] for r in records) for i in range(7)])
    for i, disease in enumerate(DISEASE_NAMES):
        pos = all_pos_counts[i]
        print(f"   {disease}: {pos}/{len(records)} ({pos/len(records)*100:.2f}%)")
    
    # ========== FIX 2: REMOVE RARE DISEASE CLASSES ==========
    MIN_POSITIVES = 50  # Minimum threshold for a disease to be included
    
    valid_diseases = [i for i, count in enumerate(all_pos_counts) if count >= MIN_POSITIVES]
    removed_diseases = [DISEASE_NAMES[i] for i in range(7) if i not in valid_diseases]
    
    if removed_diseases:
        print(f"\n⚠️  Removing rare diseases (< {MIN_POSITIVES} positives): {removed_diseases}")
    
    # Update disease names
    DISEASE_NAMES_FILTERED = [DISEASE_NAMES[i] for i in valid_diseases]
    print(f"✅ Keeping diseases: {DISEASE_NAMES_FILTERED}")
    
    # Update record labels to only include valid diseases
    for r in records:
        r['labels'] = [r['labels'][i] for i in valid_diseases]
    
    num_classes = len(valid_diseases)
    print(f"📊 Number of classes: {num_classes}")
    
    # ========== FIX 3: PATIENT-LEVEL SPLIT WITH VALIDATION ==========
    patient_ids = list(set(r['patient_id'] for r in records))
    train_pids, temp_pids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)
    
    # Convert to sets for O(1) lookup
    train_pids_set = set(train_pids)
    val_pids_set = set(val_pids)
    test_pids_set = set(test_pids)
    
    # Validate NO patient overlap
    print(f"\n🔍 Patient Split Validation:")
    print(f"   Total patients: {len(patient_ids)}")
    print(f"   Train patients: {len(train_pids)}")
    print(f"   Val patients: {len(val_pids)}")
    print(f"   Test patients: {len(test_pids)}")
    
    overlap_train_val = train_pids_set & val_pids_set
    overlap_train_test = train_pids_set & test_pids_set
    overlap_val_test = val_pids_set & test_pids_set
    
    print(f"   Overlap check (MUST be 0):")
    print(f"     Train∩Val: {len(overlap_train_val)}")
    print(f"     Train∩Test: {len(overlap_train_test)}")
    print(f"     Val∩Test: {len(overlap_val_test)}")
    
    if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        raise ValueError("🚨 CRITICAL: Patient overlap detected! Data leakage!")
    
    print("   ✅ No patient overlap - data leakage prevented")
    
    train_records = [r for r in records if r['patient_id'] in train_pids_set]
    val_records = [r for r in records if r['patient_id'] in val_pids_set]
    test_records = [r for r in records if r['patient_id'] in test_pids_set]
    
    print(f"\n📊 Split: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    
    # ========== FIX 4: LABEL DISTRIBUTION PER SPLIT ==========
    print("\n📊 Label Distribution per Split:")
    for split_name, split_records in [("Train", train_records), ("Val", val_records), ("Test", test_records)]:
        print(f"\n   {split_name} ({len(split_records)} images):")
        for i, disease in enumerate(DISEASE_NAMES_FILTERED):
            pos = sum(r['labels'][i] for r in split_records)
            print(f"      {disease}: {pos} ({pos/len(split_records)*100:.1f}%)")
    
    # Calculate pos_weights for class imbalance (using filtered classes)
    pos_counts = np.array([sum(r['labels'][i] for r in train_records) for i in range(num_classes)])
    neg_counts = np.array([len(train_records) - c for c in pos_counts])
    pos_counts = np.maximum(pos_counts, 1)  # Avoid division by zero
    raw_weights = neg_counts / pos_counts
    
    # Cap max pos_weight for numerical stability
    MAX_WEIGHT = 10.0
    capped_weights = np.minimum(raw_weights, MAX_WEIGHT)
    pos_weights = torch.tensor(capped_weights, dtype=torch.float32).to(device)
    
    print("\n⚖️  Class weights (pos_weight, capped at 10.0):")
    for i, disease in enumerate(DISEASE_NAMES_FILTERED):
        print(f"   {disease}: {pos_weights[i].item():.2f} (raw: {raw_weights[i]:.2f})")
    
    # ========== DATASET CLASS ==========
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
            labels = torch.tensor(r['labels'], dtype=torch.float32)
            return {'image': img, 'labels': labels, 'path': r['path']}
    
    # ========== TRANSFORMS ==========
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    img_size = config.get('img_size', 224)
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    train_dataset = MultiLabelDataset(train_records, train_transform)
    val_dataset = MultiLabelDataset(val_records, val_transform)
    test_dataset = MultiLabelDataset(test_records, val_transform)
    
    # Create loaders (num_workers=2 for T4 memory safety)
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n🚂 Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # ========== MODEL COMPONENTS (inline for Modal) ==========
    
    class GlobalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return y
    
    class LocalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.conv(x)
    
    class GLAAMBlock(nn.Module):
        def __init__(self, in_channels, reduction=16, use_residual=True, dropout_rate=0.0):
            """
            GLAAM Block with configurable reduction ratio.
            
            Args:
                in_channels: Number of input channels
                reduction: Channel reduction ratio (4, 8, 16, 32) - lower = more capacity
                use_residual: Whether to use residual connection
                dropout_rate: Dropout after attention (0.0 recommended - dropout destroys spatial patterns!)
            """
            super().__init__()
            self.global_branch = GlobalAttentionBranch(in_channels, reduction)
            self.local_branch = LocalAttentionBranch(in_channels, reduction)
            self.use_residual = use_residual
            self.dropout_rate = dropout_rate
            
            # Learnable fusion parameter (GLAAI innovation)
            # Instead of simple multiplication, learn optimal balance
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        def forward(self, x, return_attention=False):
            global_weights = self.global_branch(x)  # (B, C, 1, 1)
            local_weights = self.local_branch(x)    # (B, C, H, W)
            
            # Learnable attention fusion (more expressive than multiplication)
            # Alpha balances global vs local attention importance
            combined_attention = self.alpha * global_weights + (1 - self.alpha) * local_weights
            out = x * combined_attention
            
            # IMPORTANT: Only apply dropout if > 0 (dropout destroys spatial attention patterns!)
            if self.dropout_rate > 0 and self.training:
                out = nn.functional.dropout(out, p=self.dropout_rate, training=True)
            
            if return_attention:
                attn_dict = {
                    'combined_attention': combined_attention,
                    'global_weight': global_weights,
                    'local_weight': local_weights,
                    'alpha': self.alpha.item()
                }
                return (x + out) if self.use_residual else out, attn_dict
            else:
                return (x + out) if self.use_residual else out
    
    # ========== MODEL CREATION ==========
    print(f"\n🔧 Creating {config['model_type']} model...")
    
    # Correct MobileNetV2 channel dimensions
    stage_channels = {3: 24, 6: 32, 10: 64, 13: 96, 16: 160, 17: 320}
    
    if config['model_type'] == 'baseline':
        # Standard MobileNetV2
        class MobileNetBaseline(nn.Module):
            def __init__(self, n_diseases=7, dropout_rate=0.2):
                super().__init__()
                mobilenet = models.mobilenet_v2(pretrained=config.get('pretrained', True))
                self.features = mobilenet.features
                self.dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    nn.Linear(256, n_diseases)
                )
            
            def forward(self, x, return_attention=False):
                x = self.features(x)
                x = nn.functional.adaptive_avg_pool2d(x, 1)
                x = torch.flatten(x, 1)
                return {'logits': self.classifier(x), 'features': x, 'attention_maps': {}}
        
        model = MobileNetBaseline(n_diseases=num_classes, dropout_rate=config['dropout_rate'])
    
    elif config['model_type'] == 'glaam':
        # GLAAM-enhanced MobileNetV2
        # Extract configurable parameters
        reduction_ratio = config.get('reduction_ratio', 16)
        attention_dropout = config.get('attention_dropout', 0.0)  # Default 0.0 - dropout destroys spatial patterns!
        
        print(f"   Reduction ratio: {reduction_ratio}")
        print(f"   Attention dropout: {attention_dropout}")
        
        class MobileNetWithGLAAM(nn.Module):
            def __init__(self, n_diseases=7, dropout_rate=0.2, attention_stages=[6, 13, 17], reduction=16, attn_dropout=0.0):
                super().__init__()
                mobilenet = models.mobilenet_v2(pretrained=config.get('pretrained', True))
                self.features = mobilenet.features
                
                self.attention_blocks = nn.ModuleDict()
                for stage in attention_stages:
                    if stage in stage_channels:
                        self.attention_blocks[f'glaam_{stage}'] = GLAAMBlock(
                            stage_channels[stage], reduction=reduction, dropout_rate=attn_dropout
                        )
                
                self.dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    nn.Linear(256, n_diseases)
                )
                self._attention_storage = {}

            
            def forward(self, x, return_attention=False):
                if return_attention:
                    self._attention_storage = {}
                
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if f'glaam_{i}' in self.attention_blocks:
                        if return_attention:
                            x, attn_dict = self.attention_blocks[f'glaam_{i}'](x, return_attention=True)
                            self._attention_storage[f'stage_{i}'] = attn_dict
                        else:
                            x = self.attention_blocks[f'glaam_{i}'](x)
                
                x = nn.functional.adaptive_avg_pool2d(x, 1)
                features = torch.flatten(x, 1)
                logits = self.classifier(features)
                
                return {'logits': logits, 'features': features, 'attention_maps': self._attention_storage}
        
        model = MobileNetWithGLAAM(
            n_diseases=num_classes,
            dropout_rate=config['dropout_rate'],
            attention_stages=config['attention_stages'],
            reduction=reduction_ratio,
            attn_dropout=attention_dropout
        )
    
    elif config['model_type'] == 'bayesian-glaam':
        # Base GLAAM model
        class MobileNetWithGLAAM(nn.Module):
            def __init__(self, n_diseases=7, dropout_rate=0.2, attention_stages=[6, 13, 17]):
                super().__init__()
                mobilenet = models.mobilenet_v2(pretrained=config.get('pretrained', True))
                self.features = mobilenet.features
                
                self.attention_blocks = nn.ModuleDict()
                for stage in attention_stages:
                    if stage in stage_channels:
                        self.attention_blocks[f'glaam_{stage}'] = GLAAMBlock(
                            stage_channels[stage], reduction=16
                        )
                
                self.dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Sequential(
                    nn.Linear(1280, 256),
                    nn.ReLU(inplace=True),
                    self.dropout,
                    nn.Linear(256, n_diseases)
                )
                self._attention_storage = {}
            
            def forward(self, x, return_attention=False):
                if return_attention:
                    self._attention_storage = {}
                
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if f'glaam_{i}' in self.attention_blocks:
                        if return_attention:
                            x, attn_dict = self.attention_blocks[f'glaam_{i}'](x, return_attention=True)
                            self._attention_storage[f'stage_{i}'] = attn_dict
                        else:
                            x = self.attention_blocks[f'glaam_{i}'](x)
                
                x = nn.functional.adaptive_avg_pool2d(x, 1)
                features = torch.flatten(x, 1)
                logits = self.classifier(features)
                
                return {'logits': logits, 'features': features, 'attention_maps': self._attention_storage}
        
        # Bayesian wrapper with MC Dropout
        class BayesianWrapper(nn.Module):
            def __init__(self, base, num_samples=10, dropout_rate=0.3):
                super().__init__()
                self.base = base
                self.num_samples = num_samples
                self.base.dropout.p = dropout_rate  # Increase dropout for MC
            
            def forward(self, x, return_uncertainty=False, return_attention=False):
                def enable_dropout(m):
                    if isinstance(m, nn.Dropout):
                        m.train()
                
                self.base.apply(enable_dropout)
                
                logit_samples = []
                attention_samples = []
                
                for _ in range(self.num_samples):
                    outputs = self.base(x, return_attention=True)
                    logit_samples.append(outputs['logits'])
                    
                    if outputs['attention_maps']:
                        final_stage = list(outputs['attention_maps'].keys())[-1]
                        attn_map = outputs['attention_maps'][final_stage]['combined_attention']
                        attention_samples.append(attn_map.cpu())
                
                logits_stack = torch.stack(logit_samples)
                predictions = {
                    'logits': logits_stack.mean(dim=0),
                    'logits_std': logits_stack.std(dim=0),
                    'features': outputs['features'],
                    'attention_maps': outputs['attention_maps']
                }
                
                if return_uncertainty and attention_samples:
                    uncertainty = {
                        'attention_variance': torch.stack(attention_samples).var(dim=0),
                        'num_samples': self.num_samples
                    }
                    return predictions, uncertainty
                
                return predictions
        
        base_model = MobileNetWithGLAAM(
            n_diseases=num_classes,
            dropout_rate=config['dropout_rate'],
            attention_stages=config['attention_stages']
        )
        model = BayesianWrapper(base_model, num_samples=config.get('mc_samples', 30), dropout_rate=config['dropout_rate'])
    
    elif config['model_type'] == 'glaam-4x':
        # 🔥 GLAAM-4X: Disease-Specific Attention Specialists
        print("   Loading GLAAM-4X model with disease-specific attention heads...")
        
        # Define GLAAM-4X inline (simpler than importing for Modal)
        DISEASE_NAMES_4X = ['DR', 'Glaucoma', 'Cataract', 'Myopia']
        
        class MultiScaleGLAAM(nn.Module):
            """Multi-scale attention for DR microaneurysms."""
            def __init__(self, in_channels, reduction=4):
                super().__init__()
                self.attention_fine = GLAAMBlock(in_channels, reduction=reduction)
                self.attention_medium = GLAAMBlock(in_channels, reduction=reduction * 2)
                self.attention_coarse = GLAAMBlock(in_channels, reduction=reduction * 4)
                self.scale_fusion = nn.Sequential(
                    nn.Conv2d(in_channels * 3, in_channels, 1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
                self.scale_weights = nn.Parameter(torch.ones(3) / 3)
            
            def forward(self, x, return_attention=False):
                B, C, H, W = x.shape
                fine = self.attention_fine(x)
                x_medium = nn.functional.avg_pool2d(x, 2)
                medium = self.attention_medium(x_medium)
                medium_up = nn.functional.interpolate(medium, size=(H, W), mode='bilinear', align_corners=False)
                x_coarse = nn.functional.avg_pool2d(x, 4)
                coarse = self.attention_coarse(x_coarse)
                coarse_up = nn.functional.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)
                combined = torch.cat([fine, medium_up, coarse_up], dim=1)
                output = self.scale_fusion(combined)
                if return_attention:
                    return output, (fine + medium_up + coarse_up) / 3
                return output
        
        class DiseaseGatingNetwork(nn.Module):
            """Learn which specialist to trust."""
            def __init__(self, in_channels, n_diseases=4):
                super().__init__()
                self.gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(256, n_diseases),
                    nn.Softmax(dim=1)
                )
            def forward(self, x):
                return self.gate(x)
        
        class GLAAM_4X(nn.Module):
            """4 disease-specific attention specialists."""
            def __init__(self, pretrained=True, dropout_rate=0.3, n_diseases=4):
                super().__init__()
                mobilenet = models.mobilenet_v2(pretrained=pretrained)
                self.backbone = mobilenet.features
                self.attention_heads = nn.ModuleDict({
                    'DR': MultiScaleGLAAM(1280, reduction=4),
                    'Glaucoma': GLAAMBlock(1280, reduction=8),
                    'Cataract': GLAAMBlock(1280, reduction=16),
                })
                self.disease_gate = DiseaseGatingNetwork(1280, n_diseases=4)
                self.classifiers = nn.ModuleDict({
                    'DR': nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)),
                    'Glaucoma': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
                    'Cataract': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
                    'Myopia': nn.Sequential(nn.Linear(1280, 64), nn.ReLU(), nn.Linear(64, 1))
                })
                self._attention_storage = {}
            
            def forward(self, x, return_attention=False):
                features = self.backbone(x)
                gate_weights = self.disease_gate(features)
                specialist_features = {}
                attention_maps = {}
                for disease in DISEASE_NAMES_4X:
                    if disease in self.attention_heads:
                        if return_attention:
                            attended, attn_map = self.attention_heads[disease](features, return_attention=True)
                            attention_maps[disease] = attn_map
                        else:
                            attended = self.attention_heads[disease](features)
                    else:
                        attended = features
                    pooled = nn.functional.adaptive_avg_pool2d(attended, 1).flatten(1)
                    specialist_features[disease] = pooled
                logits = []
                for disease in DISEASE_NAMES_4X:
                    disease_logit = self.classifiers[disease](specialist_features[disease])
                    logits.append(disease_logit.squeeze(-1))
                output_logits = torch.stack(logits, dim=1)
                if return_attention:
                    self._attention_storage = {'attention_maps': attention_maps, 'gate_weights': gate_weights}
                    return {'logits': output_logits, 'features': specialist_features, 'attention_maps': self._attention_storage}
                return {'logits': output_logits, 'features': None, 'attention_maps': {}}
        
        model = GLAAM_4X(
            pretrained=config.get('pretrained', True),
            dropout_rate=config['dropout_rate'],
            n_diseases=num_classes
        )
        print(f"   ✅ GLAAM-4X initialized with MultiScaleGLAAM for DR")
    
    else:
        raise ValueError(f"❌ Unknown model_type: {config['model_type']}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # ========== TRAINING FUNCTIONS ==========
    
    def compute_metrics(preds, labels, threshold=0.5):
        """Compute comprehensive clinical-quality metrics for imbalanced data"""
        from sklearn.metrics import balanced_accuracy_score, roc_curve
        
        class_aucs = {}
        class_pr_aucs = {}
        class_balanced_accs = {}
        class_sens_at_95_spec = {}
        class_optimal_thresholds = {}
        valid_aucs = []
        valid_pr_aucs = []
        valid_baccs = []
        
        binary_preds = (preds > threshold).astype(int)
        
        for i, disease in enumerate(DISEASE_NAMES_FILTERED):
            if len(np.unique(labels[:, i])) > 1:
                try:
                    # ROC-AUC
                    auc = roc_auc_score(labels[:, i], preds[:, i])
                    class_aucs[disease] = auc
                    valid_aucs.append(auc)
                    
                    # PR-AUC (more informative for imbalanced data)
                    pr_auc = average_precision_score(labels[:, i], preds[:, i])
                    class_pr_aucs[disease] = pr_auc
                    valid_pr_aucs.append(pr_auc)
                    
                    # Balanced Accuracy
                    bacc = balanced_accuracy_score(labels[:, i], binary_preds[:, i])
                    class_balanced_accs[disease] = bacc
                    valid_baccs.append(bacc)
                    
                    # Sensitivity @ 95% Specificity (clinical gold standard)
                    fpr, tpr, thresholds = roc_curve(labels[:, i], preds[:, i])
                    idx = np.where(fpr <= 0.05)[0]
                    if len(idx) > 0:
                        class_sens_at_95_spec[disease] = tpr[idx[-1]]
                    else:
                        class_sens_at_95_spec[disease] = 0.0
                    
                    # Youden's optimal threshold
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    class_optimal_thresholds[disease] = thresholds[optimal_idx]
                    
                except Exception as e:
                    class_aucs[disease] = 0.0
                    class_pr_aucs[disease] = 0.0
                    class_balanced_accs[disease] = 0.0
        
        return {
            'mean_auc': np.mean(valid_aucs) if valid_aucs else 0.0,
            'mean_pr_auc': np.mean(valid_pr_aucs) if valid_pr_aucs else 0.0,
            'mean_balanced_acc': np.mean(valid_baccs) if valid_baccs else 0.0,
            'class_auc': class_aucs,
            'class_pr_auc': class_pr_aucs,
            'class_balanced_acc': class_balanced_accs,
            'class_sens_at_95_spec': class_sens_at_95_spec,
            'class_optimal_thresholds': class_optimal_thresholds,
            'valid_classes': len(valid_aucs)
        }
    # MixUp augmentation helper
    def mixup_data(x, y, alpha=0.2):
        """MixUp: mixes two samples for better generalization"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(model, dataloader, criterion, optimizer, device, config):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        use_mixup = config.get('use_mixup', False)
        mixup_alpha = config.get('mixup_alpha', 0.2)
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Apply MixUp if enabled
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            
            if config['model_type'] == 'bayesian-glaam':
                outputs = model(images, return_uncertainty=False)
            else:
                outputs = model(images)
            
            # Compute loss (with MixUp interpolation if enabled)
            if use_mixup:
                loss = lam * criterion(outputs['logits'], labels_a) + (1 - lam) * criterion(outputs['logits'], labels_b)
            else:
                loss = criterion(outputs['logits'], labels)
            
            loss.backward()
            
            # Gradient clipping for stability (increased for Focal Loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs['logits']).detach().cpu().numpy())
            # Use original labels for metrics (not mixed)
            if use_mixup:
                all_labels.extend(labels_a.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        return total_loss / len(dataloader), metrics
    
    def validate_epoch(model, dataloader, criterion, device, config):
        """Validate for one epoch"""
        model.eval()
        
        if config['model_type'] == 'bayesian-glaam':
            def enable_dropout(m):
                if isinstance(m, nn.Dropout):
                    m.train()
            model.apply(enable_dropout)
        
        total_loss = 0
        all_preds, all_labels = [], []
        all_uncertainties = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                
                if config['model_type'] == 'bayesian-glaam':
                    outputs, uncertainty = model(images, return_uncertainty=True)
                    all_uncertainties.append(outputs['logits_std'].cpu().numpy().mean())
                else:
                    outputs = model(images)
                
                loss = criterion(outputs['logits'], labels)
                total_loss += loss.item()
                
                all_preds.extend(torch.sigmoid(outputs['logits']).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        
        if config['model_type'] == 'bayesian-glaam' and all_uncertainties:
            metrics['mean_uncertainty'] = np.mean(all_uncertainties)
        
        return total_loss / len(dataloader), metrics
    
    # ========== FOCAL LOSS (Better for imbalanced data) ==========
    class FocalLoss(nn.Module):
        """Focal Loss for imbalanced classification - down-weights easy negatives"""
        def __init__(self, alpha=0.25, gamma=2, pos_weight=None):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.pos_weight = pos_weight
        
        def forward(self, logits, targets):
            # Compute binary cross-entropy
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            )
            
            # Compute probabilities
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            
            # Focal loss formula: FL = α * (1-p_t)^γ * CE
            focal_weight = self.alpha * (1 - p_t) ** self.gamma
            focal_loss = focal_weight * bce_loss
            
            return focal_loss.mean()
    
    # ========== LOSS & OPTIMIZER ==========
    # Use Focal Loss instead of BCE for better handling of class imbalance
    criterion = FocalLoss(alpha=0.5, gamma=2, pos_weight=pos_weights)
    print("\n📉 Using Focal Loss (alpha=0.5, gamma=2) for class imbalance")
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 0.05)  # Configurable, default 0.05
    )
    print(f"🔧 Using weight_decay={config.get('weight_decay', 0.05)} for regularization")
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # ========== TRAINING LOOP ==========
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")
        
        # Training phase
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validation phase
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, config)
        
        # Print metrics
        print(f"\n📈 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Train AUC: {train_metrics['mean_auc']:.4f} | Val AUC: {val_metrics['mean_auc']:.4f}")
        print(f"   Val Balanced Acc: {val_metrics['mean_balanced_acc']:.4f} | Val PR-AUC: {val_metrics['mean_pr_auc']:.4f}")
        
        if config['model_type'] == 'bayesian-glaam':
            print(f"   Mean Uncertainty: {val_metrics.get('mean_uncertainty', 0):.4f}")
        
        # Per-disease metrics (comprehensive)
        print("\n   Per-disease Val Metrics:")
        print("   Disease        | AUC    | PR-AUC | BalAcc | Sens@95%Spec")
        print("   " + "-"*55)
        for disease in val_metrics.get('class_auc', {}).keys():
            auc = val_metrics['class_auc'].get(disease, 0)
            pr_auc = val_metrics.get('class_pr_auc', {}).get(disease, 0)
            bacc = val_metrics.get('class_balanced_acc', {}).get(disease, 0)
            sens = val_metrics.get('class_sens_at_95_spec', {}).get(disease, 0)
            print(f"   {disease:13} | {auc:.4f} | {pr_auc:.4f} | {bacc:.4f} | {sens:.4f}")
        
        # Scheduler
        scheduler.step(val_metrics['mean_auc'])
        
        # Checkpointing
        if val_metrics['mean_auc'] > best_val_auc:
            best_val_auc = val_metrics['mean_auc']
            patience_counter = 0
            
            save_path = f"/checkpoints/{config['model_name']}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': config,
                'class_auc': val_metrics.get('class_auc', {})
            }, save_path)
            
            print(f"💾 Saved best model: {config['model_name']}_best.pth (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
    
    # ========== FINAL TEST EVALUATION ==========
    print(f"\n{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}")
    
    # Load best model
    checkpoint = torch.load(f"/checkpoints/{config['model_name']}_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    if config['model_type'] == 'bayesian-glaam':
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        model.apply(enable_dropout)
    
    all_preds, all_labels, all_paths = [], [], []
    all_logits = []  # NEW: Track raw logits for temperature scaling
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            if config['model_type'] == 'bayesian-glaam':
                outputs = model(images, return_uncertainty=False)
            else:
                outputs = model(images)
            
            # Save raw logits BEFORE sigmoid
            all_logits.extend(outputs['logits'].cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs['logits']).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(batch['path'])
    
    test_preds = np.array(all_preds)
    test_labels = np.array(all_labels)
    test_metrics = compute_metrics(test_preds, test_labels)
    
    print(f"\n🎯 Final Test AUC: {test_metrics['mean_auc']:.4f}")
    
    if 'class_auc' in test_metrics:
        print("\nPer-disease Test AUC:")
        for disease, auc in test_metrics['class_auc'].items():
            print(f"  {disease}: {auc:.4f}")
    
    # Save predictions for statistical analysis
    pred_save_path = f"/checkpoints/{config['model_name']}_predictions.pkl"
    with open(pred_save_path, 'wb') as f:
        pickle.dump({
            'predictions': test_preds,
            'logits': np.array(all_logits),  # NEW: Save raw logits for temperature scaling
            'labels': test_labels,
            'paths': all_paths,
            'disease_names': DISEASE_NAMES_FILTERED
        }, f)
    print(f"💾 Saved predictions to: {config['model_name']}_predictions.pkl")
    print(f"   ✅ Included raw logits for calibration")
    
    # Generate attention maps for XAI (if GLAAM model)
    if config['model_type'] in ['glaam', 'bayesian-glaam']:
        print("\n📸 Generating disease-specific attention maps for XAI...")

        # ✅ FIX: Select balanced disease-specific samples instead of first batch
        # Old code: batch = next(iter(test_loader)) → always first batch (mostly Normal!)
        # New code: pick 2 samples per disease + 2 Normal = 10 total

        # Collect all test data indices grouped by disease
        all_test_images = []
        all_test_labels_list = []
        all_test_paths_list = []

        for batch in test_loader:
            all_test_images.append(batch['image'])
            all_test_labels_list.append(batch['labels'])
            all_test_paths_list.extend(batch['path'])

        all_test_images_cat = torch.cat(all_test_images, dim=0)  # (N, 3, H, W)
        all_test_labels_cat = torch.cat(all_test_labels_list, dim=0)  # (N, num_classes)

        selected_indices = []
        samples_per_disease = 2

        # Select disease-positive samples (2 per disease)
        for disease_idx in range(num_classes):
            disease_positive_idx = (all_test_labels_cat[:, disease_idx] == 1).nonzero(as_tuple=True)[0]
            if len(disease_positive_idx) >= samples_per_disease:
                chosen = disease_positive_idx[:samples_per_disease]
            else:
                chosen = disease_positive_idx
            selected_indices.extend(chosen.tolist())

        # Fill remaining slots with Normal samples (all labels = 0)
        normal_idx = (all_test_labels_cat.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        n_normal = max(0, 10 - len(selected_indices))
        if len(normal_idx) >= n_normal:
            selected_indices.extend(normal_idx[:n_normal].tolist())

        selected_indices = selected_indices[:10]  # Cap at 10

        x = all_test_images_cat[selected_indices].to(device)
        labels_batch = all_test_labels_cat[selected_indices]
        paths = [all_test_paths_list[i] for i in selected_indices]

        # Log selected samples for verification
        print(f"   Selected {len(selected_indices)} samples for attention maps:")
        for i, (lbl, pth) in enumerate(zip(labels_batch, paths)):
            disease_flags = {DISEASE_NAMES_FILTERED[j]: int(lbl[j].item()) for j in range(num_classes)}
            active = [d for d, v in disease_flags.items() if v == 1] or ['Normal']
            print(f"     [{i+1}] {pth.split('/')[-1]} → {active}")

        with torch.no_grad():
            outputs = model(x, return_attention=True)

        if outputs['attention_maps']:
            attn_save_path = f"/checkpoints/{config['model_name']}_attention_maps.pkl"
            with open(attn_save_path, 'wb') as f:
                pickle.dump({
                    'attention_maps': {k: v['combined_attention'].cpu() for k, v in outputs['attention_maps'].items()},
                    'images': x.cpu(),
                    'labels': labels_batch,
                    'paths': paths
                }, f)
            print(f"💾 Saved attention maps for 10 samples")
    
    # Commit volumes
    checkpoint_volume.commit()
    
    print(f"\n✅ Training complete!")
    print(f"   Best Val AUC: {best_val_auc:.4f}")
    print(f"   Test AUC: {test_metrics['mean_auc']:.4f}")
    
    return {
        'best_val_auc': best_val_auc,
        'test_auc': test_metrics['mean_auc'],
        'model_name': config['model_name'],
        'model_type': config['model_type'],
        'class_auc': test_metrics['class_auc']
    }


@app.local_entrypoint()
def main(config: str = "configs/odir_glaam_config.json"):
    """
    Entry point for Modal training.
    
    Usage:
        modal run modal_train_glaam_odir.py  # Uses default GLAAM config
        modal run modal_train_glaam_odir.py --config configs/odir_baseline_config.json
        modal run modal_train_glaam_odir.py --config configs/odir_bayesian_config.json
    """
    print(f"🚀 Loading config from: {config}")
    
    # Load config locally and pass as dict
    with open(config, 'r') as f:
        config_dict = json.load(f)
    
    print(f"📊 Model: {config_dict['model_name']} ({config_dict['model_type']})")
    
    result = train_glaam_odir.remote(config_dict)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Model: {result['model_name']}")
    print(f"   Type: {result['model_type']}")
    print(f"   Best Val AUC: {result['best_val_auc']:.4f}")
    print(f"   Test AUC: {result['test_auc']:.4f}")
    print(f"\n💾 Checkpoints saved to Modal volume: cataract-checkpoints")
    print("="*70)
