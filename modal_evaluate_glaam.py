# modal_evaluate_glaam.py
"""
Standalone Modal evaluation to extract logits from saved GLAAM checkpoint.

Run:
    modal run modal_evaluate_glaam.py
"""

import modal
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "numpy<2.0",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "scikit-learn",
    "pandas",
    "tqdm",
    "pillow"
])

app = modal.App("glaam-evaluation", image=image)

# Create Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# Mount local code directory so Modal can import models
code_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/code"
)

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'AMD', 'Hypertension', 'Myopia', 'Others']


# ========== STANDALONE EVALUATION FUNCTION ==========
@app.function(
    gpu="T4",
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume
    },
    mounts=[code_mount],
    timeout=3600  # 1 hour
)
def evaluate_saved_checkpoint(
    checkpoint_path="/checkpoints/glaam_final_best.pth",
    output_name="glaam_final_with_logits"
):
    """
    Standalone evaluation function - loads saved checkpoint and extracts logits.
    Does NOT train - only runs inference on test set.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        output_name: Name for output predictions file (without .pkl extension)
    
    Returns:
        dict with evaluation results
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    import pickle
    
    print("=" * 70)
    print("🔍 GLAAM STANDALONE EVALUATION (Logits Extraction)")
    print("=" * 70)
    print(f"📂 Checkpoint: {checkpoint_path}")
    print(f"📊 Output: {output_name}.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    # ========== LOAD DATA (same as training) ==========
    print("\n📁 Loading ODIR-5K dataset...")
    odir_img_dir = Path("/data/odir/preprocessed_images")
    odir_csv = Path("/data/odir/full_df.csv")
    
    if not odir_csv.exists():
        raise FileNotFoundError(f"❌ ODIR CSV not found: {odir_csv}")
    if not odir_img_dir.exists():
        raise FileNotFoundError(f"❌ ODIR image directory not found: {odir_img_dir}")
    
    df = pd.read_csv(odir_csv)
    
    # Build records (same logic as training)
    records = []
    missing_images = 0
    for _, row in df.iterrows():
        patient_id = row['ID']
        img_path = odir_img_dir / row['filename']
        if not img_path.exists():
            missing_images += 1
            continue
        
        labels = [
            row.get('Cataract', 0),
            row.get('DR', 0),
            row.get('Glaucoma', 0),
            row.get('AMD', 0),
            row.get('Hypertension', 0),
            row.get('Myopia', 0),
            row.get('Others', 0)
        ]
        
        records.append({
            'patient_id': patient_id,
            'path': str(img_path),
            'labels': labels
        })
    
    print(f"✅ Loaded {len(records)} images ({missing_images} missing)")
    
    # Filter rare diseases (same as training)
    MIN_POSITIVES = 50
    all_pos_counts = np.array([sum(r['labels'][i] for r in records) for i in range(7)])
    valid_diseases = [i for i, count in enumerate(all_pos_counts) if count >= MIN_POSITIVES]
    DISEASE_NAMES_FILTERED = [DISEASE_NAMES[i] for i in valid_diseases]
    
    print(f"✅ Kept {len(DISEASE_NAMES_FILTERED)} diseases: {DISEASE_NAMES_FILTERED}")
    
    # Update labels
    for r in records:
        r['labels'] = [r['labels'][i] for i in valid_diseases]
    
    num_classes = len(valid_diseases)
    
    # Patient-level split (SAME random_state=42 as training!)
    patient_ids = list(set(r['patient_id'] for r in records))
    train_pids, temp_pids = train_test_split(patient_ids, test_size=0.3, random_state=42)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.5, random_state=42)
    
    test_pids_set = set(test_pids)
    test_records = [r for r in records if r['patient_id'] in test_pids_set]
    
    print(f"📊 Test set: {len(test_records)} images from {len(test_pids)} patients")
    
    # Dataset class
    class MultiLabelDataset(Dataset):
        def __init__(self, records, transform=None):
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
    
    # Transforms (same as evaluation/validation - no augmentation)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_dataset = MultiLabelDataset(test_records, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"🚂 Test batches: {len(test_loader)}")
    
    # ========== LOAD MODEL ==========
    print(f"\n📦 Loading model from checkpoint...")
    
    # Add mounted code to Python path
    import sys
    sys.path.insert(0, "/root/code")
    
    # Now we can import local model classes
    from models.backbones.mobilenet_glaam_odir import MobileNetV2WithGLAAM_ODIR
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   Model: {config.get('model_name', 'unknown')}")
        print(f"   Type: {config.get('model_type', 'unknown')}")
    else:
        # Default GLAAM config
        config = {
            'model_type': 'glaam',
            'attention_stages': [6, 13, 17],
            'dropout_rate': 0.2
        }
        print("   ⚠️  No config in checkpoint, using default GLAAM settings")
    
    # Create model
    if config.get('model_type') == 'baseline':
        # Baseline without attention
        model = MobileNetV2WithGLAAM_ODIR(
            num_classes=num_classes,
            pretrained=False,
            attention_stages=[],
            dropout_rate=config.get('dropout_rate', 0.2)
        )
    else:
        # GLAAM model
        model = MobileNetV2WithGLAAM_ODIR(
            num_classes=num_classes,
            pretrained=False,
            attention_stages=config.get('attention_stages', [6, 13, 17]),
            dropout_rate=config.get('dropout_rate', 0.2)
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    
    # ========== RUN EVALUATION ==========
    print(f"\n🔬 Running evaluation on test set...")
    
    all_logits = []
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            
            # Extract raw logits BEFORE sigmoid
            logits = outputs['logits']
            all_logits.append(logits.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_paths.extend(batch['path'])
    
    # Concatenate results
    test_logits = np.concatenate(all_logits, axis=0)
    test_preds = np.concatenate(all_preds, axis=0)
    test_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Samples: {len(test_preds)}")
    print(f"   Logits shape: {test_logits.shape}")
    print(f"   Logits range: [{test_logits.min():.2f}, {test_logits.max():.2f}]")
    print(f"   Predictions range: [{test_preds.min():.3f}, {test_preds.max():.3f}]")
    
    # Compute metrics
    class_aucs = {}
    for i, disease in enumerate(DISEASE_NAMES_FILTERED):
        if len(np.unique(test_labels[:, i])) > 1:
            auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
            class_aucs[disease] = auc
    
    mean_auc = np.mean(list(class_aucs.values())) if class_aucs else 0.0
    
    print(f"\n📊 Performance:")
    print(f"   Mean AUC: {mean_auc:.4f}")
    for disease, auc in class_aucs.items():
        print(f"     {disease}: {auc:.4f}")
    
    # ========== SAVE RESULTS ==========
    output_path = f"/checkpoints/{output_name}_predictions.pkl"
    
    results = {
        'predictions': test_preds,
        'logits': test_logits,  # ✅ RAW LOGITS!
        'labels': test_labels,
        'paths': all_paths,
        'disease_names': DISEASE_NAMES_FILTERED
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Saved predictions with logits:")
    print(f"   Path: {output_path}")
    print(f"   Keys: {list(results.keys())}")
    print(f"   ✅ Included 'logits' for temperature scaling!")
    
    # Commit volume changes
    checkpoint_volume.commit()
    
    return {
        'samples': len(test_preds),
        'mean_auc': mean_auc,
        'class_auc': class_aucs,
        'logits_range': [float(test_logits.min()), float(test_logits.max())],
        'output_path': output_path
    }


@app.local_entrypoint()
def main():
    """
    Local entrypoint - runs the evaluation function.
    """
    print("🚀 Starting GLAAM evaluation on Modal...")
    result = evaluate_saved_checkpoint.remote()
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Samples evaluated: {result['samples']}")
    print(f"📈 Mean AUC: {result['mean_auc']:.4f}")
    print(f"📁 Output saved to: {result['output_path']}")
    print(f"   Logits range: [{result['logits_range'][0]:.2f}, {result['logits_range'][1]:.2f}]")
    print("\n🎯 Next step: Download predictions and run temperature scaling")
    print("=" * 70)
