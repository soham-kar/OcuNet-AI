"""
Create v4 Dataset Splits — Multi-Corpus Unified Dataset
========================================================

Integrates all datasets into unified train/val_tune/test splits.

New datasets added in v4:
  - DDR (13,673 images) → DR labels
  - PALM (400 images) → Myopia labels
  - G1020 (1,020 images) → Glaucoma labels
  - ORIGA (650 images) → Glaucoma labels
  - REFUGE (1,200 images) → Glaucoma labels
  - Synthetic Cataract (1,500 images) → Cataract labels
  - Quality Negatives (200 images) → Cataract=0 (blur ≠ cataract)

Strategy:
  - test_v3.csv → test_v4.csv (UNCHANGED — preserves comparability)
  - New data split: 85% train, 15% val_tune
  - Stratified by disease presence
  - All paths relative to data/raw/

Usage:
    python scripts/create_v4_splits.py

Outputs:
    data/train_v4.csv       — Training set
    data/val_tune_v4.csv    — Validation/tuning set
    data/test_v4.csv        — Test set (unchanged from v3)
"""

import pandas as pd
import numpy as np
import csv
import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

BASE_DIR = Path("/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection")
DATA_RAW = BASE_DIR / "data" / "raw"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def make_relative_path(absolute_path):
    """Convert absolute path to relative (from data/raw/)."""
    try:
        return str(Path(absolute_path).relative_to(DATA_RAW))
    except ValueError:
        # If not under data/raw, return as-is
        return absolute_path

def print_distribution(df, name):
    """Print disease distribution for a dataframe."""
    print(f"\n  {name}: {len(df)} images")
    for col in ['Cataract', 'DR', 'Glaucoma', 'Myopia']:
        if col in df.columns:
            pos = df[col].sum()
            print(f"    {col}: {pos} ({pos/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 1: Load v3 Splits (Base)
# ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  V4 Dataset Integration")
print("=" * 60)

print("\n[1/7] Loading v3 base splits...")

train_v3 = pd.read_csv(BASE_DIR / "data" / "train_v3.csv")
val_tune_v3 = pd.read_csv(BASE_DIR / "data" / "val_tune_v3.csv")
test_v3 = pd.read_csv(BASE_DIR / "data" / "test_v3.csv")

# test_v4 = test_v3 (unchanged for comparability)
test_v4 = test_v3.copy()

print(f"  train_v3:     {len(train_v3)} images")
print(f"  val_tune_v3:  {len(val_tune_v3)} images")
print(f"  test_v3:      {len(test_v3)} images (preserved as test_v4)")

# ──────────────────────────────────────────────────────────────
# Step 2: Integrate DDR (DR Labels)
# ──────────────────────────────────────────────────────────────

print("\n[2/7] Integrating DDR (DR labels)...")

ddr_rows = []
for split_name in ['train', 'valid', 'test']:
    label_file = DATA_RAW / "DDR" / "DR_grading" / f"{split_name}.txt"
    img_dir = DATA_RAW / "DDR" / "DR_grading" / split_name
    
    if not label_file.exists():
        print(f"  ⚠️  {label_file} not found, skipping")
        continue
    
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                grade = int(parts[1])
                
                img_path = img_dir / filename
                if img_path.exists():
                    ddr_rows.append({
                        'image_path': make_relative_path(str(img_path)),
                        'Cataract': 0,
                        'DR': 1 if grade > 0 else 0,
                        'Glaucoma': 0,
                        'Myopia': 0,
                        'source': f'DDR_{split_name}',
                        'dr_grade': grade,
                    })

ddr_df = pd.DataFrame(ddr_rows)
print(f"  DDR images: {len(ddr_df)}")
print(f"    DR positive: {ddr_df['DR'].sum()} ({ddr_df['DR'].sum()/len(ddr_df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 3: Integrate PALM (Myopia Labels)
# ──────────────────────────────────────────────────────────────

print("\n[3/7] Integrating PALM (myopia labels)...")

palm_dir = DATA_RAW / "PALM" / "Training" / "Classification"
palm_rows = []

if palm_dir.exists():
    for f in sorted(os.listdir(palm_dir)):
        if not f.endswith('.jpg'):
            continue
        
        img_path = palm_dir / f
        
        # P = pathological myopia, H = high myopia, N = normal
        if f.startswith('P') or f.startswith('H'):
            myopia = 1
        else:
            myopia = 0
        
        palm_rows.append({
            'image_path': make_relative_path(str(img_path)),
            'Cataract': 0,
            'DR': 0,
            'Glaucoma': 0,
            'Myopia': myopia,
            'source': 'PALM',
        })

palm_df = pd.DataFrame(palm_rows)
print(f"  PALM images: {len(palm_df)}")
print(f"    Myopia positive: {palm_df['Myopia'].sum()} ({palm_df['Myopia'].sum()/len(palm_df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 4: Integrate G1020 (Glaucoma Labels)
# ──────────────────────────────────────────────────────────────

print("\n[4/7] Integrating G1020 (glaucoma labels)...")

g1020_csv = DATA_RAW / "other_dataset" / "G1020" / "G1020.csv"
g1020_img_dir = DATA_RAW / "other_dataset" / "G1020" / "Images"
g1020_rows = []

if g1020_csv.exists():
    with open(g1020_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['imageID']
            glaucoma = int(row['binaryLabels'])
            
            img_path = g1020_img_dir / filename
            if img_path.exists():
                g1020_rows.append({
                    'image_path': make_relative_path(str(img_path)),
                    'Cataract': 0,
                    'DR': 0,
                    'Glaucoma': glaucoma,
                    'Myopia': 0,
                    'source': 'G1020',
                })

g1020_df = pd.DataFrame(g1020_rows)
print(f"  G1020 images: {len(g1020_df)}")
print(f"    Glaucoma positive: {g1020_df['Glaucoma'].sum()} ({g1020_df['Glaucoma'].sum()/len(g1020_df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 5: Integrate ORIGA (Glaucoma Labels)
# ──────────────────────────────────────────────────────────────

print("\n[5/7] Integrating ORIGA (glaucoma labels)...")

origa_csv = DATA_RAW / "other_dataset" / "ORIGA" / "OrigaList.csv"
origa_img_dir = DATA_RAW / "other_dataset" / "ORIGA" / "Images"
origa_rows = []

if origa_csv.exists():
    with open(origa_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename']
            glaucoma = int(row['Glaucoma'])
            
            img_path = origa_img_dir / filename
            if img_path.exists():
                origa_rows.append({
                    'image_path': make_relative_path(str(img_path)),
                    'Cataract': 0,
                    'DR': 0,
                    'Glaucoma': glaucoma,
                    'Myopia': 0,
                    'source': 'ORIGA',
                })

origa_df = pd.DataFrame(origa_rows)
print(f"  ORIGA images: {len(origa_df)}")
print(f"    Glaucoma positive: {origa_df['Glaucoma'].sum()} ({origa_df['Glaucoma'].sum()/len(origa_df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 6: Integrate REFUGE (Glaucoma Labels)
# ──────────────────────────────────────────────────────────────

print("\n[6/7] Integrating REFUGE (glaucoma labels)...")

refuge_img_dir = DATA_RAW / "other_dataset" / "REFUGE" / "Images_Square"
refuge_rows = []

for split_name in ['train', 'val']:
    index_file = DATA_RAW / "other_dataset" / "REFUGE" / split_name / "index.json"
    
    if not index_file.exists():
        print(f"  ⚠️  {index_file} not found, skipping REFUGE {split_name}")
        continue
    
    with open(index_file) as f:
        data = json.load(f)
    
    for key, info in data.items():
        filename = info['ImgName']
        glaucoma = int(info.get('Label', 0))
        
        img_path = refuge_img_dir / filename
        if img_path.exists():
            refuge_rows.append({
                'image_path': make_relative_path(str(img_path)),
                'Cataract': 0,
                'DR': 0,
                'Glaucoma': glaucoma,
                'Myopia': 0,
                'source': f'REFUGE_{split_name}',
            })

refuge_df = pd.DataFrame(refuge_rows)
print(f"  REFUGE images: {len(refuge_df)}")
print(f"    Glaucoma positive: {refuge_df['Glaucoma'].sum()} ({refuge_df['Glaucoma'].sum()/len(refuge_df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# Step 7: Integrate Synthetic Cataract + Quality Negatives
# ──────────────────────────────────────────────────────────────

print("\n[7/7] Integrating synthetic cataract data...")

syn_cataract_csv = BASE_DIR / "data" / "synthetic" / "cataract_v4_manifest.csv"
syn_quality_csv = BASE_DIR / "data" / "synthetic" / "cataract_v4_manifest_quality_negatives.csv"

syn_rows = []

# Synthetic cataract (cataract=1)
if syn_cataract_csv.exists():
    syn_cat = pd.read_csv(syn_cataract_csv)
    for _, row in syn_cat.iterrows():
        syn_rows.append({
            'image_path': make_relative_path(row['image_path']),
            'Cataract': 1,
            'DR': 0,
            'Glaucoma': 0,
            'Myopia': 0,
            'source': 'SyntheticCataract',
        })

# Quality negatives (cataract=0)
if syn_quality_csv.exists():
    syn_qual = pd.read_csv(syn_quality_csv)
    for _, row in syn_qual.iterrows():
        syn_rows.append({
            'image_path': make_relative_path(row['image_path']),
            'Cataract': 0,
            'DR': 0,
            'Glaucoma': 0,
            'Myopia': 0,
            'source': 'SyntheticQualityNeg',
        })

syn_df = pd.DataFrame(syn_rows)
print(f"  Synthetic cataract: {len(syn_df[syn_df['Cataract']==1])} images")
print(f"  Quality negatives:  {len(syn_df[syn_df['Cataract']==0])} images")
print(f"  Total synthetic:    {len(syn_df)} images")

# ──────────────────────────────────────────────────────────────
# Combine All New Data
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Combining All Data Sources")
print("=" * 60)

# Combine all new datasets
new_data_dfs = []
for name, df in [
    ("DDR", ddr_df),
    ("PALM", palm_df),
    ("G1020", g1020_df),
    ("ORIGA", origa_df),
    ("REFUGE", refuge_df),
    ("Synthetic", syn_df),
]:
    if len(df) > 0:
        # Ensure consistent columns
        cols = ['image_path', 'Cataract', 'DR', 'Glaucoma', 'Myopia', 'source']
        df_clean = df[[c for c in cols if c in df.columns]].copy()
        new_data_dfs.append(df_clean)
        print_distribution(df_clean, name)

new_data_all = pd.concat(new_data_dfs, ignore_index=True)
print(f"\n  Total new data: {len(new_data_all)} images")

# ──────────────────────────────────────────────────────────────
# Split New Data into Train/Val_Tune
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Creating v4 Splits")
print("=" * 60)

# Create stratification key
new_data_all['n_diseases'] = new_data_all[['Cataract', 'DR', 'Glaucoma', 'Myopia']].sum(axis=1)
new_data_all['strat_group'] = new_data_all['n_diseases'].clip(upper=2).astype(str)

# Split new data: 85% train, 15% val_tune
new_train, new_val_tune = train_test_split(
    new_data_all,
    test_size=0.15,
    stratify=new_data_all['strat_group'],
    random_state=SEED,
)

# Drop helper columns
new_train = new_train.drop(columns=['n_diseases', 'strat_group'])
new_val_tune = new_val_tune.drop(columns=['n_diseases', 'strat_group'])

print(f"\n  New data split:")
print(f"    → train:    {len(new_train)} images")
print(f"    → val_tune: {len(new_val_tune)} images")

# ──────────────────────────────────────────────────────────────
# Merge with v3 Splits
# ──────────────────────────────────────────────────────────────

# Ensure consistent columns
base_cols = ['image_path', 'Cataract', 'DR', 'Glaucoma', 'Myopia', 'source']

train_v3_clean = train_v3[base_cols].copy()
val_tune_v3_clean = val_tune_v3[base_cols].copy()
test_v4_clean = test_v4[base_cols].copy()

# Merge
train_v4 = pd.concat([train_v3_clean, new_train[base_cols]], ignore_index=True)
val_tune_v4 = pd.concat([val_tune_v3_clean, new_val_tune[base_cols]], ignore_index=True)

# Shuffle
train_v4 = train_v4.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_tune_v4 = val_tune_v4.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────

train_v4.to_csv(BASE_DIR / "data" / "train_v4.csv", index=False)
val_tune_v4.to_csv(BASE_DIR / "data" / "val_tune_v4.csv", index=False)
test_v4_clean.to_csv(BASE_DIR / "data" / "test_v4.csv", index=False)

print(f"\n✅ Saved:")
print(f"   data/train_v4.csv:     {len(train_v4)} images")
print(f"   data/val_tune_v4.csv:  {len(val_tune_v4)} images")
print(f"   data/test_v4.csv:      {len(test_v4_clean)} images (unchanged from v3)")

# ──────────────────────────────────────────────────────────────
# Final Summary
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  V4 Dataset Summary")
print("=" * 60)

for name, df in [("train_v4", train_v4), ("val_tune_v4", val_tune_v4), ("test_v4", test_v4_clean)]:
    print_distribution(df, name)

print(f"\n  Total v4 dataset: {len(train_v4) + len(val_tune_v4) + len(test_v4_clean)} images")

# Source breakdown
print(f"\n  Source breakdown (train_v4):")
source_counts = Counter(train_v4['source'])
for src, count in source_counts.most_common():
    print(f"    {src}: {count}")

print(f"\n  V3 → V4 Growth:")
print(f"    train:     {len(train_v3)} → {len(train_v4)} (+{len(train_v4)-len(train_v3)})")
print(f"    val_tune:  {len(val_tune_v3)} → {len(val_tune_v4)} (+{len(val_tune_v4)-len(val_tune_v3)})")
print(f"    test:      {len(test_v3)} → {len(test_v4_clean)} (unchanged)")

print("\n" + "=" * 60)
print("  ✅ V4 dataset creation complete!")
print("=" * 60)
