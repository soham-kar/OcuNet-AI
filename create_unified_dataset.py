"""
Create a unified multi-label fundus dataset CSV from all available sources:
  - ODIR-5K (excludes test set used for final evaluation)
  - JSIEC (1,000 images with fine-grained categories)
  - RFMiD (2,560 images with 45 disease labels)
  - PALM (400 pathologic myopia images)

Output: data/combined_fundus_dataset.csv
Columns: image_path, Cataract, DR, Glaucoma, Myopia, source
"""

import os
import glob
import pickle
import pandas as pd
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_CSV = "data/combined_fundus_dataset.csv"

# ODIR test IDs to exclude (from glaam_final_predictions.pkl)
PREDICTIONS_PKL = "checkpoints_glaam/glaam_final_predictions.pkl"

# ═══════════════════════════════════════════════════════════════════════════
# 1. ODIR-5K
# ═══════════════════════════════════════════════════════════════════════════

def load_odir():
    """Load ODIR-5K, map to target diseases, exclude test set."""
    print("Loading ODIR-5K...")
    odir_root = "data/raw/odir"
    df = pd.read_csv(os.path.join(odir_root, "full_df.csv"))

    # Map ODIR columns: N=Normal, D=DR, G=Glaucoma, C=Cataract, A=AMD, H=Hypertension, M=Myopia, O=Others
    df['Cataract'] = df['C'].astype(int)
    df['DR'] = df['D'].astype(int)
    df['Glaucoma'] = df['G'].astype(int)
    df['Myopia'] = df['M'].astype(int)

    # Build image paths relative to data/raw/ (for Modal cloud compatibility)
    df['image_path'] = df['filename'].apply(lambda x: f"odir/preprocessed_images/{x}")

    # Exclude test set IDs
    if os.path.exists(PREDICTIONS_PKL):
        pred_data = pickle.load(open(PREDICTIONS_PKL, 'rb'))
        test_filenames = set([os.path.basename(p) for p in pred_data['paths']])
        before = len(df)
        df = df[~df['filename'].isin(test_filenames)]
        after = len(df)
        print(f"  Excluded {before - after} test images. Remaining: {after}")
    else:
        print(f"  Warning: {PREDICTIONS_PKL} not found, including all ODIR images")

    # Keep only images that exist (check with data/raw/ prefix)
    df['full_path'] = df['image_path'].apply(lambda x: os.path.join("data/raw", x))
    df = df[df['full_path'].apply(os.path.exists)]
    df = df.drop(columns=['full_path'])
    df['source'] = 'ODIR'
    return df[['image_path', 'Cataract', 'DR', 'Glaucoma', 'Myopia', 'source']]


# ═══════════════════════════════════════════════════════════════════════════
# 2. JSIEC
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from JSIEC folder names to [Cataract, DR, Glaucoma, Myopia]
JSIEC_MAPPING = {
    '0.0.Normal': [0, 0, 0, 0],
    '0.1.Tessellated fundus': [0, 0, 0, 0],
    '0.2.Large optic cup': [0, 0, 1, 0],
    '0.3.DR1': [0, 1, 0, 0],
    '1.0.DR2': [0, 1, 0, 0],
    '1.1.DR3': [0, 1, 0, 0],
    '2.0.BRVO': [0, 0, 0, 0],
    '2.1.CRVO': [0, 0, 0, 0],
    '3.RAO': [0, 0, 0, 0],
    '4.Rhegmatogenous RD': [0, 0, 0, 0],
    '5.0.CSCR': [0, 0, 0, 0],
    '5.1.VKH disease': [0, 0, 0, 0],
    '6.Maculopathy': [0, 0, 0, 0],
    '7.ERM': [0, 0, 0, 0],
    '8.MH': [0, 0, 0, 0],
    '9.Pathological myopia': [0, 0, 0, 1],
    '10.0.Possible glaucoma': [0, 0, 1, 0],
    '10.1.Optic atrophy': [0, 0, 1, 0],
    '11.Severe hypertensive retinopathy': [0, 0, 0, 0],
    '12.Disc swelling and elevation': [0, 0, 0, 0],
    '13.Dragged Disc': [0, 0, 0, 0],
    '14.Congenital disc abnormality': [0, 0, 0, 0],
    '15.0.Retinitis pigmentosa': [0, 0, 0, 0],
    '15.1.Bietti crystalline dystrophy': [0, 0, 0, 0],
    '16.Peripheral retinal degeneration and break': [0, 0, 0, 0],
    '17.Myelinated nerve fiber': [0, 0, 0, 0],
    '18.Vitreous particles': [0, 0, 0, 0],
    '19.Fundus neoplasm': [0, 0, 0, 0],
    '20.Massive hard exudates': [0, 1, 0, 0],
    '21.Yellow-white spots-flecks': [0, 0, 0, 0],
    '22.Cotton-wool spots': [0, 1, 0, 0],
    '23.Vessel tortuosity': [0, 0, 0, 0],
    '24.Chorioretinal atrophy-coloboma': [0, 0, 0, 0],
    '25.Preretinal hemorrhage': [0, 1, 0, 0],
    '26.Fibrosis': [0, 0, 0, 0],
    '27.Laser Spots': [0, 1, 0, 0],
    '28.Silicon oil in eye': [0, 0, 0, 0],
    '29.0.Blur fundus without PDR': [0, 0, 0, 0],
    '29.1.Blur fundus with suspected PDR': [0, 1, 0, 0],
}


def load_jsiec():
    """Load JSIEC dataset from folder structure."""
    print("Loading JSIEC...")
    jsiec_root = "data/raw/JSIEC"

    records = []
    for folder_name in os.listdir(jsiec_root):
        folder_path = os.path.join(jsiec_root, folder_name)
        if not os.path.isdir(folder_path) or folder_name == '1000images':
            continue  # Skip 1000images subfolder (duplicates)

        labels = JSIEC_MAPPING.get(folder_name, [0, 0, 0, 0])
        image_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.JPG")) + \
                      glob.glob(os.path.join(folder_path, "*.png"))

        for img_path in image_files:
            # Store relative path for Modal cloud compatibility
            rel_path = img_path.replace("data/raw/", "")
            records.append({
                'image_path': rel_path,
                'Cataract': labels[0],
                'DR': labels[1],
                'Glaucoma': labels[2],
                'Myopia': labels[3],
                'source': 'JSIEC',
            })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} images from {len(set(r['image_path'].split('/')[-2] for r in records))} categories")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. RFMiD
# ═══════════════════════════════════════════════════════════════════════════

def load_rfmid():
    """Load RFMiD training and validation sets."""
    print("Loading RFMiD...")
    rfmid_root = "data/raw/RFMiD"

    records = []
    for split in ['Training_set', 'Validation_set']:
        csv_path = os.path.join(rfmid_root, split, f"RFMiD_{split.split('_')[0]}_Labels.csv")
        img_dir = os.path.join(rfmid_root, split)

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_id = str(row['ID'])
            # Find image file (could be .png or .jpg)
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(img_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Store relative path for Modal cloud compatibility
            rel_path = img_path.replace("data/raw/", "")

            # Map RFMiD diseases to our targets
            # DR -> DR, MYA -> Myopia
            # ODC (Optic Disc Cupping), ODP (Optic Disc Pallor), ODE (Optic Disc Edema) -> Glaucoma
            # No explicit Cataract in RFMiD
            records.append({
                'image_path': rel_path,
                'Cataract': 0,
                'DR': int(row.get('DR', 0)),
                'Glaucoma': int(any(row.get(c, 0) for c in ['ODC', 'ODP', 'ODE'])),
                'Myopia': int(row.get('MYA', 0)),
                'source': f'RFMiD_{split.split("_")[0]}',
            })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} images")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. PALM
# ═══════════════════════════════════════════════════════════════════════════

def load_palm():
    """Load PALM pathologic myopia dataset."""
    print("Loading PALM...")
    palm_img_dir = "data/raw/PALM/Training/Classification"

    image_files = glob.glob(os.path.join(palm_img_dir, "*.jpg")) + \
                  glob.glob(os.path.join(palm_img_dir, "*.JPG")) + \
                  glob.glob(os.path.join(palm_img_dir, "*.png"))

    records = []
    for img_path in image_files:
        # Store relative path for Modal cloud compatibility
        rel_path = img_path.replace("data/raw/", "")
        records.append({
            'image_path': rel_path,
            'Cataract': 0,
            'DR': 0,
            'Glaucoma': 0,
            'Myopia': 1,
            'source': 'PALM',
        })

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} images")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Building Unified Fundus Dataset")
    print("=" * 60)
    print()

    # Load all datasets
    datasets = []

    try:
        datasets.append(load_odir())
    except Exception as e:
        print(f"  Error loading ODIR: {e}")

    try:
        datasets.append(load_jsiec())
    except Exception as e:
        print(f"  Error loading JSIEC: {e}")

    try:
        datasets.append(load_rfmid())
    except Exception as e:
        print(f"  Error loading RFMiD: {e}")

    try:
        datasets.append(load_palm())
    except Exception as e:
        print(f"  Error loading PALM: {e}")

    # Combine
    combined = pd.concat(datasets, ignore_index=True)

    # Remove duplicates by image path
    before_dedup = len(combined)
    combined.drop_duplicates(subset='image_path', inplace=True)
    after_dedup = len(combined)
    print(f"\nRemoved {before_dedup - after_dedup} duplicate images")

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(combined)}")
    print()
    print("Per-source counts:")
    print(combined['source'].value_counts())
    print()
    print("Per-disease positive counts:")
    for disease in ['Cataract', 'DR', 'Glaucoma', 'Myopia']:
        pos = combined[disease].sum()
        pct = pos / len(combined) * 100
        print(f"  {disease:12s}: {pos:5d} ({pct:5.1f}%)")
    print()
    print("Multi-label distribution:")
    combined['n_diseases'] = combined[['Cataract', 'DR', 'Glaucoma', 'Myopia']].sum(axis=1)
    print(combined['n_diseases'].value_counts().sort_index())
    print()
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
