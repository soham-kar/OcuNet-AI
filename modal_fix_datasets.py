# modal_fix_datasets.py
"""
Fix the unified dataset CSVs on the Modal 'cataract-data' volume.

Fixes applied:
  1. FIX SYNTHETIC PATHS: The existing CSVs store synthetic paths as
     'data/synthetic/cataract/v4/...' but the training script resolves them
     relative to /data/raw/, producing '/data/raw/data/synthetic/...' which
     does NOT exist. This silently drops ALL synthetic images from training.
     We rewrite them to 'synthetic/cataract/v4/...' so they resolve correctly.

  2. ADD IDRiD (DR): IDRiD has DR grading labels (0-4). 413 train + 103 test
     images. We map grade>=1 -> DR=1. Train set -> train_v4, test set -> test_v4.

  3. ADD PAPILA (Glaucoma): PAPILA has Glaucoma diagnosis (0=normal, 1=glaucoma,
     2=suspect). 488 images (244 OD + 244 OS). We map diagnosis>=1 -> Glaucoma=1.
     Split into train/val/test.

  SKIPPED: REFUGE2 is a byte-for-byte duplicate of REFUGE (already in the CSVs).
  Adding it would cause train/test leakage, so we leave it out.

Usage:
    modal run modal_fix_datasets.py
"""
import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "pandas",
    "openpyxl",
    "numpy",
])

app = modal.App("fix-datasets", image=image)
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=False)


@app.function(
    volumes={"/data": data_volume},
    timeout=1800,
    memory=8192,
    image=image,
)
def fix_datasets():
    import os
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    DATA_ROOT = Path("/data")
    RAW_ROOT = DATA_ROOT / "raw"

    DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

    def resolve_path(p):
        """Mirror the training script's resolve_path logic."""
        if os.path.isabs(p):
            return p
        full = RAW_ROOT / p
        if full.exists():
            return str(full)
        full2 = DATA_ROOT / p
        if full2.exists():
            return str(full2)
        return str(full)

    print("=" * 70)
    print("  STEP 1: FIX SYNTHETIC PATHS IN EXISTING CSVs")
    print("=" * 70)

    csv_files = ["train_v4.csv", "val_tune_v4.csv", "test_v4.csv"]
    synthetic_fixed = 0
    for csv_name in csv_files:
        csv_path = DATA_ROOT / csv_name
        if not csv_path.exists():
            print(f"  ⚠️  {csv_name} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        # Synthetic paths start with 'data/synthetic/...' but should be 'synthetic/...'
        mask = df['image_path'].str.startswith('data/synthetic/')
        n = mask.sum()
        if n > 0:
            df.loc[mask, 'image_path'] = df.loc[mask, 'image_path'].str.replace(
                '^data/synthetic/', 'synthetic/', regex=True)
            synthetic_fixed += n
            print(f"  ✅ {csv_name}: fixed {n} synthetic paths")
        else:
            print(f"  {csv_name}: no synthetic paths to fix")
        df.to_csv(csv_path, index=False)

    print(f"\n  Total synthetic paths fixed: {synthetic_fixed}")

    # Verify the fix
    test_df = pd.read_csv(DATA_ROOT / "train_v4.csv")
    syn_rows = test_df[test_df['image_path'].str.startswith('synthetic/')]
    if len(syn_rows) > 0:
        sample = syn_rows['image_path'].iloc[0]
        resolved = resolve_path(sample)
        print(f"  ✅ Verify: '{sample}' -> exists={Path(resolved).exists()}")

    print("\n" + "=" * 70)
    print("  STEP 2: ADD IDRiD (DR)")
    print("=" * 70)

    idrid_train_csv = RAW_ROOT / "IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_test_csv = RAW_ROOT / "IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    idrid_train_img = RAW_ROOT / "IDRiD/B. Disease Grading/1. Original Images/a. Training Set"
    idrid_test_img = RAW_ROOT / "IDRiD/B. Disease Grading/1. Original Images/b. Testing Set"

    def parse_idrid(label_csv, img_dir, split_name):
        labels = pd.read_csv(label_csv)
        rows = []
        for _, r in labels.iterrows():
            name = str(r['Image name']).strip()
            grade = int(r['Retinopathy grade'])
            img_path = img_dir / f"{name}.jpg"
            if not img_path.exists():
                continue
            # Paths are stored relative to /data/raw/ (matching existing CSV convention)
            rel = img_path.relative_to(RAW_ROOT)
            rows.append({
                'image_path': str(rel),
                'Cataract': 0,
                'DR': 1 if grade >= 1 else 0,
                'Glaucoma': 0,
                'Myopia': 0,
                'source': f'IDRiD_{split_name}',
            })
        return pd.DataFrame(rows)

    idrid_train = parse_idrid(idrid_train_csv, idrid_train_img, "train")
    idrid_test = parse_idrid(idrid_test_csv, idrid_test_img, "test")
    print(f"  IDRiD train: {len(idrid_train)} | test: {len(idrid_test)}")
    print(f"  IDRiD train DR positives: {idrid_train['DR'].sum()}")

    print("\n" + "=" * 70)
    print("  STEP 3: ADD PAPILA (Glaucoma)")
    print("=" * 70)

    papila_od = RAW_ROOT / "PAPILA/ClinicalData/patient_data_od.xlsx"
    papila_os = RAW_ROOT / "PAPILA/ClinicalData/patient_data_os.xlsx"
    papila_img = RAW_ROOT / "PAPILA/FundusImages"

    def parse_papila(xlsx_path, eye_suffix, split_name):
        raw = pd.read_excel(xlsx_path, header=None)
        rows = []
        for _, r in raw.iloc[3:].iterrows():
            pid = str(r[0]).strip().replace('#', '')
            if not pid:
                continue
            diag = int(r[3])
            img_path = papila_img / f"RET{pid}{eye_suffix}.jpg"
            if not img_path.exists():
                continue
            # Paths are stored relative to /data/raw/ (matching existing CSV convention)
            rel = img_path.relative_to(RAW_ROOT)
            rows.append({
                'image_path': str(rel),
                'Cataract': 0,
                'DR': 0,
                'Glaucoma': 1 if diag >= 1 else 0,
                'Myopia': 0,
                'source': f'PAPILA_{split_name}',
            })
        return pd.DataFrame(rows)

    papila_od_df = parse_papila(papila_od, "OD", "train")
    papila_os_df = parse_papila(papila_os, "OS", "train")
    papila_all = pd.concat([papila_od_df, papila_os_df], ignore_index=True)
    print(f"  PAPILA total: {len(papila_all)} | Glaucoma positives: {papila_all['Glaucoma'].sum()}")

    # Split PAPILA into train/val/test (70/15/15)
    rng = np.random.RandomState(42)
    papila_all = papila_all.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    n = len(papila_all)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    papila_train = papila_all.iloc[:n_train].copy()
    papila_val = papila_all.iloc[n_train:n_train + n_val].copy()
    papila_test = papila_all.iloc[n_train + n_val:].copy()
    papila_train['source'] = 'PAPILA_train'
    papila_val['source'] = 'PAPILA_val'
    papila_test['source'] = 'PAPILA_test'
    print(f"  PAPILA split: train={len(papila_train)} val={len(papila_val)} test={len(papila_test)}")

    print("\n" + "=" * 70)
    print("  STEP 5: ADD ODIR-5K (real cataract + DR + glaucoma + myopia)")
    print("=" * 70)

    # ODIR-5K full_df.csv has per-eye labels. Columns:
    # ID,Age,Sex,Left-Fundus,Right-Fundus,Left-Diag,Right-Diag,N,D,G,C,A,H,M,O,filepath,labels,target,filename
    # N=normal, D=DR, G=glaucoma, C=cataract, A=age-related, H=hypertension, M=myopia, O=other
    odir_full = RAW_ROOT / "odir" / "full_df.csv"
    odir_img_base = RAW_ROOT / "odir" / "ODIR-5K" / "ODIR-5K"
    odir_rows = []
    if odir_full.exists():
        odir_df = pd.read_csv(odir_full)
        for _, r in odir_df.iterrows():
            fn = str(r['filename']).strip()
            # Locate the image in Training or Testing Images
            img_path = odir_img_base / "Training Images" / fn
            if not img_path.exists():
                img_path = odir_img_base / "Testing Images" / fn
            if not img_path.exists():
                continue
            rel = img_path.relative_to(RAW_ROOT)
            odir_rows.append({
                'image_path': str(rel),
                'Cataract': 1 if str(r['C']) == '1' else 0,
                'DR': 1 if str(r['D']) == '1' else 0,
                'Glaucoma': 1 if str(r['G']) == '1' else 0,
                'Myopia': 1 if str(r['M']) == '1' else 0,
                'source': 'ODIR5K',
            })
    odir5k = pd.DataFrame(odir_rows)
    print(f"  ODIR-5K parsed: {len(odir5k)} images")
    print(f"    Cataract={odir5k['Cataract'].sum()} DR={odir5k['DR'].sum()} "
          f"Glaucoma={odir5k['Glaucoma'].sum()} Myopia={odir5k['Myopia'].sum()}")

    print("\n" + "=" * 70)
    print("  STEP 6: ADD LAG (glaucoma)")
    print("=" * 70)

    def parse_binary_folders(base_dir, source_name, glau_folders, norm_folders):
        """Parse a dataset with glaucoma/normal subfolders (optionally nested under 'image')."""
        rows = []
        for folder, is_glau in list(glau_folders) + list(norm_folders):
            # Some datasets nest images under an 'image' subfolder
            candidates = [folder]
            if (folder / "image").exists():
                candidates = [folder / "image"]
            for cand in candidates:
                for img in cand.iterdir():
                    if img.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                        continue
                    rel = img.relative_to(RAW_ROOT)
                    rows.append({
                        'image_path': str(rel),
                        'Cataract': 0, 'DR': 0,
                        'Glaucoma': 1 if is_glau else 0,
                        'Myopia': 0,
                        'source': source_name,
                    })
        return pd.DataFrame(rows)

    lag_base = RAW_ROOT / "LAG"
    lag_train = parse_binary_folders(
        lag_base, "LAG_train",
        [(lag_base / "train" / "glaucoma", True)],
        [(lag_base / "train" / "non_glaucoma", False)])
    lag_test = parse_binary_folders(
        lag_base, "LAG_test",
        [(lag_base / "test" / "glaucoma", True)],
        [(lag_base / "test" / "non_glaucoma", False)])
    print(f"  LAG train: {len(lag_train)} (glau={lag_train['Glaucoma'].sum()}) | "
          f"test: {len(lag_test)} (glau={lag_test['Glaucoma'].sum()})")

    print("\n" + "=" * 70)
    print("  STEP 7: ADD ACRIMA (glaucoma)")
    print("=" * 70)

    acrima_base = RAW_ROOT / "ACRIMA"
    acrima_train = parse_binary_folders(
        acrima_base, "ACRIMA_train",
        [(acrima_base / "train" / "Glaucoma", True)],
        [(acrima_base / "train" / "Non Glaucoma", False)])
    acrima_test = parse_binary_folders(
        acrima_base, "ACRIMA_test",
        [(acrima_base / "test" / "Glaucoma", True)],
        [(acrima_base / "test" / "Non Glaucoma", False)])
    print(f"  ACRIMA train: {len(acrima_train)} (glau={acrima_train['Glaucoma'].sum()}) | "
          f"test: {len(acrima_test)} (glau={acrima_test['Glaucoma'].sum()})")

    print("\n" + "=" * 70)
    print("  STEP 8: ADD RIM-ONE (glaucoma)")
    print("=" * 70)

    rimone_base = RAW_ROOT / "RIM-ONE_DL_images" / "partitioned_randomly"
    rimone_train = parse_binary_folders(
        rimone_base, "RIMONE_train",
        [(rimone_base / "training_set" / "glaucoma", True)],
        [(rimone_base / "training_set" / "normal", False)])
    rimone_test = parse_binary_folders(
        rimone_base, "RIMONE_test",
        [(rimone_base / "test_set" / "glaucoma", True)],
        [(rimone_base / "test_set" / "normal", False)])
    print(f"  RIM-ONE train: {len(rimone_train)} (glau={rimone_train['Glaucoma'].sum()}) | "
          f"test: {len(rimone_test)} (glau={rimone_test['Glaucoma'].sum()})")

    print("\n" + "=" * 70)
    print("  STEP 9: ADD UNTAPPED G1020 (glaucoma)")
    print("=" * 70)

    # G1020.csv has imageID (image_0.jpg) + binaryLabels. The training CSV already
    # uses some G1020 images; we add the unused ones. Labels map directly by imageID.
    g1020_csv = RAW_ROOT / "other_dataset" / "G1020" / "G1020.csv"
    g1020_img = RAW_ROOT / "other_dataset" / "G1020" / "Images"
    # Load existing G1020 image paths already in the CSVs to avoid duplicates
    existing_g1020 = set()
    for csv_name in ["train_v4.csv", "val_tune_v4.csv", "test_v4.csv"]:
        p = DATA_ROOT / csv_name
        if p.exists():
            tmp = pd.read_csv(p)
            existing_g1020.update(tmp[tmp['source'] == 'G1020']['image_path'].tolist())
    g1020_rows = []
    if g1020_csv.exists():
        g1020_df = pd.read_csv(g1020_csv)
        for _, r in g1020_df.iterrows():
            img_id = str(r['imageID']).strip()
            img_path = g1020_img / img_id
            if not img_path.exists():
                continue
            rel = img_path.relative_to(RAW_ROOT)
            if str(rel) in existing_g1020:
                continue  # already used
            g1020_rows.append({
                'image_path': str(rel),
                'Cataract': 0, 'DR': 0,
                'Glaucoma': 1 if int(r['binaryLabels']) == 1 else 0,
                'Myopia': 0,
                'source': 'G1020',
            })
    g1020_new = pd.DataFrame(g1020_rows)
    if len(g1020_new) > 0:
        print(f"  New G1020 images to add: {len(g1020_new)} (glau={g1020_new['Glaucoma'].sum()})")
    else:
        print(f"  New G1020 images to add: 0 (all G1020 images already in CSVs)")

    print("\n" + "=" * 70)
    print("  STEP 4: MERGE INTO EXISTING CSVs")
    print("=" * 70)

    # Load existing CSVs
    train_df = pd.read_csv(DATA_ROOT / "train_v4.csv")
    val_df = pd.read_csv(DATA_ROOT / "val_tune_v4.csv")
    test_df = pd.read_csv(DATA_ROOT / "test_v4.csv")

    # Add IDRiD: train -> train, test -> test
    train_df = pd.concat([train_df, idrid_train], ignore_index=True)
    test_df = pd.concat([test_df, idrid_test], ignore_index=True)

    # Add PAPILA: split into train/val/test
    train_df = pd.concat([train_df, papila_train], ignore_index=True)
    val_df = pd.concat([val_df, papila_val], ignore_index=True)
    test_df = pd.concat([test_df, papila_test], ignore_index=True)

    # Add ODIR-5K: split into train/val/test (80/10/10)
    if len(odir5k) > 0:
        rng5 = np.random.RandomState(7)
        odir5k = odir5k.sample(frac=1.0, random_state=rng5).reset_index(drop=True)
        n5 = len(odir5k)
        n5_tr = int(n5 * 0.80)
        n5_va = int(n5 * 0.10)
        train_df = pd.concat([train_df, odir5k.iloc[:n5_tr]], ignore_index=True)
        val_df = pd.concat([val_df, odir5k.iloc[n5_tr:n5_tr + n5_va]], ignore_index=True)
        test_df = pd.concat([test_df, odir5k.iloc[n5_tr + n5_va:]], ignore_index=True)

    # Add LAG: train -> train, test -> test
    train_df = pd.concat([train_df, lag_train], ignore_index=True)
    test_df = pd.concat([test_df, lag_test], ignore_index=True)

    # Add ACRIMA: train -> train, test -> test
    train_df = pd.concat([train_df, acrima_train], ignore_index=True)
    test_df = pd.concat([test_df, acrima_test], ignore_index=True)

    # Add RIM-ONE: train -> train, test -> test
    train_df = pd.concat([train_df, rimone_train], ignore_index=True)
    test_df = pd.concat([test_df, rimone_test], ignore_index=True)

    # Add untapped G1020: all to train
    if len(g1020_new) > 0:
        train_df = pd.concat([train_df, g1020_new], ignore_index=True)

    # Deduplicate by image_path (keep first)
    def dedup(df):
        before = len(df)
        df = df.drop_duplicates(subset='image_path', keep='first').reset_index(drop=True)
        return df, before

    train_df, bt = dedup(train_df)
    val_df, bv = dedup(val_df)
    test_df, bt2 = dedup(test_df)

    print(f"  train_v4: {bt} -> {len(train_df)}")
    print(f"  val_tune_v4: {bv} -> {len(val_df)}")
    print(f"  test_v4: {bt2} -> {len(test_df)}")

    # Verify all image paths resolve
    print("\n  Verifying all image paths resolve...")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = 0
        for p in df['image_path']:
            if not Path(resolve_path(p)).exists():
                missing += 1
        print(f"  {name}: {len(df)} rows, {missing} missing images")

    # Save back
    train_df.to_csv(DATA_ROOT / "train_v4.csv", index=False)
    val_df.to_csv(DATA_ROOT / "val_tune_v4.csv", index=False)
    test_df.to_csv(DATA_ROOT / "test_v4.csv", index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  {name}: {len(df)} rows")
        print(f"    Sources: {df['source'].value_counts().to_dict()}")
        print(f"    Disease sums: {df[DISEASE_NAMES].sum().to_dict()}")

    data_volume.commit()
    print("\n  ✅ Volume committed!")
    # Return a plain JSON string to avoid numpy pickling issues on the local client
    return json.dumps({
        "synthetic_fixed": int(synthetic_fixed),
        "idrid_train": int(len(idrid_train)),
        "idrid_test": int(len(idrid_test)),
        "papila_train": int(len(papila_train)),
        "papila_val": int(len(papila_val)),
        "papila_test": int(len(papila_test)),
        "odir5k": int(len(odir5k)),
        "lag_train": int(len(lag_train)),
        "lag_test": int(len(lag_test)),
        "acrima_train": int(len(acrima_train)),
        "acrima_test": int(len(acrima_test)),
        "rimone_train": int(len(rimone_train)),
        "rimone_test": int(len(rimone_test)),
        "g1020_new": int(len(g1020_new)),
        "final_train": int(len(train_df)),
        "final_val": int(len(val_df)),
        "final_test": int(len(test_df)),
    })


@app.local_entrypoint()
def main():
    import json as _json
    result = _json.loads(fix_datasets.remote())
    print("\n" + "=" * 70)
    print("  FIX COMPLETE")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k}: {v}")
