# modal_inspect_structure.py
"""Inspect the directory structure of Modal volumes (top-level only)."""
import modal

image = modal.Image.debian_slim(python_version="3.10").pip_install(["pandas"])

app = modal.App("inspect-structure", image=image)
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=False)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=False)


@app.function(
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=300,
    image=image,
)
def inspect():
    import os
    from pathlib import Path
    from collections import defaultdict

    def show_tree(path, max_depth=2, prefix=""):
        try:
            entries = sorted(Path(path).iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for entry in entries:
                if entry.is_dir():
                    n_files = sum(1 for _ in entry.rglob("*") if _.is_file())
                    size_mb = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file()) / (1024*1024)
                    print(f"{prefix}📁 {entry.name}/  ({n_files} files, {size_mb:.1f} MB)")
                    if max_depth > 0:
                        show_tree(entry, max_depth - 1, prefix + "  ")
                else:
                    size_mb = entry.stat().st_size / (1024*1024)
                    if size_mb > 0.01:
                        print(f"{prefix}📄 {entry.name}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"{prefix}❌ {e}")

    print("=" * 70)
    print("  /data (cataract-data volume)")
    print("=" * 70)
    show_tree("/data", max_depth=2)

    print("\n" + "=" * 70)
    print("  /checkpoints (cataract-checkpoints volume)")
    print("=" * 70)
    show_tree("/checkpoints", max_depth=2)

    # Check the CSV files for column names and row counts
    print("\n" + "=" * 70)
    print("  CSV FILE DETAILS")
    print("=" * 70)
    import pandas as pd
    csv_files = [
        "/data/train_v4.csv", "/data/val_tune_v4.csv", "/data/test_v4.csv",
        "/data/test_v4_extended.csv", "/data/combined_fundus_dataset.csv",
        "/data/train_combined.csv", "/data/val_combined.csv",
        "/data/raw/odir/full_df.csv",
        "/data/raw/RFMiD/Training_set/RFMiD_Training_Labels.csv",
    ]
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            print(f"\n  📊 {csv_path}")
            print(f"     Rows: {len(df)} | Columns: {list(df.columns)}")
            if 'image_path' in df.columns:
                # Show first path to understand structure
                print(f"     First path: {df['image_path'].iloc[0]}")
            # Show disease columns if present
            disease_cols = [c for c in df.columns if c in ['Cataract', 'DR', 'Glaucoma', 'Myopia', 'C', 'D', 'G', 'M']]
            if disease_cols:
                print(f"     Disease sums: {df[disease_cols].sum().to_dict()}")
        except Exception as e:
            print(f"\n  ❌ {csv_path}: {e}")

    # Check image directories
    print("\n" + "=" * 70)
    print("  IMAGE DIRECTORIES")
    print("=" * 70)
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for d in Path("/data").rglob("*"):
        if d.is_dir():
            imgs = [p for p in d.iterdir() if p.suffix.lower() in img_exts]
            if len(imgs) > 50:
                print(f"  {d.relative_to('/data')}: {len(imgs)} images (direct children)")

    return "done"


@app.local_entrypoint()
def main():
    inspect.remote()