"""
Upload unified dataset to Modal volumes for cloud training.

Usage:
    modal run upload_data_to_modal.py
"""

import modal
import os
from pathlib import Path

app = modal.App("upload-unified-data")

data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume}, timeout=3600)
def upload_files():
    """Verify data is present in Modal volume."""
    import pandas as pd

    train_csv = Path("/data/train_combined.csv")
    val_csv = Path("/data/val_combined.csv")

    print("Checking Modal volume contents...")
    print(f"Train CSV exists: {train_csv.exists()}")
    print(f"Val CSV exists: {val_csv.exists()}")

    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"Train rows: {len(df)}")
        print(f"Train columns: {list(df.columns)}")

    if val_csv.exists():
        df = pd.read_csv(val_csv)
        print(f"Val rows: {len(df)}")

    # Check image directories
    for subdir in ["odir", "RFMiD", "JSIEC", "PALM"]:
        path = Path(f"/data/{subdir}")
        if path.exists():
            print(f"{subdir}: {len(list(path.glob('**/*.jpg')))} images")


@app.local_entrypoint()
def main():
    print("To upload data to Modal volume, run:")
    print("  modal volume put cataract-data data/train_combined.csv /train_combined.csv")
    print("  modal volume put cataract-data data/val_combined.csv /val_combined.csv")
    print("  modal volume put cataract-data data/raw/odir /odir")
    print("  modal volume put cataract-data data/raw/RFMiD /RFMiD")
    print("  modal volume put cataract-data data/raw/JSIEC /JSIEC")
    print("  modal volume put cataract-data data/raw/PALM /PALM")
    print()
    print("Then verify with:")
    print("  modal run upload_data_to_modal.py")
    print()
    print("Note: CSV paths are now relative to /data (e.g., 'odir/preprocessed_images/xxx.jpg')")
