# Quick verification script
import pandas as pd
from pathlib import Path

# ODIR Dataset
odir_csv = Path('data/raw/odir/full_df.csv')
if odir_csv.exists():
    df = pd.read_csv(odir_csv)
    print("=" * 50)
    print("ODIR-5K Dataset Summary")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Cataract images (C=1): {df['C'].sum()}")
    print(f"Normal images (N=1): {df['N'].sum()}")
    print(f"Columns: {list(df.columns)}")
else:
    print(f"ODIR CSV not found at: {odir_csv}")

# Check image folder
img_folders = [
    'data/raw/odir/ODIR-5K/ODIR-5K/Training Images',
    'data/raw/odir/Training Images',
    'data/raw/odir/preprocessed_images'
]

print("\nImage folders:")
for folder in img_folders:
    p = Path(folder)
    if p.exists():
        count = len(list(p.glob('*.jpg')))
        print(f"  {folder}: {count} images")

# Mendeley Dataset  
print("\n" + "=" * 50)
print("Mendeley Slit-Lamp Dataset Summary")
print("=" * 50)

mendeley_base = Path('data/raw/slitlamp')
if mendeley_base.exists():
    # Find the dataset folder
    for folder in mendeley_base.rglob('*'):
        if folder.is_dir() and folder.name.isdigit():
            patient_folders = [f for f in mendeley_base.rglob('*') if f.is_dir() and f.name.isdigit()]
            print(f"Patient folders found: {len(set([f.name for f in patient_folders]))}")
            break
    
    # Count total images
    jpg_files = list(mendeley_base.rglob('*.jpg')) + list(mendeley_base.rglob('*.JPG'))
    print(f"Total slit-lamp images: {len(jpg_files)}")
else:
    print(f"Mendeley folder not found at: {mendeley_base}")
