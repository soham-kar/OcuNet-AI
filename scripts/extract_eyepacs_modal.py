"""
Extract EyePACS multi-part archive on Modal
Files expected at: /data/eyepacs/train.zip.001-005, trainLabels.csv.zip

Run AFTER upload:
    modal run scripts/extract_eyepacs_modal.py
"""

import modal
import subprocess
from pathlib import Path

image = modal.Image.debian_slim().apt_install(["p7zip-full"]).pip_install(["pandas", "tqdm"])
app = modal.App("extract-eyepacs", image=image)

eyepacs_volume = modal.Volume.from_name("eyepacs-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": eyepacs_volume},
    timeout=21600,  # 6 hours
    cpu=4,
    memory=16384,
)
def extract_eyepacs():
    """Extract multi-part train archive"""
    import pandas as pd
    import zipfile
    
    data_dir = Path("/data/eyepacs")
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for files
    train_part1 = data_dir / "train.zip.001"
    labels_zip = data_dir / "trainLabels.csv.zip"
    
    if not train_part1.exists():
        print("❌ train.zip.001 not found at /data/eyepacs/")
        print("   Upload with:")
        print("   modal volume put eyepacs-data train.zip.001 ... /data/eyepacs/")
        return False
    
    print("📦 Found files:")
    for f in sorted(data_dir.glob("*.zip*")):
        print(f"   {f.name}: {f.stat().st_size / 1e9:.2f} GB")
    
    # Step 1: Extract labels
    if labels_zip.exists():
        print("\n📄 Extracting labels...")
        with zipfile.ZipFile(labels_zip, 'r') as zf:
            zf.extractall(data_dir)
        print("✅ Labels extracted")
    
    # Step 2: Extract multi-part train archive with 7z
    print("\n📦 Extracting train images (this takes a while)...")
    result = subprocess.run(
        ["7z", "x", str(train_part1), f"-o{data_dir}", "-y"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"⚠️ Extraction warning: {result.stderr[:500]}")
    
    # Step 3: Verify
    print("\n� Verification:")
    
    # Check both possible locations
    for check_dir in [train_dir, data_dir / "train"]:
        if check_dir.exists():
            n_images = len(list(check_dir.glob("*.jpeg")))
            if n_images > 0:
                print(f"   Training images: {n_images:,}")
                break
    
    # Check labels
    label_file = data_dir / "trainLabels.csv"
    if label_file.exists():
        df = pd.read_csv(label_file)
        print(f"\n📈 DR Distribution (level 0-4):")
        print(df['level'].value_counts().sort_index())
        print(f"\n   Total: {len(df):,}")
        print(f"   DR-positive: {len(df[df['level'] > 0]):,}")
    
    # Step 4: Clean up zip files
    print("\n🧹 Cleaning up zip parts...")
    for f in data_dir.glob("train.zip.*"):
        f.unlink()
        print(f"   Removed {f.name}")
    if labels_zip.exists():
        labels_zip.unlink()
    
    # Commit
    print("\n💾 Committing volume...")
    eyepacs_volume.commit()
    
    print("\n" + "="*60)
    print("🎉 EyePACS ready at /data/eyepacs/train/")
    print("="*60)
    return True

@app.local_entrypoint()
def main():
    success = extract_eyepacs.remote()
    print("✅ Done!" if success else "❌ Failed")
