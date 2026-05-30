"""
Download EyePACS dataset to Modal Volume
Run: modal run scripts/download_eyepacs_modal.py
"""

import modal
import subprocess
import os
from pathlib import Path

# Modal setup
image = modal.Image.debian_slim().pip_install(["kaggle", "tqdm", "pandas"])
app = modal.App("download-eyepacs", image=image)

# Create persistent volume
eyepacs_volume = modal.Volume.from_name("eyepacs-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": eyepacs_volume},
    secrets=[modal.Secret.from_name("kaggle-creds")],
    timeout=7200,  # 2 hours for download + extraction
    cpu=2,
    memory=4096,
)
def download_eyepacs():
    """Download and extract EyePACS to Modal Volume"""
    import pandas as pd
    from tqdm import tqdm
    import zipfile
    
    # Set Kaggle credentials from Modal secret
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
    
    print("🔑 Kaggle credentials set from Modal secret")
    
    # Download directory
    download_dir = Path("/data/eyepacs")
    download_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(download_dir)
    
    # Download dataset
    print("\n📥 Downloading EyePACS dataset (~35GB)...")
    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "diabetic-retinopathy-detection",
            "--force"
        ], check=True)
        print("✅ Download complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Extract training set
    train_zip = download_dir / "train.zip"
    if train_zip.exists():
        print("\n📦 Extracting train.zip...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Extracting", unit="file"):
                zip_ref.extract(file, download_dir / "train")
        print("✅ Extracted training images")
    
    # Extract labels
    labels_zip = download_dir / "trainLabels.csv.zip"
    if labels_zip.exists():
        print("\n📄 Extracting trainLabels.csv...")
        with zipfile.ZipFile(labels_zip, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("✅ Extracted labels")
    
    # Clean up zip files
    print("\n🧹 Cleaning up zip files...")
    for zip_file in download_dir.glob("*.zip"):
        zip_file.unlink()
        print(f"   Removed {zip_file.name}")
    
    # Verify
    train_dir = download_dir / "train"
    if train_dir.exists():
        n_images = len(list(train_dir.glob("*.jpeg")))
        print(f"\n📊 Verification:")
        print(f"   Training images: {n_images}")
        print(f"   Expected: ~35,126")
    
    # Show label distribution
    label_csv = download_dir / "trainLabels.csv"
    if label_csv.exists():
        df = pd.read_csv(label_csv)
        print(f"\n📈 Severity Distribution:")
        print(df['level'].value_counts().sort_index())
        print(f"\n   DR-positive (level>0): {len(df[df['level'] > 0])}")
    
    # Commit volume
    print("\n💾 Committing volume...")
    eyepacs_volume.commit()
    
    print("\n" + "="*70)
    print("🎉 EyePACS dataset ready at: /data/eyepacs/")
    print("="*70)
    
    return True

@app.local_entrypoint()
def main():
    """Launch download on Modal"""
    success = download_eyepacs.remote()
    if success:
        print("\n✅ Download completed successfully")
    else:
        print("\n❌ Download failed")
