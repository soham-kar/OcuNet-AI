"""
Fixed EyePACS download script that explicitly writes kaggle.json
Run: modal run scripts/download_eyepacs_modal_v2.py
"""

import modal
import subprocess
import os
import json
from pathlib import Path

image = modal.Image.debian_slim().pip_install(["kaggle", "pandas", "tqdm", "requests"])
app = modal.App("download-eyepacs-v2", image=image)

eyepacs_volume = modal.Volume.from_name("eyepacs-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": eyepacs_volume},
    secrets=[modal.Secret.from_name("kaggle-creds")],
    timeout=7200,
    cpu=2,
    memory=4096,
)
def download_eyepacs():
    """Download with explicit kaggle.json creation"""
    import pandas as pd
    from tqdm import tqdm
    import zipfile
    
    # FIX 1: Explicitly create kaggle.json from environment
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        print("❌ ERROR: KAGGLE_USERNAME or KAGGLE_KEY not found!")
        print(f"   Available env vars: {[k for k in os.environ.keys() if 'KAGGLE' in k]}")
        return False
    
    # Write kaggle.json explicitly
    kaggle_json = kaggle_dir / "kaggle.json"
    with open(kaggle_json, 'w') as f:
        json.dump({"username": username, "key": key}, f)
    kaggle_json.chmod(0o600)
    
    print(f"✅ Created {kaggle_json} for user: {username}")
    
    # Verify credentials
    print("\n🔍 Verifying Kaggle API access...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ Kaggle API authentication successful")
        else:
            print(f"❌ Kaggle API failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False
    
    # Download directory
    download_dir = Path("/data/eyepacs")
    download_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(download_dir)
    
    # Check if already downloaded
    train_dir = download_dir / "train"
    if train_dir.exists() and len(list(train_dir.glob("*.jpeg"))) > 30000:
        print("✅ EyePACS already downloaded. Skipping.")
        return True
    
    # Download with error handling
    print("\n📥 Downloading EyePACS dataset (~35GB)...")
    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "diabetic-retinopathy-detection", "--force"
        ], check=True, timeout=3600)
        print("✅ Download complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("\n🔍 Have you accepted competition rules?")
        print("   Visit: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules")
        return False
    
    # Extract training images
    train_zip = download_dir / "train.zip"
    if train_zip.exists():
        print(f"\n📦 Extracting train.zip...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting"):
                try:
                    zip_ref.extract(file, download_dir / "train")
                except:
                    continue
        print("✅ Extracted training images")
    
    # Extract labels
    labels_zip = download_dir / "trainLabels.csv.zip"
    if labels_zip.exists():
        with zipfile.ZipFile(labels_zip, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("✅ Extracted labels")
    
    # Verify
    if train_dir.exists():
        n_images = len(list(train_dir.glob("*.jpeg")))
        print(f"\n📊 Found {n_images:,} images")
    
    # Show distribution
    label_csv = download_dir / "trainLabels.csv"
    if label_csv.exists():
        df = pd.read_csv(label_csv)
        print(f"\n📈 Severity Distribution:")
        print(df['level'].value_counts().sort_index())
    
    # Clean up
    for zip_file in download_dir.glob("*.zip"):
        zip_file.unlink()
    
    # Commit volume
    print("\n💾 Committing volume...")
    eyepacs_volume.commit()
    
    print("\n🎉 EyePACS dataset ready at: /data/eyepacs/")
    return True

@app.local_entrypoint()
def main():
    success = download_eyepacs.remote()
    if success:
        print("\n✅ Download completed successfully")
    else:
        print("\n❌ Download failed")
