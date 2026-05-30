"""
EyePACS Download v3 - Uses kaggle.json file-based secret
Run: modal run scripts/download_eyepacs_modal_v3.py

Setup:
1. Download new kaggle.json from kaggle.com/settings → API → Create New Token
2. Create Modal secret from file:
   modal secret create kaggle-json-file --file C:\Users\pc\Downloads\kaggle.json
3. Run this script:
   modal run scripts/download_eyepacs_modal_v3.py
"""

import modal
import subprocess
import os
import json
from pathlib import Path

image = modal.Image.debian_slim().pip_install(["kaggle", "pandas", "tqdm"])
app = modal.App("download-eyepacs-v3", image=image)

eyepacs_volume = modal.Volume.from_name("eyepacs-data", create_if_missing=True)

# Try both secret formats
try:
    secrets = [modal.Secret.from_name("kaggle-json-file")]
except:
    secrets = [modal.Secret.from_name("kaggle-creds")]

@app.function(
    image=image,
    volumes={"/data": eyepacs_volume},
    secrets=secrets,
    timeout=7200,
    cpu=2,
    memory=4096,
)
def download_eyepacs():
    """Download with robust credential handling"""
    import pandas as pd
    from tqdm import tqdm
    import zipfile
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Method 1: Try to read from environment (file-based secret)
    kaggle_content = os.getenv('KAGGLE_CONFIG_DIR_CONTENT')
    if kaggle_content:
        print("📁 Using file-based Kaggle credentials")
        with open(kaggle_json, 'w') as f:
            f.write(kaggle_content)
    else:
        # Method 2: Try individual env vars
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if username and key:
            print(f"🔑 Using env-based credentials for user: {username}")
            with open(kaggle_json, 'w') as f:
                json.dump({"username": username, "key": key}, f)
        else:
            # Method 3: Check if kaggle.json exists in common locations
            for path in ["/run/secrets/kaggle.json", "/secrets/kaggle.json"]:
                if Path(path).exists():
                    print(f"📁 Found kaggle.json at {path}")
                    import shutil
                    shutil.copy(path, kaggle_json)
                    break
            else:
                print("❌ No Kaggle credentials found!")
                print("   Create secret with: modal secret create kaggle-json-file --file path/to/kaggle.json")
                return False
    
    kaggle_json.chmod(0o600)
    
    # Verify
    print("\n🔍 Verifying Kaggle API...")
    result = subprocess.run(["kaggle", "competitions", "list", "-p", "1"], 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Kaggle API failed: {result.stderr}")
        print("\n💡 Troubleshooting:")
        print("   1. Regenerate API key at kaggle.com/settings")
        print("   2. Accept competition rules at:")
        print("      https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules")
        return False
    
    print("✅ Kaggle API working!")
    
    # Download
    download_dir = Path("/data/eyepacs")
    download_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(download_dir)
    
    # Check if already done
    train_dir = download_dir / "train"
    if train_dir.exists():
        n = len(list(train_dir.glob("*.jpeg")))
        if n > 30000:
            print(f"✅ Already have {n} images. Skipping download.")
            return True
    
    print("\n📥 Downloading EyePACS (~35GB)...")
    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "diabetic-retinopathy-detection"
        ], check=True, timeout=3600)
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Extract
    for zip_name in ["train.zip", "trainLabels.csv.zip"]:
        zip_path = download_dir / zip_name
        if zip_path.exists():
            print(f"\n📦 Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(download_dir if "Labels" in zip_name else download_dir / "train")
            zip_path.unlink()
    
    # Verify
    if train_dir.exists():
        n = len(list(train_dir.glob("*.jpeg")))
        print(f"\n✅ Extracted {n} images")
    
    eyepacs_volume.commit()
    print("\n🎉 Done! Dataset at /data/eyepacs/")
    return True

@app.local_entrypoint()
def main():
    success = download_eyepacs.remote()
    print("✅ Success!" if success else "❌ Failed")
