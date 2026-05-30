"""
Kaggle Credentials Diagnostic
Run: modal run scripts/kaggle_debug.py
"""

import modal
import os

image = modal.Image.debian_slim().pip_install(["kaggle"])
app = modal.App("kaggle-debug", image=image)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kaggle-creds")],
    timeout=120,
)
def debug_kaggle():
    """Debug Kaggle credentials"""
    import subprocess
    import json
    from pathlib import Path
    
    print("="*60)
    print("🔍 KAGGLE CREDENTIALS DIAGNOSTIC")
    print("="*60)
    
    # Check environment variables
    print("\n📋 1. Checking environment variables...")
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if username:
        print(f"   ✅ KAGGLE_USERNAME: {username}")
    else:
        print("   ❌ KAGGLE_USERNAME: NOT FOUND")
    
    if key:
        print(f"   ✅ KAGGLE_KEY: {key[:8]}...{key[-4:]} (masked)")
    else:
        print("   ❌ KAGGLE_KEY: NOT FOUND")
    
    if not username or not key:
        print("\n❌ FAIL: Missing credentials in Modal secret!")
        print("   Run: modal secret create kaggle-creds KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx")
        return False
    
    # Create kaggle.json
    print("\n📁 2. Creating kaggle.json...")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    
    with open(kaggle_json, 'w') as f:
        json.dump({"username": username, "key": key}, f)
    kaggle_json.chmod(0o600)
    print(f"   ✅ Created: {kaggle_json}")
    
    # Test API access
    print("\n🌐 3. Testing Kaggle API...")
    result = subprocess.run(
        ["kaggle", "competitions", "list", "-p", "1"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("   ✅ API authentication SUCCESSFUL!")
        print(f"   Output: {result.stdout[:200]}...")
    else:
        print(f"   ❌ API authentication FAILED!")
        print(f"   Error: {result.stderr}")
        return False
    
    # Check competition access
    print("\n🏆 4. Testing competition access...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", 
         "-c", "diabetic-retinopathy-detection", "--help"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("   ✅ Competition command works")
    else:
        print(f"   ⚠️ Competition access issue: {result.stderr}")
    
    # Try to download just the labels (small file)
    print("\n📥 5. Testing small file download...")
    result = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", "diabetic-retinopathy-detection",
         "-f", "trainLabels.csv.zip"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("   ✅ Download works!")
    else:
        print(f"   ❌ Download failed: {result.stderr}")
        if "403" in result.stderr or "401" in result.stderr:
            print("\n   💡 You need to accept competition rules!")
            print("   Visit: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED - Ready to download!")
    print("="*60)
    return True

@app.local_entrypoint()
def main():
    success = debug_kaggle.remote()
    if success:
        print("\n✅ Credentials working! Run the download script now.")
    else:
        print("\n❌ Fix the issues above first.")
