"""
Modal Cloud Training for YOLO-GLAAM Cataract Detection.

SAFE TEST MODE INCLUDED - validates entire pipeline in <10 min for ~$0.50

USAGE:
    # 1. Create volumes (one-time)
    modal volume create cataract-data
    modal volume create cataract-checkpoints
    
    # 2. Upload data
    modal volume put cataract-data ./data/raw/slitlamp /slitlamp
    modal volume put cataract-data ./training /training
    modal volume put cataract-data ./models /models
    modal volume put cataract-data ./utils /utils
    
    # 3. RUN TEST FIRST (~5 min, ~$0.50)
    modal run modal_train.py::test_pipeline
    
    # 4. If test passes, run full training
    modal run modal_train.py::train_full_model --epochs=100 --use-yolo=True
    
    # 5. Download results
    modal volume get cataract-checkpoints ./checkpoints_from_modal/
"""

import modal
import subprocess
import os
import sys
from pathlib import Path

# Persistent volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# GPU environment with system dependencies for OpenCV
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])  # Required for OpenCV
    .pip_install([
        "numpy<2.0",  # Pin numpy 1.x for torch compatibility
        "torch==2.1.0", 
        "torchvision==0.16.0", 
        "ultralytics>=8.0.0",
        "opencv-python-headless", 
        "scikit-learn", 
        "pandas",
        "tqdm", 
        "pyyaml", 
        "tensorboard", 
        "pillow", 
        "grad-cam"
    ])
)

app = modal.App("cataract-glaam", image=image)


@app.function(
    gpu="A100",
    timeout=600,  # 10 minutes max
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def test_pipeline():
    """
    ⚡ QUICK PIPELINE VALIDATION (< 10 min, ~$0.50)
    
    Tests:
    1. Data loading
    2. Model instantiation (YOLO + GLAAM)
    3. Forward pass
    4. Loss computation
    5. Checkpoint saving
    
    Run this FIRST before full training!
    
    Usage:
        modal run modal_train.py::test_pipeline
    """
    import torch
    
    print("=" * 60)
    print("🧪 PIPELINE VALIDATION TEST (Dry Run)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    sys.path.insert(0, "/data")
    
    # Step 1: Verify data exists
    print("\n[1/6] Checking data...")
    slitlamp_path = Path("/data/slitlamp")
    if not slitlamp_path.exists():
        print("❌ FAIL: /data/slitlamp not found")
        print("   Upload with: modal volume put cataract-data ./data/raw/slitlamp /slitlamp")
        return False
    
    # Count images
    jpg_files = list(slitlamp_path.rglob("*.jpg")) + list(slitlamp_path.rglob("*.JPG"))
    print(f"   ✓ Found {len(jpg_files)} images in /data/slitlamp")
    
    if len(jpg_files) == 0:
        print("❌ FAIL: No images found!")
        return False
    
    # Step 2: Test data loader
    print("\n[2/6] Testing data loader...")
    try:
        from utils.dataset import create_dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            data_root="/data/slitlamp",
            dataset_type="slitlamp",
            batch_size=2,
            num_workers=0,
            image_size=224
        )
        
        batch = next(iter(train_loader))
        images = batch['image']
        binary_labels = batch['binary_label']
        severity_labels = batch['severity_label']
        
        print(f"   ✓ Batch shape: {images.shape}")
        print(f"   ✓ Labels: binary={binary_labels.tolist()}, severity={severity_labels.tolist()}")
        
    except Exception as e:
        print(f"❌ FAIL: Data loader error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test model instantiation
    print("\n[3/6] Testing model creation...")
    try:
        from models import HybridCataractModel
        
        # Test without YOLO
        model = HybridCataractModel(use_yolo=False)
        print("   ✓ GLAAM model created (no YOLO)")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
        
        # Check for YOLO weights
        yolo_weights_path = "/checkpoints/yolo/lens_detector.pt"
        if os.path.exists(yolo_weights_path):
            model_yolo = HybridCataractModel(
                use_yolo=True, 
                yolo_weights=yolo_weights_path
            )
            print("   ✓ YOLO-GLAAM hybrid created")
        else:
            print("   ⚠ YOLO weights not found (train YOLO first for full model)")
        
    except Exception as e:
        print(f"❌ FAIL: Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test forward pass
    print("\n[4/6] Testing forward pass...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dummy_images = torch.randn(2, 3, 224, 224).to(device)
        model = model.to(device)
        
        with torch.no_grad():
            output = model(dummy_images)
        
        print(f"   ✓ Output keys: {list(output.keys())}")
        print(f"   ✓ Binary logits shape: {output['binary_logits'].shape}")
        print(f"   ✓ Severity logits shape: {output['severity_logits'].shape}")
        
    except Exception as e:
        print(f"❌ FAIL: Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test loss computation
    print("\n[5/6] Testing loss computation...")
    try:
        from models import MultiTaskLoss
        
        criterion = MultiTaskLoss()
        binary_labels_t = torch.tensor([0, 1]).to(device)
        severity_labels_t = torch.tensor([0, 3]).to(device)
        
        losses = criterion(
            output['binary_logits'],
            output['severity_logits'],
            binary_labels_t,
            severity_labels_t
        )
        
        print(f"   ✓ Total loss: {losses['loss'].item():.4f}")
        print(f"   ✓ Binary loss: {losses['binary_loss'].item():.4f}")
        print(f"   ✓ Severity loss: {losses['severity_loss'].item():.4f}")
        
    except Exception as e:
        print(f"❌ FAIL: Loss computation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test checkpoint saving
    print("\n[6/6] Testing checkpoint saving...")
    try:
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'best_val_acc': 0.0
        }
        
        save_dir = "/checkpoints/test"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/validation_test.pth"
        torch.save(checkpoint, save_path)
        
        # Verify it saved
        loaded = torch.load(save_path)
        assert 'epoch' in loaded
        
        print(f"   ✓ Checkpoint saved and verified: {save_path}")
        
        # Commit volume
        checkpoint_volume.commit()
        
    except Exception as e:
        print(f"❌ FAIL: Checkpoint saving error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nPipeline is ready for full training.")
    print("\nNext steps:")
    print("  1. Train YOLO detector (if using):")
    print("     modal run modal_train.py::train_yolo_detector --epochs=50")
    print("  2. Train full model:")
    print("     modal run modal_train.py::train_full_model --epochs=100 --use-yolo=True")
    
    return True


@app.function(
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def train_yolo_detector(epochs: int = 50):
    """
    Train YOLOv8-nano lens detector.
    
    Usage:
        modal run modal_train.py::train_yolo_detector --epochs=50
    
    Expected time: ~1 hour
    Cost: ~$3
    Output: /checkpoints/yolo/lens_detector.pt
    """
    sys.path.insert(0, "/data")
    
    print("=" * 50)
    print("YOLO Lens Detector Training")
    print("=" * 50)
    print(f"Epochs: {epochs}")
    
    # Verify training script exists
    if not os.path.exists("/data/training/train_yolo.py"):
        print("❌ ERROR: /data/training/train_yolo.py not found!")
        print("   Upload with: modal volume put cataract-data ./training /training")
        return 1
    
    os.makedirs("/checkpoints/yolo", exist_ok=True)
    
    cmd = [
        sys.executable, "/data/training/train_yolo.py",
        "--data_dir", "/data/yolo_annotations",
        "--epochs", str(epochs),
        "--batch_size", "16",
        "--output", "/checkpoints/yolo/lens_detector.pt"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ YOLO training complete!")
        checkpoint_volume.commit()
    else:
        print(f"❌ Training failed with code {result.returncode}")
    
    return result.returncode


@app.function(
    gpu="A100",
    timeout=28800,  # 8 hours
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def train_full_model(
    epochs: int = 100, 
    use_yolo: bool = False, 
    dual_view: bool = False,
    run_name: str = "exp1",
    batch_size: int = 32
):
    """
    Train full YOLO-GLAAM hybrid model.
    
    Usage:
        modal run modal_train.py::train_full_model --epochs=100 --use-yolo=True --run-name="exp1"
    
    Expected time: 6-8 hours
    Cost: ~$18
    Output: /checkpoints/{run_name}/best.pth
    """
    sys.path.insert(0, "/data")
    
    print("=" * 50)
    print("YOLO-GLAAM Hybrid Model Training")
    print("=" * 50)
    print(f"Run name: {run_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Use YOLO: {use_yolo}")
    print(f"Dual View: {dual_view}")
    
    # Verify training script exists
    if not os.path.exists("/data/training/train_hybrid.py"):
        print("❌ ERROR: /data/training/train_hybrid.py not found!")
        print("   Upload with: modal volume put cataract-data ./training /training")
        return 1
    
    checkpoint_dir = f"/checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "/data/training/train_hybrid.py",
        "--data_root", "/data/slitlamp",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--checkpoint_dir", checkpoint_dir,
        "--lr", "0.001",
        "--patience", "15",
        "--mc_samples", "10"
    ]
    
    # Add YOLO if weights exist
    yolo_weights = "/checkpoints/yolo/lens_detector.pt"
    if use_yolo:
        if os.path.exists(yolo_weights):
            cmd.extend(["--use_yolo", "--yolo_weights", yolo_weights])
        else:
            print("⚠ WARNING: YOLO weights not found, training without YOLO")
    
    if dual_view:
        cmd.append("--dual_view")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("=" * 50)
        print(f"✅ Training {run_name} complete!")
        print(f"Checkpoint: {checkpoint_dir}/best.pth")
        print("=" * 50)
        print("\nDownload with:")
        print(f"  modal volume get cataract-checkpoints ./{run_name}")
        checkpoint_volume.commit()
    else:
        print(f"❌ Training failed with code {result.returncode}")
    
    return result.returncode


@app.function(
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume}
)
def train_odir_baseline(epochs: int = 50, batch_size: int = 32):
    """
    Train GLAAM baseline on ODIR-5K fundus dataset.
    
    Usage:
        modal run modal_train.py::train_odir_baseline --epochs=50
    
    Expected time: 2-3 hours
    Cost: ~$6
    """
    sys.path.insert(0, "/data")
    
    print("=" * 50)
    print("ODIR-5K GLAAM Baseline Training")
    print("=" * 50)
    
    if not os.path.exists("/data/odir"):
        print("❌ ERROR: ODIR data not found!")
        print("   Upload with: modal volume put cataract-data ./data/raw/odir /odir")
        return 1
    
    checkpoint_dir = "/checkpoints/odir_baseline"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "/data/training/train_hybrid.py",
        "--data_root", "/data/odir",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--checkpoint_dir", checkpoint_dir
        # No YOLO for fundus baseline
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ ODIR baseline complete!")
        checkpoint_volume.commit()
    
    return result.returncode


@app.local_entrypoint()
def main():
    """Print usage instructions."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║         YOLO-GLAAM Cataract Detection - Modal Training             ║
╚════════════════════════════════════════════════════════════════════╝

STEP 1: Create volumes (one-time)
──────────────────────────────────
  modal volume create cataract-data
  modal volume create cataract-checkpoints

STEP 2: Upload data & code
──────────────────────────
  modal volume put cataract-data ./data/raw/slitlamp /slitlamp
  modal volume put cataract-data ./data/raw/odir /odir
  modal volume put cataract-data ./training /training
  modal volume put cataract-data ./models /models
  modal volume put cataract-data ./utils /utils
  modal volume put cataract-data ./configs /configs

  # Verify upload:
  modal volume ls cataract-data /

STEP 3: ⚡ RUN TEST FIRST (~5 min, ~$0.50)
─────────────────────────────────────────
  modal run modal_train.py::test_pipeline

STEP 4: Train YOLO detector (~1 hour, ~$3)
──────────────────────────────────────────
  modal run modal_train.py::train_yolo_detector --epochs=50

STEP 5: Train full model (~6-8 hours, ~$18)
───────────────────────────────────────────
  modal run modal_train.py::train_full_model \\
    --epochs=100 \\
    --use-yolo=True \\
    --run-name="exp1"

STEP 6: Download results
────────────────────────
  modal volume get cataract-checkpoints ./checkpoints_from_modal/

OTHER COMMANDS:
───────────────
  # ODIR baseline (~2 hours, ~$6)
  modal run modal_train.py::train_odir_baseline --epochs=50
""")
