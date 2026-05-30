"""
Smoke test for GLAAM-ODIR integration
Runs in < 1 minute to verify shapes and forward/backward pass
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.backbones.mobilenet_glaam_odir import MobileNetV2WithGLAAM_ODIR, BayesianGLAAM_ODIR
from utils.losses import ODIRMultiLabelLoss


def test_integration():
    print("=" * 60)
    print("🔧 Testing GLAAM-ODIR Integration")
    print("=" * 60)
    
    # Simulate ODIR-5K batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # ODIR fundus size
    targets = torch.randint(0, 2, (batch_size, 7)).float()  # Multi-label
    
    # Test 1: Baseline model
    print("\n1️⃣  Testing Baseline MobileNetV2...")
    baseline = MobileNetV2WithGLAAM_ODIR(pretrained=False, attention_stages=[])  # No GLAAM
    out_baseline = baseline(images)
    print(f"   ✓ Output shape: {out_baseline['logits'].shape}")
    assert out_baseline['logits'].shape == (batch_size, 7), "Shape mismatch!"
    print("   ✅ Baseline test passed")
    
    # Test 2: GLAAM model
    print("\n2️⃣  Testing GLAAM-Enhanced Model...")
    glaam_model = MobileNetV2WithGLAAM_ODIR(pretrained=False, attention_stages=[6, 13, 17])
    out_glaam = glaam_model(images, return_attention=True)
    print(f"   ✓ Output shape: {out_glaam['logits'].shape}")
    print(f"   ✓ Attention stages: {len(out_glaam['attention_maps'])}")
    assert out_glaam['logits'].shape == (batch_size, 7), "Shape mismatch!"
    print("   ✅ GLAAM test passed")
    
    # Test 3: Bayesian wrapper
    print("\n3️⃣  Testing Bayesian GLAAM...")
    bayesian_model = BayesianGLAAM_ODIR(glaam_model, dropout_rate=0.2, num_samples=5)
    out_bayes, uncertainty = bayesian_model(images, return_uncertainty=True)
    print(f"   ✓ Output shape: {out_bayes['logits'].shape}")
    print(f"   ✓ Uncertainty shape: {out_bayes['logits_std'].shape}")
    if uncertainty['attention_variance'] is not None:
        print(f"   ✓ Attention variance: {uncertainty['attention_variance'].shape}")
    assert out_bayes['logits'].shape == (batch_size, 7), "Shape mismatch!"
    print("   ✅ Bayesian test passed")
    
    # Test 4: Loss function
    print("\n4️⃣  Testing ODIR Loss...")
    criterion = ODIRMultiLabelLoss()
    loss = criterion(out_glaam['logits'], targets)
    print(f"   ✓ Loss value: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    print("   ✅ Loss test passed")
    
    # Test 5: Backward pass
    print("\n5️⃣  Testing Backward Pass...")
    loss.backward()
    print("   ✓ Gradients computed successfully")
    
    # Check gradient flow through GLAAM
    glaam_grad_found = False
    for name, param in glaam_model.named_parameters():
        if 'glaam' in name and param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                glaam_grad_found = True
                break
    
    assert glaam_grad_found, "No gradients flowing through GLAAM!"
    print("   ✓ GLAAM gradients verified")
    print("   ✅ Backward pass test passed")
    
    # Test 6: Parameter count
    print("\n6️⃣  Testing Parameter Count...")
    total_params = sum(p.numel() for p in glaam_model.parameters())
    glaam_params = sum(p.numel() for p in glaam_model.attention_blocks.parameters())
    overhead = (glaam_params / total_params) * 100
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ GLAAM parameters: {glaam_params:,}")
    print(f"   ✓ GLAAM overhead: {overhead:.2f}%")
    assert overhead < 15, f"GLAAM overhead too high: {overhead:.2f}%"
    print("   ✅ Parameter count test passed")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! GLAAM-ODIR is ready for training.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_integration()
        if success:
            print("\n🎉 Integration successful! Next steps:")
            print("   1. Create training configuration")
            print("   2. Prepare ODIR-5K dataset")
            print("   3. Run training: modal run modal_train_glaam_odir.py")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
