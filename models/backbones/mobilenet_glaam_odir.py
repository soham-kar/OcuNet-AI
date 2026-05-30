# models/backbones/mobilenet_glaam_odir.py (CORRECTED)
"""
GLAAM-enhanced MobileNetV2 for ODIR-5K multi-label classification
Fixed: Shape mismatch in baseline mode
"""

import torch
import torch.nn as nn
from models.attention.glaam_exportable import GLAAMBlockExportable
import torchvision.models as models


class MobileNetV2WithGLAAM_ODIR(nn.Module):
    """
    MobileNetV2 with GLAAM attention for ODIR-5K multi-label classification
    
    Args:
        num_classes: Number of disease labels (default: 7 for ODIR-5K)
        pretrained: Load ImageNet pretrained weights
        attention_stages: Which MobileNet blocks to insert GLAAM
        dropout_rate: Dropout for Bayesian uncertainty
    """
    
    def __init__(self, num_classes=7, pretrained=True, attention_stages=[6, 13, 17], dropout_rate=0.2):
        super().__init__()
        
        # Load backbone
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features
        
        # CRITICAL FIX: Always use final stage features for classification
        # MobileNetV2 final output channels = 1280 (not 320)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Stage info for GLAAM insertion
        # CORRECTED based on actual MobileNetV2 output:
        # Stage 0: 32, 1: 16, 2-3: 24, 4-6: 32, 7-10: 64, 11-13: 96, 14-16: 160, 17: 320, 18: 1280
        self.stage_info = {
            3: {'channels': 24, 'name': 'early'},
            6: {'channels': 32, 'name': 'mid1'},
            10: {'channels': 64, 'name': 'mid2'},
            13: {'channels': 96, 'name': 'deep1'},
            16: {'channels': 160, 'name': 'deep2'},
            17: {'channels': 320, 'name': 'final'}  # FIXED: was 160, should be 320
        }
        
        # Insert GLAAM blocks only at specified stages
        self.attention_blocks = nn.ModuleDict()
        self.use_glaam = len(attention_stages) > 0  # Flag to check if GLAAM is enabled
        
        for stage_idx in attention_stages:
            if stage_idx in self.stage_info:
                channels = self.stage_info[stage_idx]['channels']
                self.attention_blocks[f'glaam_stage_{stage_idx}'] = GLAAMBlockExportable(
                    in_channels=channels, reduction=16, use_residual=True
                )
        
        # Classification head (always takes 1280 from final features)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(256, num_classes)
        )
        
        self._attention_storage = {}
    
    def forward(self, x, return_attention=False):
        """Forward pass - FIXED to always output 1280-dim features"""
        if return_attention:
            self._attention_storage = {}
        
        # Pass through all MobileNet features
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Apply GLAAM if configured for this stage
            if f'glaam_stage_{i}' in self.attention_blocks:
                attn_block = self.attention_blocks[f'glaam_stage_{i}']
                
                if return_attention:
                    x, attn_dict = attn_block(x, return_attention=True)
                    self._attention_storage[f'stage_{i}'] = {
                        'combined_attention': attn_dict['combined_attention']
                    }
                else:
                    x = attn_block(x)
        
        # CRITICAL: x now has shape (B, 1280, 7, 7) for 224x224 input
        # Global pooling reduces to (B, 1280, 1, 1)
        pooled = self.global_pool(x)
        features = torch.flatten(pooled, 1)  # (B, 1280)
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'attention_maps': self._attention_storage if return_attention else {}
        }


class BayesianGLAAM_ODIR(nn.Module):
    """Bayesian wrapper for MobileNetV2WithGLAAM_ODIR"""
    
    def __init__(self, base_model: MobileNetV2WithGLAAM_ODIR, dropout_rate=0.2, num_samples=10):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
        # Add dropout to classifier (simple approach)
        self._enhance_dropout()
    
    def _enhance_dropout(self):
        """Increase dropout in classifier"""
        # Replace the existing dropout with higher rate
        self.base_model.dropout.p = self.dropout_rate
    
    def forward(self, x, return_uncertainty=False):
        # Monte Carlo Dropout sampling
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        
        self.base_model.apply(enable_dropout)
        
        logit_samples = []
        attention_samples = []
        
        for _ in range(self.num_samples):
            outputs = self.base_model(x, return_attention=True)
            logit_samples.append(outputs['logits'])
            
            # Store attention maps
            if outputs['attention_maps']:
                final_stage = list(outputs['attention_maps'].keys())[-1]
                attn_map = outputs['attention_maps'][final_stage]['combined_attention']
                attention_samples.append(attn_map.cpu())
        
        # Stack samples
        logits_stack = torch.stack(logit_samples)
        
        predictions = {
            'logits': logits_stack.mean(dim=0),
            'logits_std': logits_stack.std(dim=0)
        }
        
        if return_uncertainty and attention_samples:
            uncertainty = {
                'attention_variance': torch.stack(attention_samples).var(dim=0),
                'num_samples': self.num_samples
            }
            return predictions, uncertainty
        
        return predictions


# ✅ FIXED Smoke test
if __name__ == "__main__":
    print("🔧 Running smoke test for GLAAM-ODIR...\n")
    
    # Test 1: Baseline mode (no GLAAM)
    print("1️⃣ Testing baseline mode...")
    baseline_model = MobileNetV2WithGLAAM_ODIR(num_classes=7, pretrained=False, attention_stages=[])
    x = torch.randn(2, 3, 224, 224)
    out = baseline_model(x)
    print(f"   Baseline output: {out['logits'].shape}")
    print(f"   Baseline features: {out['features'].shape}")
    assert out['features'].shape[1] == 1280, f"Expected 1280 features, got {out['features'].shape[1]}"
    print("   ✅ Baseline test passed")
    
    # Test 2: GLAAM mode
    print("\n2️⃣ Testing GLAAM mode...")
    glaam_model = MobileNetV2WithGLAAM_ODIR(num_classes=7, pretrained=False, attention_stages=[6, 13, 17])
    out_glaam = glaam_model(x, return_attention=True)
    print(f"   GLAAM output: {out_glaam['logits'].shape}")
    print(f"   Attention stages: {len(out_glaam['attention_maps'])}")
    assert out_glaam['features'].shape[1] == 1280, f"Expected 1280 features, got {out_glaam['features'].shape[1]}"
    print("   ✅ GLAAM test passed")
    
    # Test 3: Bayesian wrapper
    print("\n3️⃣ Testing Bayesian wrapper...")
    bayesian_model = BayesianGLAAM_ODIR(glaam_model, dropout_rate=0.2, num_samples=5)
    out_bayes, uncertainty = bayesian_model(x, return_uncertainty=True)
    print(f"   Bayesian logits: {out_bayes['logits'].shape}")
    print(f"   Uncertainty: {uncertainty['attention_variance'].shape}")
    print("   ✅ Bayesian test passed")
    
    # Test 4: Backward pass
    print("\n4️⃣ Testing backward pass...")
    loss = out_glaam['logits'].mean()
    loss.backward()
    print("   ✓ Gradients computed")
    
    # Test 5: Verify no gradient issues
    for name, param in glaam_model.named_parameters():
        if 'glaam' in name and param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    print("   ✅ Gradient verification passed")
    
    print("\n" + "=" * 60)
    print("✅ All smoke tests passed! Model is ready for training.")
    print("=" * 60)
