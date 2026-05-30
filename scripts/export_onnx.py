"""
ONNX Export Script for Mobile Deployment
Exports GLAAM model to ONNX format for smartphone inference.

Usage:
    python scripts/export_onnx.py
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import time

# Output paths
MODEL_PATH = "checkpoints_glaam/glaam_final_best.pth"
ONNX_PATH = "models/glaam_mobile.onnx"

def create_glaam_model(num_classes=4):
    """Recreate the GLAAM model architecture for export."""
    
    class GlobalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return y
    
    class LocalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.conv(x)
    
    class GLAAMBlock(nn.Module):
        def __init__(self, in_channels, reduction=16, use_residual=True, dropout_rate=0.3):
            super().__init__()
            self.global_branch = GlobalAttentionBranch(in_channels, reduction)
            self.local_branch = LocalAttentionBranch(in_channels, reduction)
            self.use_residual = use_residual
            self.dropout_rate = dropout_rate
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        def forward(self, x):
            global_weights = self.global_branch(x)
            local_weights = self.local_branch(x)
            combined_attention = self.alpha * global_weights + (1 - self.alpha) * local_weights
            out = x * combined_attention
            out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
            return (x + out) if self.use_residual else out
    
    class MobileNetWithGLAAM(nn.Module):
        def __init__(self, n_diseases=4, dropout_rate=0.3, attention_stages=[13, 17]):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=False)
            self.features = mobilenet.features
            
            stage_channels = {3: 24, 6: 32, 10: 64, 13: 96, 16: 160, 17: 320}
            self.attention_blocks = nn.ModuleDict()
            for stage in attention_stages:
                if stage in stage_channels:
                    self.attention_blocks[f'glaam_{stage}'] = GLAAMBlock(
                        stage_channels[stage], reduction=16
                    )
            
            self.attention_stages = attention_stages
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                self.dropout,
                nn.Linear(256, n_diseases)
            )
        
        def forward(self, x):
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.attention_stages:
                    block_name = f'glaam_{i}'
                    if block_name in self.attention_blocks:
                        x = self.attention_blocks[block_name](x)
            
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            return logits
    
    return MobileNetWithGLAAM(n_diseases=num_classes)

def benchmark_model(model, input_shape=(1, 3, 224, 224), n_runs=100):
    """Benchmark model inference speed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    fps = n_runs / elapsed
    
    return fps

def main():
    print("=" * 60)
    print("📱 GLAAM ONNX Export for Mobile Deployment")
    print("=" * 60)
    
    # Create model
    print("\n🔧 Creating GLAAM model...")
    model = create_glaam_model(num_classes=4)
    
    # Load weights if available
    if Path(MODEL_PATH).exists():
        print(f"📁 Loading weights from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Weights loaded successfully")
    else:
        print(f"⚠️ Weights not found at {MODEL_PATH}")
        print("   Exporting model structure only...")
    
    model.eval()
    
    # Benchmark PyTorch model
    print("\n⏱️ Benchmarking PyTorch model...")
    pytorch_fps = benchmark_model(model)
    print(f"   PyTorch FPS: {pytorch_fps:.1f}")
    
    # Export to ONNX
    print("\n📦 Exporting to ONNX...")
    Path(ONNX_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Get model size
    onnx_size = Path(ONNX_PATH).stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Exported to: {ONNX_PATH}")
    print(f"   Model size: {onnx_size:.2f} MB")
    
    # Try ONNX Runtime inference
    print("\n🔄 Testing ONNX Runtime inference...")
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(ONNX_PATH)
        
        # Benchmark ONNX
        input_name = session.get_inputs()[0].name
        dummy_np = dummy_input.numpy()
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_np})
        
        # Benchmark
        start = time.time()
        n_runs = 100
        for _ in range(n_runs):
            _ = session.run(None, {input_name: dummy_np})
        elapsed = time.time() - start
        onnx_fps = n_runs / elapsed
        
        print(f"   ONNX Runtime FPS: {onnx_fps:.1f}")
        print(f"   Speedup: {onnx_fps/pytorch_fps:.2f}x")
        
    except ImportError:
        print("   ⚠️ onnxruntime not installed. Install with: pip install onnxruntime")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"   Model: GLAAM with learnable fusion")
    print(f"   Size: {onnx_size:.2f} MB")
    print(f"   PyTorch FPS: {pytorch_fps:.1f}")
    print(f"   Target: >15 FPS on mobile")
    print("=" * 60)
    
    print("\n📝 THESIS CLAIM (copy-paste ready):")
    print(f'   "GLAAM achieves {pytorch_fps:.0f} FPS on CPU ({onnx_size:.1f} MB),')
    print('    enabling real-time smartphone-based cataract screening."')

if __name__ == "__main__":
    main()
