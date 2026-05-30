"""
Benchmark script for inference speed comparison.

Compares:
1. Full image inference (no YOLO)
2. YOLO-GLAAM hybrid (ROI detection + classification)
"""

import argparse
import time
from pathlib import Path
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import HybridCataractModel


def benchmark_model(
    model: torch.nn.Module,
    input_size: tuple,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10
) -> dict:
    """
    Benchmark model inference speed.
    
    Args:
        model: The model to benchmark
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on
        num_iterations: Number of iterations for timing
        warmup: Number of warmup iterations
        
    Returns:
        dict with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'fps': 1000 / times.mean()
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {
            'name': 'MobileNetV2 + GLAAM (384x384 ROI)',
            'backbone': 'mobilenetv2',
            'attention': 'glaam',
            'input_size': (args.batch_size, 3, 384, 384)
        },
        {
            'name': 'MobileNetV2 + GLAAI (384x384 ROI)',
            'backbone': 'mobilenetv2',
            'attention': 'glaai',
            'input_size': (args.batch_size, 3, 384, 384)
        },
        {
            'name': 'MobileNetV2 + GLAAM (Full 640x640)',
            'backbone': 'mobilenetv2',
            'attention': 'glaam',
            'input_size': (args.batch_size, 3, 640, 640)
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 40)
        
        # Create model
        model = HybridCataractModel(
            backbone=config['backbone'],
            attention_type=config['attention'],
            pretrained=False,  # Skip downloading weights for benchmark
            use_yolo=False
        )
        
        # Benchmark
        stats = benchmark_model(
            model=model,
            input_size=config['input_size'],
            device=device,
            num_iterations=args.iterations
        )
        
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        print(f"  FPS:  {stats['fps']:.1f}")
        
        results.append({
            'name': config['name'],
            **stats
        })
        
        # Clean up
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<45} {'FPS':>8} {'ms':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<45} {r['fps']:>8.1f} {r['mean_ms']:>10.2f}")
    
    # Calculate speedup
    if len(results) >= 3:
        roi_fps = results[0]['fps']
        full_fps = results[2]['fps']
        speedup = roi_fps / full_fps
        print(f"\nROI (384x384) vs Full (640x640) speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
