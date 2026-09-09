"""
Batch Generate Synthetic Cataract Images for v4 Dataset
========================================================

Generates 1,500 synthetic cataract images from clear fundus sources.
Distribution: 60% moderate, 25% mild, 15% severe

Also includes ~200 naturally poor-quality images as cataract=0 negatives
to prevent the model from learning "blur = cataract".

Usage:
    python scripts/generate_synthetic_cataract.py \
        --num_images 1500 \
        --output_dir data/synthetic/cataract/v4 \
        --csv_path data/synthetic/cataract_v4_manifest.csv
"""

import sys
sys.path.insert(0, '/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection')

import numpy as np
import cv2
import os
import random
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
from augmentation.synthetic_cataract import SyntheticCataract


def find_clear_source_images(max_per_dataset=300):
    """
    Collect clear fundus images from all available datasets.
    Prioritizes normal/healthy images (no disease labels).
    """
    sources = []
    
    # Dataset paths and their normal image patterns
    datasets = [
        # PALM normal images (N prefix = normal)
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/PALM/Training/Classification', 
         lambda f: f.startswith('N') and f.endswith('.jpg')),
        
        # DDR test set (many are normal/early DR)
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/DDR/DR_grading/test',
         lambda f: f.endswith('.jpg')),
        
        # DDR train set
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/DDR/DR_grading/train',
         lambda f: f.endswith('.jpg')),
        
        # RFMiD test set
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/RFMiD/Test_set',
         lambda f: f.endswith('.png')),
        
        # RFMiD validation set
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/RFMiD/Validation_set',
         lambda f: f.endswith('.png')),
        
        # G1020 normal images (binaryLabels=0)
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/other_dataset/G1020/Images',
         lambda f: f.endswith('.jpg')),
        
        # ORIGA normal images (Glaucoma=0 in OrigaList.csv)
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/other_dataset/ORIGA/Images',
         lambda f: f.endswith('.jpg')),
        
        # REFUGE normal images (Label=0 in index.json)
        ('/media/sk/DE5A6C5F5A6C3705/projects/cataract_detection/data/raw/other_dataset/REFUGE/Images_Square',
         lambda f: f.endswith('.jpg')),
    ]
    
    for base_dir, filter_fn in datasets:
        if not os.path.isdir(base_dir):
            continue
        
        imgs = []
        for f in os.listdir(base_dir):
            if filter_fn(f):
                imgs.append(os.path.join(base_dir, f))
        
        # Shuffle and take subset
        random.shuffle(imgs)
        selected = imgs[:max_per_dataset]
        sources.extend(selected)
        print(f"  {base_dir}: {len(selected)}/{len(imgs)} images")
    
    random.shuffle(sources)
    print(f"\nTotal clear source images: {len(sources)}")
    return sources


def generate_synthetic_cataract(
    num_images=1500,
    output_dir='data/synthetic/cataract/v4',
    csv_path='data/synthetic/cataract_v4_manifest.csv',
    seed=42,
):
    """
    Generate synthetic cataract images with proper distribution.
    
    Parameters
    ----------
    num_images : int
        Number of synthetic cataract images to generate.
    output_dir : str
        Directory to save generated images.
    csv_path : str
        Path to save CSV manifest.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Severity distribution
    severities = (
        ['mild'] * int(num_images * 0.25) +
        ['moderate'] * int(num_images * 0.60) +
        ['severe'] * int(num_images * 0.15)
    )
    # Adjust for rounding
    while len(severities) < num_images:
        severities.append('moderate')
    severities = severities[:num_images]
    random.shuffle(severities)
    
    print(f"\nGenerating {num_images} synthetic cataract images...")
    print(f"  Mild: {severities.count('mild')} ({severities.count('mild')/num_images*100:.0f}%)")
    print(f"  Moderate: {severities.count('moderate')} ({severities.count('moderate')/num_images*100:.0f}%)")
    print(f"  Severe: {severities.count('severe')} ({severities.count('severe')/num_images*100:.0f}%)")
    
    # Collect source images
    source_images = find_clear_source_images(max_per_dataset=300)
    
    if len(source_images) < num_images:
        print(f"\n⚠️  Warning: Only {len(source_images)} source images available.")
        print(f"   Will cycle through sources with different seeds.")
        # Cycle through sources
        source_images = (source_images * ((num_images // len(source_images)) + 1))[:num_images]
    else:
        source_images = source_images[:num_images]
    
    # Generate images
    manifest = []
    
    for i, (src_path, severity) in enumerate(tqdm(zip(source_images, severities), total=num_images, desc="Generating")):
        # Load source image
        img = cv2.imread(src_path)
        if img is None:
            print(f"⚠️  Failed to load: {src_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size if needed
        if img.shape[0] != 384 or img.shape[1] != 384:
            img = cv2.resize(img, (384, 384))
        
        # Apply synthetic cataract
        aug = SyntheticCataract(severity=severity, seed=seed + i)
        synthetic, params = aug(img, force_apply=True)
        
        # Save image
        filename = f"syn_cataract_{i+1:05d}_{severity}_{params['blur_sigma']:.1f}blur.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), cv2.cvtColor(synthetic, cv2.COLOR_RGB2BGR))
        
        # Record in manifest
        manifest.append({
            'image_path': str(save_path),
            'source_image': src_path,
            'severity': severity,
            'blur_sigma': round(params['blur_sigma'], 2),
            'contrast_factor': round(params['contrast_factor'], 2),
            'haze_alpha': round(params['haze_alpha'], 2),
            'color_shift': round(params['color_shift_strength'], 2),
            'noise_sigma': round(params['noise_sigma'], 3),
            'vessel_attenuation': round(params['vessel_attenuation'], 2),
            'DR': 0,
            'Glaucoma': 0,
            'Cataract': 1,
            'Myopia': 0,
        })
    
    # Save manifest CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(manifest)} synthetic cataract images")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Manifest: {csv_path}")
    print(f"{'='*60}")
    
    return manifest


def generate_quality_negatives(
    num_images=200,
    output_dir='data/synthetic/cataract/v4_quality_negatives',
    csv_path='data/synthetic/cataract_quality_negatives_manifest.csv',
    seed=42,
):
    """
    Generate naturally poor-quality images as cataract=0 negatives.
    These teach the model that 'blur' ≠ 'cataract'.
    
    We apply mild blur/contrast to clear images but label as cataract=0.
    """
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {num_images} quality-negative images (cataract=0)...")
    
    # Collect source images
    source_images = find_clear_source_images(max_per_dataset=50)
    source_images = source_images[:num_images]
    
    # Cycle if needed
    if len(source_images) < num_images:
        source_images = (source_images * ((num_images // len(source_images)) + 1))[:num_images]
    
    manifest = []
    
    for i, src_path in enumerate(tqdm(source_images, total=num_images, desc="Quality Negs")):
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 384 or img.shape[1] != 384:
            img = cv2.resize(img, (384, 384))
        
        # Apply MILD degradation (not cataract-level)
        # Just enough to look "poor quality" but not "cataract"
        aug = SyntheticCataract(severity='mild', seed=seed + 2000 + i)
        degraded, params = aug(img, force_apply=True)
        
        # But override severity label
        filename = f"syn_quality_neg_{i+1:05d}_mild_{params['blur_sigma']:.1f}blur.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
        
        manifest.append({
            'image_path': str(save_path),
            'source_image': src_path,
            'severity': 'quality_negative',
            'blur_sigma': round(params['blur_sigma'], 2),
            'contrast_factor': round(params['contrast_factor'], 2),
            'haze_alpha': round(params['haze_alpha'], 2),
            'color_shift': round(params['color_shift_strength'], 2),
            'noise_sigma': round(params['noise_sigma'], 3),
            'vessel_attenuation': round(params['vessel_attenuation'], 2),
            'DR': 0,
            'Glaucoma': 0,
            'Cataract': 0,  # KEY: labeled as NOT cataract
            'Myopia': 0,
        })
    
    # Save manifest
    csv_path = Path(csv_path)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    
    print(f"\n✅ Generated {len(manifest)} quality-negative images")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Manifest: {csv_path}")
    
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cataract dataset")
    parser.add_argument("--num_images", type=int, default=1500, help="Number of synthetic cataract images")
    parser.add_argument("--num_quality_negs", type=int, default=200, help="Number of quality-negative images")
    parser.add_argument("--output_dir", type=str, default="data/synthetic/cataract/v4")
    parser.add_argument("--csv_path", type=str, default="data/synthetic/cataract_v4_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*60)
    print("  Synthetic Cataract Dataset Generation — v4")
    print("="*60)
    
    # Generate synthetic cataract images
    manifest_cataract = generate_synthetic_cataract(
        num_images=args.num_images,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        seed=args.seed,
    )
    
    # Generate quality negatives
    manifest_negs = generate_quality_negatives(
        num_images=args.num_quality_negs,
        output_dir=args.output_dir + "_quality_negatives",
        csv_path=args.csv_path.replace(".csv", "_quality_negatives.csv"),
        seed=args.seed,
    )
    
    print(f"\n{'='*60}")
    print(f"  TOTAL GENERATED")
    print(f"  ├── Synthetic Cataract: {len(manifest_cataract)} (cataract=1)")
    print(f"  ├── Quality Negatives:  {len(manifest_negs)} (cataract=0)")
    print(f"  └── Combined:           {len(manifest_cataract) + len(manifest_negs)}")
    print(f"{'='*60}")
