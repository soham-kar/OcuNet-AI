"""
Training script for YOLOv8-nano lens detector.

Usage:
    python training/train_yolo.py --data_yaml data/lens_detection/data.yaml --epochs 50
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolo import LensDetector, LensDetectorDataset


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Lens Detector")
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='lens_detector', help='Experiment name')
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8 Lens Detector Training")
    print("=" * 60)
    
    # Initialize detector with pretrained YOLOv8-nano
    detector = LensDetector()
    
    # Train
    print(f"\nTraining for {args.epochs} epochs on {args.data_yaml}")
    print(f"Image size: {args.imgsz}, Batch size: {args.batch}")
    
    best_weights = detector.train(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name
    )
    
    print(f"\nTraining complete!")
    print(f"Best weights saved to: {best_weights}")
    
    # Export to ONNX for mobile
    print("\nExporting to ONNX...")
    onnx_path = detector.export(format="onnx", imgsz=args.imgsz)
    print(f"ONNX model saved to: {onnx_path}")


def create_annotation_instructions():
    """Print instructions for annotating lens regions."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           LENS ANNOTATION INSTRUCTIONS                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Step 1: Install LabelImg                                       ║
║     pip install labelImg                                        ║
║                                                                  ║
║  Step 2: Run LabelImg                                           ║
║     labelImg path/to/images/                                    ║
║                                                                  ║
║  Step 3: Annotate lens region                                   ║
║     - Click "Create RectBox"                                    ║
║     - Draw box around the LENS (circular/oval area)             ║
║     - Label it as "lens"                                        ║
║     - Save in YOLO format (one .txt file per image)             ║
║                                                                  ║
║  Step 4: Create directory structure                             ║
║     data/lens_detection/                                        ║
║     ├── images/                                                 ║
║     │   ├── train/   (80% of images)                           ║
║     │   └── val/     (20% of images)                           ║
║     ├── labels/                                                 ║
║     │   ├── train/   (matching .txt files)                     ║
║     │   └── val/                                                ║
║     └── data.yaml                                               ║
║                                                                  ║
║  Step 5: Create data.yaml                                       ║
║     path: data/lens_detection                                   ║
║     train: images/train                                         ║
║     val: images/val                                             ║
║     names:                                                      ║
║       0: lens                                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help-annotate":
        create_annotation_instructions()
    else:
        main()
