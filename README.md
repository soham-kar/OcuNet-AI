# OcuNet-AI: Multi-Disease Fundus Classification with GLAAM-4X

> **A unified deep learning framework for detecting Cataract, Diabetic Retinopathy (DR), Glaucoma, and Myopia from retinal fundus photographs using disease-specific attention mechanisms.**

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Key Innovation

This work addresses critical gaps in existing cataract detection research:

| Gap in Literature | Our Solution |
|-------------------|--------------|
| GLAAM uses fundus (indirect) imaging | Apply GLAAM to **slit-lamp** (direct lens) imaging |
| No lens region localization | **YOLOv8-nano** detects lens ROI |
| Binary detection OR severity grading | **Multi-task**: both simultaneously |
| No mobile deployment validation | **ONNX export** for mobile inference |

## 📁 Project Structure

```
cataract_detection/
├── configs/
│   └── config.yaml              # Hyperparameters
├── models/
│   ├── attention/
│   │   ├── glaam.py            # GLAAM attention module
│   │   └── glaai.py            # GLAAI attention module
│   ├── backbones/
│   │   └── backbone.py         # MobileNetV2/InceptionV3
│   ├── yolo/
│   │   └── lens_detector.py    # YOLOv8 lens detector
│   └── hybrid_model.py         # Main YOLO-GLAAM model
├── training/
│   ├── train_hybrid.py         # Train main model
│   └── train_yolo.py           # Train lens detector
├── evaluation/
│   ├── grad_cam.py             # Interpretability
│   └── benchmark.py            # Speed benchmarks
├── utils/
│   └── dataset.py              # Dataset utilities
└── requirements.txt
```

## ⚙️ Installation

```bash
# Clone and enter directory
cd d:\projects\cataract_detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset Setup

### 1. Download Mendeley Slit-lamp Dataset

Download from: [Mendeley Nuclear Cataract Database](https://data.mendeley.com/datasets/6wv33nbcvv/2)

```bash
# Extract to:
data/raw/slitlamp/
├── 45_degree/
│   ├── NO_1.jpg
│   ├── NC1_1.jpg
│   └── ...
└── 135_degree/
    └── ...
```

### 2. (Optional) ODIR Fundus Dataset for Baseline

Download from: [Kaggle ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

## 🏋️ Training

### Step 1: Train YOLO Lens Detector (Optional)

First, annotate ~100-200 images with lens bounding boxes:

```bash
# See annotation instructions
python training/train_yolo.py --help-annotate

# Train YOLO
python training/train_yolo.py --data_yaml data/lens_detection/data.yaml --epochs 50
```

### Step 2: Train Hybrid Model

```bash
python training/train_hybrid.py \
    --data_root data/raw/slitlamp \
    --epochs 100 \
    --batch_size 32 \
    --backbone mobilenetv2 \
    --attention glaam
```

Training logs: `tensorboard --logdir logs/`

## 📈 Evaluation

### Grad-CAM Visualization

```bash
python evaluation/grad_cam.py \
    --model checkpoints/best.pth \
    --image test_image.jpg \
    --task binary \
    --output gradcam_output.png
```

### Benchmark Speed

```bash
python evaluation/benchmark.py --device cuda --iterations 100
```

## 📱 Mobile Export

```python
from models import HybridCataractModel

model = HybridCataractModel.load_from_checkpoint('checkpoints/best.pth')
model.export_onnx('model.onnx', input_size=(384, 384))
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{yolo-glaam-2025,
  title={YOLO-GLAAM: A Hybrid Attention Model for Real-time Cataract Detection and Severity Grading on Slit-lamp Images},
  author={Your Name},
  year={2025}
}
```

## 🔬 References

1. Kumar et al. (2025) - GLAAM and GLAAI: Pioneering attention models for robust automated cataract detection
2. Junayed et al. (2021) - CataractNet: An Automated Cataract Detection System
3. Cruz-Vega et al. (2023) - Nuclear Cataract Database for Biomedical and Machine Learning Applications

## 📄 License

MIT License
