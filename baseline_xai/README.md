# Baseline XAI - MobileNetV2 Cataract Classifier

Simple baseline model for cataract detection **without** attention mechanisms.
Used to establish baseline metrics before adding GLAAM attention.

## Structure

```
baseline_xai/
├── models/
│   └── mobilenet_baseline.py   # MobileNetV2 classifier
├── training/
│   └── train_baseline.py       # Training script
├── evaluation/
│   └── xai_tools.py           # Grad-CAM, LIME, SHAP
├── checkpoints/               # Saved models
├── logs/                      # TensorBoard logs
└── xai_outputs/              # XAI visualizations
```

## Quick Start

### 1. Test Model
```bash
cd d:\projects\cataract_detection
python -c "from baseline_xai.models import MobileNetBaseline; m=MobileNetBaseline(); print(f'{sum(p.numel() for p in m.parameters()):,} params')"
```
Expected: ~3.5M parameters

### 2. Train Baseline
```bash
python baseline_xai/training/train_baseline.py \
    --data_root data/raw/slitlamp \
    --epochs 50 \
    --batch_size 32
```

### 3. Run XAI Analysis
```bash
python baseline_xai/evaluation/xai_tools.py \
    --model_path baseline_xai/checkpoints/best_baseline.pth \
    --image path/to/test_image.jpg
```

## XAI Tools Included

| Tool | What it Shows | Output |
|------|--------------|--------|
| **Grad-CAM** | Regions model focuses on | Heatmap overlay |
| **LIME** | Important image regions | Segmented regions |
| **SHAP** | Feature importance | Attribution map |

## Why This Baseline?

1. **Same backbone as GLAAM** → Fair comparison
2. **Simple architecture** → XAI tools work easily
3. **Fast training** → Quick iteration
4. **Clear improvement path** → Add attention later

## Dependencies

```bash
pip install torch torchvision lime shap grad-cam matplotlib
```

## Next Steps

After getting baseline metrics:
1. Compare with GLAAM model
2. Show accuracy improvement from attention
3. Include in paper as ablation study
