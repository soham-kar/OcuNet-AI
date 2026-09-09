# GLAAM-4X v4 Inference Package

## Contents
- `model_weights.pth` — Clean model state dict (~14MB)
- `best_model.safetensors` — SafeTensors format (same weights)
- `thresholds.json` — Per-disease optimal thresholds
- `model_info.json` — Training metadata and config
- `predict.py` — Standalone inference script

## Quick Start

### Single Image Prediction
```bash
python predict.py \
    --weights model_weights.pth \
    --image path/to/fundus.jpg \
    --thresholds thresholds.json
```

### Load in Python
```python
import torch
from predict import load_model, predict

model = load_model("model_weights.pth", device="cuda")
results = predict(model, "fundus.jpg", thresholds={"Cataract": 0.45, ...})
print(results)
# {"Cataract": {"probability": 0.9234, "prediction": 1}, ...}
```

## Model Info
- Architecture: GLAAM-4X (MobileNetV2 + 4 attention heads)
- Input size: 384x384
- Diseases: Cataract, DR, Glaucoma, Myopia
- Best Val F1: 0.8275

## Thresholds
- Cataract: 0.49
- DR: 0.45
- Glaucoma: 0.57
- Myopia: 0.50
