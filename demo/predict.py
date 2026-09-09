#!/usr/bin/env python3
"""
Standalone inference script for GLAAM-4X v4
Load model, run prediction on single image or folder.
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pathlib import Path

DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']

class GLAAM4XClassifier(nn.Module):
    REORDER_IDX = [2, 0, 1, 3]
    def __init__(self, dropout_rate=0.3, pretrained=False):
        super().__init__()
        # NOTE: Update this import path if your project structure differs
        from models.glaam_4x import GLAAM_4X
        self.backbone = GLAAM_4X(pretrained=pretrained, dropout_rate=dropout_rate)
    def forward(self, x):
        out = self.backbone(x)
        logits = out['logits']
        return logits[:, self.REORDER_IDX]

def load_model(weights_path, device='cuda'):
    model = GLAAM4XClassifier(dropout_rate=0.3, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_path, img_size=384):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def predict(model, img_path, thresholds=None, device='cuda', img_size=384):
    if thresholds is None:
        thresholds = {d: 0.5 for d in DISEASE_NAMES}
    img = preprocess_image(img_path, img_size).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    results = {}
    for i, disease in enumerate(DISEASE_NAMES):
        prob = float(probs[i])
        pred = int(prob >= thresholds.get(disease, 0.5))
        results[disease] = {"probability": round(prob, 4), "prediction": pred}
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model_weights.pth")
    parser.add_argument("--image", required=True, help="Path to fundus image")
    parser.add_argument("--thresholds", default="thresholds.json", help="Path to thresholds.json")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--img_size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    thresholds = {d: 0.5 for d in DISEASE_NAMES}
    if Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            thresholds = json.load(f)

    results = predict(model, args.image, thresholds, device, args.img_size)
    print(json.dumps(results, indent=2))
