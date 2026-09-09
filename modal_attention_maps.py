# modal_attention_maps.py
"""
Disease-Specific Attention Map Visualization for GLAAM-4X
=========================================================

Extracts and visualizes the per-disease attention maps from the trained
GLAAM-4X model. Each disease has its own attention head that produces a
spatial attention map highlighting where the model looks for that disease.

This is the KEY interpretability contribution:
  - DR: MultiScaleGLAAM -> should focus on microaneurysms/lesions
  - Glaucoma: GLAAMBlock -> should focus on the optic disc
  - Cataract: GLAAMBlock -> should focus on lens opacity
  - Myopia: GLAAMBlock -> should focus on peripapillary atrophy / tilted disc

Usage:
    modal run modal_attention_maps.py
"""

import modal
import json
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Modal Image
# ═══════════════════════════════════════════════════════════════
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "numpy<2.0",
    "torch==2.1.0",
    "torchvision==0.16.0",
    "scikit-learn",
    "pandas",
    "tqdm",
    "pillow",
    "matplotlib",
])

app = modal.App("glaam4x-attention", image=image)

# Modal volumes
data_volume = modal.Volume.from_name("cataract-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("cataract-checkpoints", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════
# Constants (must match training)
# ═══════════════════════════════════════════════════════════════
DISEASE_NAMES = ['Cataract', 'DR', 'Glaucoma', 'Myopia']
IMG_SIZE = 384
MODEL_NAME = "glaam4x_v6_winning_recipe"


@app.function(
    gpu="T4",
    cpu=8,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=3600,
    memory=16384,
)
def generate_attention_maps(n_samples: int = 12):
    """
    Load the trained GLAAM-4X, run inference on test images, and save
    disease-specific attention map visualizations.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import json
    import os
    from pathlib import Path
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from torchvision.transforms import v2
    from tqdm import tqdm
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ═══════════════════════════════════════════════════════════════
    # LOAD TEST DATA
    # ═══════════════════════════════════════════════════════════════
    DATA_ROOT = Path("/data")
    RAW_ROOT = DATA_ROOT / "raw"
    test_csv = DATA_ROOT / "test_v4.csv"
    test_df = pd.read_csv(test_csv)

    def resolve_path(p):
        if os.path.isabs(p):
            return p
        full = RAW_ROOT / p
        if full.exists():
            return str(full)
        full2 = DATA_ROOT / p
        if full2.exists():
            return str(full2)
        return str(full)

    test_df['image_path'] = test_df['image_path'].apply(resolve_path)
    test_df = test_df[test_df['image_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"  Test set: {len(test_df)} images")

    # ═══════════════════════════════════════════════════════════════
    # TRANSFORMS
    # ═══════════════════════════════════════════════════════════════
    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ═══════════════════════════════════════════════════════════════
    # GLAAM-4X MODEL (must match trained architecture)
    # ═══════════════════════════════════════════════════════════════
    class GlobalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid())
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            return self.fc(y).view(b, c, 1, 1)

    class LocalAttentionBranch(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.BatchNorm2d(in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid())
        def forward(self, x):
            return self.conv(x)

    class GLAAMBlock(nn.Module):
        def __init__(self, in_channels, reduction=16, use_residual=True):
            super().__init__()
            self.global_branch = GlobalAttentionBranch(in_channels, reduction)
            self.local_branch = LocalAttentionBranch(in_channels, reduction)
            self.use_residual = use_residual
            self.alpha = nn.Parameter(torch.tensor(0.5))
        def forward(self, x, return_attention=False):
            gw = self.global_branch(x)
            lw = self.local_branch(x)
            combined = self.alpha * gw + (1 - self.alpha) * lw
            out = x * combined
            result = (x + out) if self.use_residual else out
            if return_attention:
                return result, combined
            return result

    class MultiScaleGLAAM(nn.Module):
        def __init__(self, in_channels, reduction=4):
            super().__init__()
            self.attention_fine = GLAAMBlock(in_channels, reduction=reduction)
            self.attention_medium = GLAAMBlock(in_channels, reduction=reduction * 2)
            self.attention_coarse = GLAAMBlock(in_channels, reduction=reduction * 4)
            self.scale_fusion = nn.Sequential(
                nn.Conv2d(in_channels * 3, in_channels, 1),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
            self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        def forward(self, x, return_attention=False):
            B, C, H, W = x.shape
            fine = self.attention_fine(x)
            medium = self.attention_medium(F.avg_pool2d(x, 2))
            medium_up = F.interpolate(medium, size=(H, W), mode='bilinear', align_corners=False)
            coarse = self.attention_coarse(F.avg_pool2d(x, 4))
            coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)
            combined = torch.cat([fine, medium_up, coarse_up], dim=1)
            output = self.scale_fusion(combined)
            if return_attention:
                return output, (fine + medium_up + coarse_up) / 3
            return output

    class DiseaseGatingNetwork(nn.Module):
        def __init__(self, in_channels, n_diseases=4):
            super().__init__()
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(in_channels, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.2), nn.Linear(256, n_diseases), nn.Softmax(dim=1))
        def forward(self, x):
            return self.gate(x)

    GLAAM_DISEASE_ORDER = ['DR', 'Glaucoma', 'Cataract', 'Myopia']
    REORDER_IDX = [2, 0, 1, 3]

    class GLAAM_4X(nn.Module):
        def __init__(self, pretrained=True, dropout_rate=0.3):
            super().__init__()
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            self.backbone = mobilenet.features
            self.attention_heads = nn.ModuleDict({
                'DR': MultiScaleGLAAM(1280, reduction=4),
                'Glaucoma': GLAAMBlock(1280, reduction=8),
                'Cataract': GLAAMBlock(1280, reduction=16),
                'Myopia': GLAAMBlock(1280, reduction=32),
            })
            self.disease_gate = DiseaseGatingNetwork(1280, n_diseases=4)
            self.classifiers = nn.ModuleDict({
                'DR': nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)),
                'Glaucoma': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
                'Cataract': nn.Sequential(nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(dropout_rate*0.5), nn.Linear(128, 1)),
                'Myopia': nn.Sequential(nn.Linear(1280, 64), nn.ReLU(), nn.Linear(64, 1)),
            })
        def forward(self, x, return_attention=False):
            features = self.backbone(x)
            specialist_features = {}
            attention_maps = {}
            for disease in GLAAM_DISEASE_ORDER:
                if disease in self.attention_heads:
                    if return_attention:
                        attended, attn = self.attention_heads[disease](features, return_attention=True)
                        attention_maps[disease] = attn
                    else:
                        attended = self.attention_heads[disease](features)
                else:
                    attended = features
                specialist_features[disease] = F.adaptive_avg_pool2d(attended, 1).flatten(1)
            logits = []
            for disease in GLAAM_DISEASE_ORDER:
                logits.append(self.classifiers[disease](specialist_features[disease]).squeeze(-1))
            return {'logits': torch.stack(logits, dim=1), 'features': None,
                    'attention_maps': attention_maps}

    model = GLAAM_4X(pretrained=True, dropout_rate=0.3).to(device)

    # ═══════════════════════════════════════════════════════════════
    # LOAD TRAINED WEIGHTS
    # ═══════════════════════════════════════════════════════════════
    RUN_DIR = Path("/checkpoints") / MODEL_NAME
    ckpt_path = RUN_DIR / "best_model.pth"
    if not ckpt_path.exists():
        # Fall back to the A1 ablation model
        ckpt_path = Path("/checkpoints/publication_analysis_v2/ablation/A1_Full_GLAAM4X_model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No trained model found at {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  ✅ Loaded model from {ckpt_path}")

    # ═══════════════════════════════════════════════════════════════
    # SELECT SAMPLE IMAGES (one per disease, positive cases)
    # ═══════════════════════════════════════════════════════════════
    samples = []
    for disease in DISEASE_NAMES:
        pos = test_df[test_df[disease] == 1]
        if len(pos) > 0:
            samples.append(pos.iloc[0])
    # Add a few more diverse samples
    for i in range(max(0, n_samples - len(samples))):
        samples.append(test_df.iloc[i])

    out_dir = RUN_DIR / "attention_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # GENERATE ATTENTION MAPS
    # ═══════════════════════════════════════════════════════════════
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    for idx, row in enumerate(samples):
        img_path = row['image_path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor, return_attention=True)
            logits = out['logits'][0].cpu().numpy()
            attention_maps = out['attention_maps']

        probs = 1 / (1 + np.exp(-logits))

        # Create figure: original + 4 disease attention maps
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original\n{row['source']}", fontsize=10)
        axes[0].axis('off')

        for j, disease in enumerate(DISEASE_NAMES):
            # Map disease name to GLAAM order key
            glaam_key = {'Cataract': 'Cataract', 'DR': 'DR',
                         'Glaucoma': 'Glaucoma', 'Myopia': 'Myopia'}[disease]
            attn = attention_maps.get(glaam_key)
            if attn is not None:
                attn_map = attn[0].mean(dim=0).cpu().numpy()  # average over channels
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                # Resize to image size
                attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(img.size))
                axes[j+1].imshow(img)
                axes[j+1].imshow(attn_resized, cmap='jet', alpha=0.5)
                axes[j+1].set_title(f"{disease}\nP={probs[j]:.2f}", fontsize=10, color=colors[j])
            else:
                axes[j+1].imshow(img)
                axes[j+1].set_title(f"{disease}\n(no attn)", fontsize=10)
            axes[j+1].axis('off')

        fig.suptitle(f"GLAAM-4X Disease-Specific Attention Maps (Sample {idx})", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"attention_sample_{idx}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved attention map for sample {idx} ({row['source']})")

    checkpoint_volume.commit()
    print(f"\n  ✅ Attention maps saved to {out_dir}/")
    return json.dumps({"saved": len(samples), "dir": str(out_dir)})


@app.local_entrypoint()
def main(n_samples: int = 12):
    import json as _json
    result = _json.loads(generate_attention_maps.remote(n_samples=n_samples))
    print(f"\n🎉 Attention maps generated: {result['saved']} samples")
    print(f"   Saved to: {result['dir']}")
