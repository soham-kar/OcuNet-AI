# GLAAM-4X v2 — Implementation Plan

**Scope:** Tier 1 (retinal pretraining, snake conv for DR, higher resolution for glaucoma) + Tier 2 Option 4 (cross-scale attention in gating network).

**Current baseline (measured, old data):** Macro F1 0.7144, AUCs: Cataract 0.944, DR 0.887, Glaucoma 0.880, Myopia 0.977.

---

## 1. Retinal-Specific Pretraining (Tier 1 — highest impact)

### What it is
Your MobileNetV2 backbone is ImageNet-pretrained. ImageNet features (dogs, cars) are suboptimal for fundus images (vessels, lesions, optic disc). Retinal pretraining learns fundus-specific features first, then fine-tunes on the classification task.

### Why it helps
- nnMobileNet++ (2025) showed retinal pretraining significantly improves fundus performance.
- Directly targets **Glaucoma (AUC 0.88)** — the optic-disc features are the most "un-ImageNet-like."
- You already have ~32k fundus images (unlabeled) to pretrain on.

### Implementation approach (self-supervised, no labels needed)

**Option A — SimCLR (contrastive learning):**
1. Take your ~32k fundus images (no labels needed).
2. Apply two random augmentations to each image → positive pair.
3. Train a projection head to pull positive pairs together, push negatives apart.
4. Discard the projection head, keep the backbone.
5. Fine-tune the backbone + GLAAM-4X heads on the classification task.

**Option B — MAE (Masked Autoencoder):**
1. Mask 75% of each fundus image.
2. Train the backbone to reconstruct the masked patches.
3. Keep the encoder, fine-tune on classification.

**Recommendation:** **SimCLR** is simpler and well-established for medical imaging. MAE is more compute-heavy.

### Files to create/modify
- **New:** `modal_pretrain_retinal.py` — self-supervised pretraining on Modal
- **Modify:** `modal_publication_analysis.py` — load the pretrained backbone instead of ImageNet

### Effort: Medium (1 new script + backbone loading change)

---

## 2. Dynamic Snake Convolution for DR (Tier 1)

### What it is
Standard convolutions use fixed square kernels. **Snake convolution** uses deformable, snake-shaped kernels that follow elongated structures (like blood vessels). DR lesions (microaneurysms, hemorrhages) cluster along vessels, so snake conv captures them better.

### Why it helps
- nnMobileNet++ (2025) uses snake conv for "elongated vascular patterns."
- Directly targets **DR (AUC 0.887)** — the vascular-lesion disease.

### Implementation approach
Add a **snake convolution branch** to the DR attention head (MultiScaleGLAAM):

```python
class SnakeConv(nn.Module):
    """Deformable snake-shaped convolution for elongated vascular structures."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Learnable offsets for the snake path
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    def forward(self, x):
        offset = self.offset_conv(x)
        # Apply deformable convolution with learned offsets
        return torchvision.ops.deform_conv2d(x, offset, self.conv.weight, self.conv.bias)
```

Then integrate into the DR head:
```python
class MultiScaleGLAAM(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.snake = SnakeConv(in_channels, in_channels)  # NEW: vascular branch
        self.attention_fine = GLAAMBlock(in_channels, reduction=reduction)
        # ... existing scales ...
    def forward(self, x):
        x_vascular = self.snake(x)  # NEW: snake-conv features
        # Fuse snake features with multi-scale attention
        ...
```

### Files to modify
- `modal_publication_analysis.py` — add `SnakeConv` class + integrate into `MultiScaleGLAAM`
- `models/glaam_4x.py` — mirror the change for consistency

### Effort: Medium (new module + integration)

---

## 3. Higher Input Resolution for Glaucoma (Tier 1)

### What it is
The optic disc (glaucoma's key region) is small. At 384×384, the disc is ~50px. At **512×512**, it's ~67px — more detail for the glaucoma head.

### Why it helps
- Directly targets **Glaucoma (AUC 0.88)** — the weakest disease.
- The optic disc is a small, localized region that benefits from higher resolution.

### Implementation approach
1. Change `IMG_SIZE = 384` → `IMG_SIZE = 512` in `modal_publication_analysis.py`.
2. The MobileNetV2 backbone handles variable input sizes (fully convolutional).
3. The attention heads and classifiers are resolution-agnostic (they use adaptive pooling).

### Trade-off
- Higher resolution = more compute (slower training, more VRAM).
- Batch size may need to drop from 64 → 32 to fit in T4 VRAM.

### Files to modify
- `modal_publication_analysis.py` — change `IMG_SIZE` constant

### Effort: Low (one constant change + batch size adjustment)

---

## 4. Cross-Scale Attention in Gating Network (Tier 2, Option 4)

### What it is
Your current `DiseaseGatingNetwork` uses global average pooling → MLP. This collapses all spatial info into a single vector. **Cross-scale attention** lets the gate consider multi-scale disease evidence (fine lesions vs. coarse structures).

### Why it helps
- The gate currently can't distinguish "small DR lesion" from "large optic disc" — both become a single pooled value.
- Cross-scale attention gives the gate richer, multi-resolution context for routing.

### Implementation approach
Replace the simple GAP+MLP gate with a multi-scale gate:

```python
class CrossScaleGatingNetwork(nn.Module):
    def __init__(self, in_channels, n_diseases=4):
        super().__init__()
        # Multi-scale pooling: fine, medium, coarse
        self.pool_fine = nn.AdaptiveAvgPool2d(1)
        self.pool_med = nn.AdaptiveAvgPool2d(2)   # 2x2
        self.pool_coarse = nn.AdaptiveAvgPool2d(4) # 4x4
        # Cross-scale attention
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 3, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, n_diseases), nn.Softmax(dim=1))
    def forward(self, x):
        f_fine = self.pool_fine(x).flatten(1)
        f_med = self.pool_med(x).flatten(1)
        f_coarse = self.pool_coarse(x).flatten(1)
        # Cross-scale attention over the three scales
        scales = torch.stack([f_fine, f_med, f_coarse], dim=0)  # (3, B, C)
        attended, _ = self.attn(scales, scales, scales)
        # Concatenate and route
        combined = torch.cat([attended[0], attended[1], attended[2]], dim=1)
        return self.mlp(combined)
```

### Files to modify
- `modal_publication_analysis.py` — replace `DiseaseGatingNetwork` with `CrossScaleGatingNetwork`

### Effort: Medium (new module + swap)

---

## Implementation Order (recommended)

| Step | Change | Effort | Expected impact |
|------|--------|--------|-----------------|
| 1 | Higher resolution (384→512) | Low | Glaucoma ↑ |
| 2 | Snake conv for DR | Medium | DR ↑ |
| 3 | Cross-scale gating | Medium | Routing ↑ |
| 4 | Retinal pretraining | Medium | All diseases ↑ |

**Rationale:** Do the low-effort, high-impact changes first (resolution), then the architectural additions (snake conv, cross-scale gating), then the pretraining (which requires a separate pretraining run).

---

## ⚠️ Critical prerequisite

**Before implementing any of these, you must complete the current retrain on the new balanced data.** The new data (glaucoma +165%) may already improve Glaucoma significantly — you need that baseline to measure the improvements against.

---

## Ablation plan for v2

To prove each improvement works, add these ablation variants:
- **V1:** Baseline (current GLAAM-4X on new data)
- **V2:** + higher resolution (512)
- **V3:** + snake conv for DR
- **V4:** + cross-scale gating
- **V5:** + retinal pretraining (full v2)

This gives a clean ablation showing each component's contribution.

---

## Summary

| Improvement | Target disease | Novelty | Effort |
|-------------|---------------|---------|--------|
| Retinal pretraining | All (esp. Glaucoma) | 🟡 (nnMobileNet++ did it) | Medium |
| Snake conv for DR | DR | 🟢 (novel for fundus DR) | Medium |
| Higher resolution | Glaucoma | 🟡 (standard) | Low |
| Cross-scale gating | Routing | 🟢 (novel) | Medium |

**Most novel:** Snake conv for DR + cross-scale gating (both are genuinely new for fundus classification).
**Highest impact:** Retinal pretraining (helps everything).
