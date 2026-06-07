# Updated Independent Analysis: OcuNet-AI Future Roadmap
## Based on Full Paper Review (10 papers analyzed)

> **Date:** 2026-06-07 | **Status:** Updated after reading all provided papers
> **Correction:** My earlier claim that nnMobileNet++ "doesn't exist" was WRONG. It's arxiv 2512.01273v1 (Dec 2025). I apologize.

---

## What Each Paper Actually Says

### 1. nnMobileNet++ (Li et al., 2025) — arxiv 2512.01273v1
**Venue:** Preprint (Dec 2025) | **Institution:** Arizona State + Mayo Clinic

**What it actually does:**
- Extends nnMobileNet (a lightweight CNN for retinal images) by adding transformer blocks
- Three components: (i) Dynamic Snake Convolution for boundary-aware features, (ii) stage-specific transformer blocks after 2nd down-sampling, (iii) retinal image pretraining
- Evaluated on MURED dataset for classification
- Claims SOTA while maintaining low computational cost

**Key insight for your project:** This is the closest existing work to your Phase 3 idea. They already did "MobileNet + Transformer for retinal images." Your differentiation must be clear.

### 2. Hybrid CNN-Transformer for Inherently Interpretable Detection (Djoumessi et al., 2025)
**Venue:** arxiv 2504.08481v4 (Sep 2025) | **Institution:** University of Tübingen

**What it actually does:**
- "Interpretable-by-design" hybrid CNN-Transformer
- Generates class-specific sparse evidence maps in a SINGLE forward pass
- Uses "Dual-Resolution Self-Attention"
- Claims: "Unlike widely used post-hoc saliency methods for ViTs, our approach generates faithful and localized evidence maps that directly reflect the model's decision process"
- Evaluated on retinal fundus images

**Key insight:** This paper ALREADY claims "inherently interpretable" hybrid CNN-Transformer for retinal disease detection. Published Sep 2025. Your Phase 3 novelty claim needs to differentiate from this specifically.

### 3. Novel Transformer-CNN Hybrid for 9 Eye Diseases (Rieck et al., 2025)
**Venue:** IEEE Access (Sep 2025) | **Institution:** Helmut-Schmidt-University, Germany

**What it actually does:**
- EfficientNet + Swin Transformer V2 hybrid
- 9 diseases + healthy class
- 81.91% balanced accuracy, DR TPR 88.93%, Glaucoma TPR 61.23%
- Stratified 5-fold cross-validation
- Targets: CSCR, DR, disc edema, glaucoma, macular scar, myopia, pterygium, retinal detachment, RP

**Key insight:** Directly targets multi-disease classification. Uses EfficientNet (not MobileNet). Their glaucoma TPR of 61.23% shows how hard glaucoma is.

### 4. AdaptiveSwin-CNN (Qureshi, 2025)
**Venue:** MDPI AI (Feb 2025) | **Institution:** Imam Mohammad Ibn Saud Islamic University

**What it actually does:**
- Swin Transformer + CNN with Self-Attention Fusion Module (SAFM)
- 98.89% accuracy on RFMiD and ODIR benchmarks
- Uses XGBoost ensemble for interpretability
- Addresses class imbalance

**Key insight:** Claims very high accuracy on YOUR datasets (RFMiD + ODIR). This is a direct benchmark comparison target.

### 5. Retinal Fundus Multi-Disease Ensemble (Singh et al., 2025)
**Venue:** arxiv 2503.21465v1 (Mar 2025) | **Institution:** NISER, India

**What it actually does:**
- Hybrid CNN-Transformer ensemble for 20 disease labels
- C-Tran ensemble: 0.9166 score
- Uses IEViT model
- Dynamic patch extraction

**Key insight:** 20-disease classification. Shows the field is moving toward broader coverage.

### 6. Focused Attention in Transformers (Playout et al., 2022)
**Venue:** Medical Image Analysis (2022) | **Institution:** Polytechnique Montréal

**What it actually does:**
- Proposes "Focused Attention" — iterative conditional patch resampling
- Produces high-resolution attribution maps from ViTs
- Validated by 4 retinal specialists
- Compared ViTs vs CNNs for interpretability

**Key insight:** Established that transformer attention CAN be clinically meaningful. But requires specialized techniques (Focused Attention), not just raw attention weights.

### 7. GLAAM and GLAAI (Kumar et al., 2025)
**Venue:** Computer Methods and Programs in Biomedicine Update (2025)

**What it actually does:**
- Your foundation paper
- MobileNet + attention for cataract detection
- 97.08% balanced accuracy
- Uses Grad-CAM for interpretability
- Single-disease focus (cataract only)

**Key insight:** Your project extends this from single-disease (cataract) to multi-disease (4 diseases). This IS a genuine contribution.

### 8. LGSF-Net — NOT RELEVANT
This paper is about deepfake detection, not retinal images. Should be removed from the reference list.

### 9-10. Self-Attention-Enhanced Framework + Explainable Hybrid CNN-ViT (Rice)
The rice seed paper is agricultural, not medical. The self-attention framework PDF was unreadable.

---

## Revised Assessment of Your Roadmap

### Phase 1: ASL + Data Split Reorganization
**Verdict: ✅ STRONG — Unchanged from previous analysis**

ASL remains the right choice. None of the reviewed papers use it for retinal images — this could be a minor novelty point.

### Phase 2: EfficientNet-B3
**Verdict: 🟡 REASONABLE — But Rieck et al. already did this**

The Rieck paper (IEEE Access, Sep 2025) already used EfficientNet + Swin Transformer for 9 diseases. If you go EfficientNet-B3, you're following an established path, not breaking new ground. Still worth doing for performance, but not novel.

### Phase 3: GLAAM-ViT — MAJOR REVISION NEEDED

**This is the critical update.** After reading the papers:

#### What's already published:
| Paper | What they did | When |
|-------|--------------|------|
| Djoumessi et al. | "Interpretable-by-design" hybrid CNN-Transformer for retinal images | Sep 2025 |
| Li et al. (nnMobileNet++) | MobileNet + Transformer hybrid for retinal images | Dec 2025 |
| Rieck et al. | EfficientNet + Swin Transformer for 9 eye diseases | Sep 2025 |
| Qureshi (AdaptiveSwin-CNN) | Swin + CNN with fusion for RFMiD/ODIR | Feb 2025 |
| Singh et al. | CNN-Transformer ensemble for 20 diseases | Mar 2025 |

**The landscape has shifted.** In 2025 alone, at least 5 papers published hybrid CNN-Transformer architectures for multi-disease retinal classification. The space is getting crowded.

#### What's still genuinely open:

1. **Disease-specific attention heads with GLAAM heritage** — None of these papers use GLAAM-style attention. They use generic transformers (Swin, ViT) or standard self-attention. Your GLAAM heritage (from Kumar et al.) is unique.

2. **Multi-disease GLAAM** — Extending GLAAM from single-disease (cataract) to multi-disease (4 diseases) with disease-specific attention heads. This is NOT done by any of the reviewed papers.

3. **Efficiency comparison** — nnMobileNet++ claims efficiency but doesn't compare against GLAAM-style attention. You could show GLAAM-ViT is more efficient than pure transformers while matching their accuracy.

4. **Explainability comparison** — Djoumessi claims "inherently interpretable." You could compare GLAAM attention maps against their evidence maps, showing which is more clinically meaningful.

#### Reframed Phase 3:

Instead of "GLAAM-ViT" as a novel architecture, frame it as:

**"GLAAM-4X: Extending Disease-Specific Attention from Single-Disease to Multi-Disease Fundus Classification"**

Your genuine contributions:
1. **Extension**: GLAAM was single-disease (cataract). You extend to 4 diseases.
2. **Architecture**: Disease-specific attention heads (MultiScaleGLAAM for DR, GLAAMBlock for others) — different from generic transformer attention
3. **Efficiency**: MobileNet backbone keeps it lightweight vs EfficientNet/Swin competitors
4. **Benchmark**: First systematic comparison of GLAAM-style attention vs transformer attention for multi-disease fundus classification

This is a MUCH stronger story than "CNN-Transformer hybrid" (which 5 papers already did in 2025).

---

## Revised Success Probability

| Outcome | Previous | Revised | Reason |
|---------|----------|---------|--------|
| DR F1 > 0.62 (Phase 1) | 40% | 40% | Unchanged |
| DR F1 > 0.65 (Phase 1+2) | 55% | 55% | Unchanged |
| Publishable at MICCAI | 35% | **45%** | Clearer novelty story with GLAAM extension framing |
| Publishable at mid-tier | 65% | **70%** | Strong benchmark comparisons available |

---

## Revised Priority Order

### Immediate (This Week):
1. ✅ Implement ASL + data split reorganization
2. ✅ Add macula auxiliary branch
3. ✅ Read Djoumessi and nnMobileNet++ papers in full detail

### Short-term (1-2 weeks):
4. 🔄 Evaluate Phase 1 results
5. 🔄 Run AdaptiveSwin-CNN on your dataset as a baseline comparison
6. 🔄 Draft the "GLAAM extension" narrative for your paper

### Medium-term (2-4 weeks):
7. ⏳ EfficientNet-B3 only if DR still below target
8. ⏳ Implement GLAAM vs Transformer comparison experiments
9. ⏳ Write paper introduction + related work

---

## Bottom Line

**I was wrong about nnMobileNet++ not existing** — it does (arxiv 2512.01273v1). But the bigger finding is that **5 papers in 2025 alone** have published hybrid CNN-Transformer architectures for multi-disease retinal classification. This makes your original Phase 3 framing ("CNN-Transformer hybrid") non-novel.

**However**, your project has a genuine unique angle: **extending GLAAM from single-disease to multi-disease classification**. None of the 2025 papers use GLAAM-style attention. They all use generic transformers. Your disease-specific attention heads (MultiScaleGLAAM for DR, GLAAMBlock for glaucoma/cataract) are architecturally different from standard self-attention.

**Reframe your contribution as "GLAAM-4X: Multi-Disease Extension of Disease-Specific Attention" rather than "yet another CNN-Transformer hybrid."** This is both honest and compelling.
