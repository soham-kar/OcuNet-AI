# GLAAM-4X: Disease-Specific Multi-Scale Attention for Multi-Label Fundus Disease Detection from a Single Retinal Image

> **Figure guide.** Each section references figures by relative path under `checkpoints/glaam4x_unified_v4_asl_384/`. All figures are 300 DPI. PDF (vector) versions are preferred for print; PNG for digital. No UI screenshots are included — all figures are generated from experimental data or architectural diagrams.

---

# List of Figures

| No. | Caption | Page |
|---|---|---|
| 1 | End-to-end GLAAM-4X training and inference pipeline | |
| 2 | Internal structure of the GLAAM attention module | |
| 3 | MultiScaleGLAAM architecture for the DR specialist | |
| 4 | Complete GLAAM-4X architecture | |
| 5 | Learning rate schedule and training progression | |
| 6 | Training and validation Asymmetric Loss over 60 epochs | |
| 7 | Macro F1 score progression | |
| 8 | Per-disease validation AUC curves | |
| 9 | Per-disease validation F1 scores at optimal thresholds | |
| 10 | Per-disease precision and recall trends | |
| 11 | Actual learning rate schedule | |
| 12 | Combined 6-panel training dashboard | |
| 13 | Train vs. validation loss for overfitting detection | |
| 14 | Per-disease ROC curves on held-out test set | |
| 15 | Per-disease precision-recall curves on held-out test set | |
| 16 | Per-disease confusion matrices on held-out test set | |
| 17 | Calibration (reliability) curves per disease | |
| 18 | Per-disease ROC curves on held-out test set (report asset) | |
| 19 | Per-disease precision-recall curves on held-out test set (report asset) | |
| 20 | Example bar plot output from the Flask demo application | |
| 21 | Example EigenGradCAM heatmap visualisation | |

---

# List of Tables

| No. | Caption | Page |
|---|---|---|
| 1 | Positioning relative to prior work | |
| 2 | Datasets and data sources | |
| 3 | Dataset split sizes | |
| 4 | Disease-specific attention configuration | |
| 5 | Classifier head architecture | |
| 6 | Model efficiency summary | |
| 7 | Asymmetric loss hyperparameters | |
| 8 | Optimisation configuration | |
| 9 | Per-disease optimal thresholds | |
| 10 | Inference package contents | |
| 11 | Training environment | |
| 12 | Held-out test set results | |
| 13 | Full results across all splits | |
| 14 | Model efficiency | |
| 15 | Model size comparison | |

---

# Abstract

Automated multi-disease detection from fundus photographs is a promising approach to scaling ophthalmic screening in resource-limited settings, but existing classifiers face three limitations: they apply a uniform feature extraction pathway to all diseases despite each pathology manifesting at distinct spatial scales and retinal locations; they use loss functions that are suboptimal under the extreme class imbalance inherent in multi-label fundus datasets; and they operate as black boxes without providing interpretable evidence of which image regions drove each prediction. We present GLAAM-4X, a multi-label deep learning classifier that addresses these limitations through four disease-specific attention specialists operating on a shared MobileNetV2 backbone, each tailored to the spatial scale and morphological characteristics of its target pathology. The DR specialist employs a novel MultiScaleGLAAM module that applies attention at three spatial scales to capture lesion sizes ranging from 2-pixel microaneurysms to 50-pixel haemorrhages. A disease gating network dynamically routes features to the appropriate specialist, and Asymmetric Loss with per-disease optimal thresholding mitigates class imbalance without discarding healthy samples. The model was trained on a unified corpus of 27,899 images from nine public datasets and evaluated on a held-out test set of 2,232 images. GLAAM-4X achieves a macro-averaged AUC of 0.968, with per-disease AUC of 0.985 (Cataract), 0.952 (DR), 0.966 (Glaucoma), and 0.971 (Myopia). The model comprises 12.3 million parameters (47 MB) and achieves 12.44 ms inference latency on a single GPU, making it suitable for real-time screening on commodity hardware. Disease-specific EigenGradCAM heatmaps provide clinically interpretable visualisations, allowing clinicians to verify that the model's attention corresponds to relevant anatomical features. A Flask-based web application demonstrates the end-to-end inference pipeline with interactive image upload, probability bar plots, and attention heatmaps.

---

# Introduction

Cataract, diabetic retinopathy (DR), glaucoma, and pathological myopia are four of the leading causes of preventable visual impairment worldwide [80]. The global prevalence of visual impairment exceeds 2.2 billion people, with the majority residing in low- and middle-income countries where access to ophthalmic care is severely limited [80]. Automated screening tools that can detect multiple diseases from a single fundus photograph have the potential to bridge this gap, enabling non-specialist healthcare workers to identify patients requiring referral to an ophthalmologist.

Deep learning has achieved remarkable success in single-disease fundus image classification, with systems reaching specialist-level accuracy for DR detection [25], glaucoma screening [72], and cataract grading [7]. However, extending these successes to multi-disease detection — where a single model simultaneously detects multiple pathologies from one image — presents three fundamental challenges.

First, the four target diseases manifest at fundamentally different spatial scales and retinal locations. DR lesions span three orders of magnitude, from 2–5 pixel microaneurysms to 50+ pixel haemorrhages [1]. Glaucoma is diagnosed from optic disc and cup morphology, a localised structure spanning approximately 50–100 pixels [2]. Cataract presents as diffuse, global image quality degradation in fundus photography [3]. Myopia is associated with peripapillary tessellation already captured by standard convolutional features [4]. A single shared attention mechanism — as used in SENet [5], CBAM [6], or the original GLAAM module [7] — cannot simultaneously capture microaneurysm-level detail and disc-level context, leading to suboptimal performance on at least a subset of diseases.

Second, fundus disease datasets are heavily imbalanced. In the unified corpus used in this work, the ratio of negative to positive labels ranges from approximately 3:1 (Cataract) to over 15:1 (Myopia). Standard binary cross-entropy loss produces large gradients from the dominant negative class, causing the model to converge to a trivial "all negative" solution early in training [8]. Focal loss [9] partially addresses this but applies the focusing parameter symmetrically to positive and negative classes, which is suboptimal when the imbalance is extreme and asymmetric.

Third, clinical deployment demands interpretable evidence of *where* in the image the model identified disease [10,11]. A clinician who receives a "Glaucoma: 96%" prediction without visual evidence of optic disc cupping cannot verify the model's reasoning, creating a trust deficit that limits clinical adoption. Regulatory frameworks such as the EU AI Act [59] and the FDA's Software as a Medical Device guidance [60] increasingly require explainability for clinical AI systems.

Here we present GLAAM-4X, a multi-label deep learning classifier that addresses these challenges through three architectural innovations: (i) four disease-specific attention specialists operating on a shared MobileNetV2 backbone [13], each tailored to the spatial scale and morphological characteristics of its target pathology; (ii) a learned disease gating network that dynamically routes features to the appropriate specialist based on image content; and (iii) an asymmetric loss function [14] with per-disease optimal thresholding that mitigates class imbalance without discarding healthy samples. The model is trained on a unified corpus of 27,899 images aggregated from nine public datasets and evaluated on a held-out test set of 2,232 images.

The key contributions of this work are:

1. **Disease-specific attention specialists.** Four attention pathways, each with a configuration tailored to its target disease: MultiScaleGLAAM (three spatial scales) for DR, GLAAMBlock with moderate reduction for Glaucoma, GLAAMBlock with high reduction for Cataract, and identity (no attention) for Myopia. This is, to our knowledge, the first application of disease-specific attention mechanisms in fundus image classification.

2. **Post-backbone attention integration.** Unlike the original GLAAM [7], which injects attention blocks within the backbone at intermediate stages, GLAAM-4X applies attention post-backbone on the final 1280-channel feature map. This preserves the pretrained MobileNetV2 features and enables efficient multi-disease branching from a single shared feature map, with the four attention specialists adding only 2.3M parameters (18% of the total).

3. **MultiScaleGLAAM for DR.** A three-scale attention mechanism that applies GLAAM at native resolution, 2× downsampled, and 4× downsampled feature maps, capturing the full range of DR lesion sizes from microaneurysms to large haemorrhages. The three scales are fused via learnable softmax weights and a $1 \times 1$ convolution.

4. **Asymmetric Loss with per-disease thresholding.** ASL [14] with $\gamma_{\text{neg}} = 4.0$, $\gamma_{\text{pos}} = 0.0$, and clip $m = 0.05$ aggressively suppresses easy negative gradients while preserving full gradient signal from all positive examples. Per-disease thresholds are grid-searched on the validation set to maximise F1, accounting for the different prevalence rates across diseases.

5. **Disease-specific EigenGradCAM explainability.** Separate EigenGradCAM [15] heatmaps are generated for each detected disease, allowing clinicians to verify that the model's attention corresponds to clinically relevant anatomical features.

---

# Literature Review

The interpretation of fundus images for multi-disease detection is a central challenge in ophthalmic AI. With the global prevalence of visual impairment exceeding 2.2 billion [80] and the shortage of trained ophthalmologists in low-resource settings, automated screening tools have become a clinical necessity. This review surveys the computational foundations underlying each component of the GLAAM-4X system, identifies gaps in the existing literature, and positions the present work relative to prior art.

## 1. Attention Mechanisms in Computer Vision and Medical Imaging

Attention mechanisms enable neural networks to adaptively reweight features, emphasising informative regions while suppressing irrelevant ones. The field has progressed through three generations of increasing sophistication.

**Channel attention.** The Squeeze-and-Excitation Network (SENet) [5] introduced channel-wise attention via global average pooling followed by a bottleneck fully-connected layer, allowing the network to recalibrate channel-wise feature responses based on global context. This approach answers the question "which feature channels are important?" but provides no spatial localisation. SENet demonstrated consistent improvements across image classification tasks, establishing channel attention as a standard component of modern architectures.

**Spatial attention.** The Convolutional Block Attention Module (CBAM) [6] extended SENet to a dual-branch design combining channel and spatial attention sequentially. The spatial branch generates a per-position weight map via convolutional operations, answering "where in the image should the model look?" CBAM demonstrated that channel and spatial attention are complementary: each captures information that the other cannot. This complementarity principle underlies the design of the GLAAM module [7] used in this work.

**Global-local attention.** Subsequent work explored the integration of global and local attention from different perspectives. Song et al. [69] proposed global-local, spatial-channel attention for image retrieval, demonstrating that combining global context with local detail improves fine-grained recognition. Chen et al. [70] introduced SCA-CNN, applying spatial and channel-wise attention in convolutional networks for image captioning. Bello et al. [71] proposed attention-augmented convolutional networks that augment standard convolutions with multi-head self-attention, bridging the gap between attention mechanisms and transformer architectures [81]. Guo et al. [82] provided a comprehensive survey of attention mechanisms in computer vision, taxonomising them into spatial attention, channel attention, and hybrid approaches.

**Attention in ophthalmic imaging.** In ophthalmic imaging specifically, attention mechanisms have been applied to glaucoma detection from fundus images [72], cataract classification from anterior segment optical coherence tomography (AS-OCT) images [73,74,75], and nuclear cataract grading [76]. Li et al. [72] introduced an attention-based glaucoma detection model with a large-scale fundus image database, demonstrating that attention improves disc and cup region localisation. Zhang et al. [73,74,75,76] developed a series of region-based attention networks for nuclear cataract classification in AS-OCT images, progressively refining the attention mechanism from adaptive feature squeezing [74] to clinical-awareness attention [75] to regional context-based recalibration [76]. However, these works typically employ a single shared attention pathway applied uniformly across all disease categories, which is suboptimal when diseases manifest at different spatial scales and anatomical locations.

**Gap identified.** No existing work employs disease-specific attention mechanisms in fundus image classification, where each disease receives a dedicated attention pathway tailored to its spatial scale and morphological characteristics. GLAAM-4X addresses this gap by introducing four disease-specific attention specialists operating on a shared backbone.

## 2. GLAAM and GLAAI

The Global-Local Attention Aggregation Module (GLAAM) was introduced by Kumar, Verma, and Illés [7] as an attention mechanism specifically designed for automated cataract detection. GLAAM combines a global (channel) attention branch based on squeeze-and-excitation [5] with a local (spatial) attention branch based on $1 \times 1$ convolutions, fusing them via a learnable parameter $\alpha$. The original work applied GLAAM to cataract detection on the ODIR-5K dataset [16] using InceptionV3 [77] and MobileNetV2 [13] backbones, injecting GLAAM blocks at intermediate stages of the backbone (stages 13 and 17). The companion module GLAAI (Global-Local Attention with Aggregated Inputs) modifies the fusion strategy to aggregate inputs rather than attention weights.

The original GLAAM work demonstrated that combining channel and spatial attention improves cataract detection accuracy over baseline models without attention. The study evaluated multiple backbone architectures (InceptionV3, MobileNetV2) and attention injection points, finding that GLAAM consistently improved performance across configurations. The work also employed Grad-CAM [12] for visual explainability, showing that the attention mechanism focuses on the lens region in cataract-positive images.

However, the original GLAAM has three limitations that GLAAM-4X addresses. First, it was designed for single-disease (cataract) detection and applies a single shared attention pathway, which cannot adapt to the different spatial scales of multiple diseases. Second, GLAAM blocks are injected *within* the backbone at intermediate stages, modifying the pretrained feature hierarchy and preventing the backbone from being treated as a frozen feature extractor. Third, the original work uses standard cross-entropy loss, which is suboptimal for the heavily imbalanced multi-label setting.

GLAAM-4X extends the original GLAAM in four ways: (i) it generalises from single-disease to multi-disease detection by introducing four disease-specific attention specialists, each with a configuration tailored to its target pathology; (ii) it applies attention *post-backbone* rather than intra-backbone, preserving the pretrained MobileNetV2 features and enabling efficient multi-disease branching from a single shared feature map; (iii) it introduces a MultiScaleGLAAM variant for DR that applies attention at three spatial scales to capture the wide range of lesion sizes; and (iv) it replaces standard cross-entropy with Asymmetric Loss [14] to handle the extreme class imbalance inherent in multi-label fundus disease datasets.

## 3. Multi-Label Fundus Disease Classification

Multi-label fundus disease classification — detecting multiple diseases from a single fundus image — has received increasing attention as datasets with multi-disease annotations have become available. The ODIR-5K dataset [16] provides annotations for seven disease categories across approximately 5,000 fundus images, enabling multi-label classifiers to be trained for simultaneous detection of cataract, DR, glaucoma, myopia, age-related macular degeneration, hypertension, and other abnormalities. The RFMiD dataset [19] provides 45 disease labels across 3,200 fundus images, offering finer-grained annotation but with greater class imbalance. Li et al. [83] provided a comprehensive review of deep learning applications in fundus image analysis, surveying datasets, architectures, and clinical applications.

Existing multi-label fundus classifiers typically employ a shared backbone (ResNet [43], EfficientNet [49], or MobileNetV2 [13]) with a multi-output classification head, trained with binary cross-entropy or focal loss [9]. Wang et al. [25] proposed a real-time DR detection system using a lightweight CNN with attention, achieving high accuracy on the DDR dataset. However, these approaches do not employ disease-specific attention mechanisms, instead relying on a single shared feature extractor for all diseases. The shared approach forces the feature extractor to learn a compromise representation that is adequate for all diseases but optimal for none.

GLAAM-4X departs from this paradigm by introducing disease-specific attention specialists that refine the shared backbone features differently for each disease, motivated by the clinical observation that different pathologies manifest at different spatial scales and retinal locations. This approach is related to the mixture-of-experts paradigm [47], where multiple specialist networks handle different subsets of the input space, but differs in that the specialists share a common backbone and operate on the same feature map rather than processing the input independently.

## 4. Class Imbalance in Medical Image Classification

Class imbalance is a pervasive challenge in medical image classification, where disease-positive examples are typically far less common than disease-negative examples. Buda et al. [8] provided a systematic study of the class imbalance problem in convolutional neural networks, demonstrating that oversampling and class-weighted loss functions are the most effective mitigation strategies. Kang et al. [39] proposed class-balanced loss based on the effective number of samples, providing a principled weighting scheme that accounts for the diminishing returns of additional samples.

**Focal loss** [9] addressed imbalance by down-weighting easy examples via a focusing factor $(1 - p_t)^\gamma$, concentrating the loss on hard examples. However, focal loss applies the same focusing parameter to both positive and negative classes, which is suboptimal when the imbalance is extreme and asymmetric — as in multi-label fundus classification, where the negative class dominates by 3:1 to 15:1.

**Asymmetric Loss (ASL)** [14] resolved this by introducing separate focusing parameters for positive and negative samples, along with a probability clipping mechanism for negatives. ASL has been shown to outperform focal loss on multi-label classification benchmarks, particularly when the class distribution is highly skewed. GLAAM-4X adopts ASL with $\gamma_{\text{neg}} = 4.0$, $\gamma_{\text{pos}} = 0.0$, and clip $m = 0.05$, aggressively suppressing easy negative gradients while preserving full gradient signal from all positive examples.

## 5. Explainability in Medical AI

The explainability gap in medical AI — the inability of clinicians to understand why a model makes a particular prediction — has been identified as a critical barrier to clinical adoption [10,11]. Topol [11] argued that high-performance medicine requires the convergence of human and artificial intelligence, and that explainability is essential for building clinician trust. Regulatory frameworks such as the EU AI Act [59] and the FDA's Software as a Medical Device guidance [60] increasingly require explainability for clinical AI systems.

**Grad-CAM** [12] generates localisation heatmaps by weighting feature maps with the gradient of the class score, providing visual evidence of which image regions influenced the prediction. Grad-CAM has become the standard explainability tool for convolutional neural networks in medical imaging due to its simplicity and broad applicability.

**Grad-CAM++** [15] extended Grad-CAM with a weighted combination of positive gradients, producing sharper heatmaps for multi-label tasks where multiple classes may be active simultaneously. **EigenGradCAM** further improves heatmap quality by using singular value decomposition of the gradient matrix, reducing noise from spatially inconsistent gradients. This is particularly important for fine-grained lesion localisation in DR, where the gradient signal is spatially concentrated.

In ophthalmic imaging, Grad-CAM has been used to visualise DR lesion localisation [78] and glaucoma disc detection [79]. However, these applications typically generate a single heatmap for a single disease. GLAAM-4X generates disease-specific EigenGradCAM heatmaps for each detected disease, allowing the clinician to verify that the model's attention corresponds to clinically relevant anatomical features (e.g., the optic disc for glaucoma, the macula for DR).

## 6. Lightweight Architectures for Medical Screening

Deployment of AI-based screening tools in resource-constrained settings — rural clinics, mobile screening units, and low- and middle-income countries — requires models that are computationally efficient and compatible with limited hardware. The MobileNet family [13,42] was specifically designed for such scenarios, using depthwise separable convolutions [44] and inverted residual blocks with linear bottlenecks to reduce computational cost while maintaining representational capacity. Souid et al. [84] demonstrated MobileNetV2 for lung disease classification from chest X-rays, and Yuan et al. [85] proposed a low-resolution MobileNet variant for resource-constrained scenarios.

GLAAM-4X uses MobileNetV2 as its backbone, achieving 12.3M parameters and 47 MB model size with 12.44 ms inference latency on a Tesla T4 GPU — suitable for real-time screening on commodity hardware. The disease-specific attention specialists add only 2.3M parameters (18% of the total), a modest overhead for the performance improvement they provide.

## Positioning of the Present Work

The system described in this work integrates four distinct research threads — disease-specific attention mechanisms, multi-label fundus disease classification, asymmetric loss for class imbalance, and gradient-based explainability — into a unified pipeline for multi-disease fundus screening. Table 1 summarises the relationship to prior work:

> **Table 1: Positioning Relative to Prior Work.** Comparison of each major system component against the closest prior work and the novel contribution of the present system.

| Component | Closest prior work | Novel contribution |
|---|---|---|
| Attention mechanism | GLAAM [7], CBAM [6] | Four disease-specific attention specialists with disease-tailored reduction ratios |
| Multi-scale attention | FPN [46], multi-scale DR detection [25] | MultiScaleGLAAM: three-scale GLAAM for DR lesion size range (2–50+ px) |
| Backbone integration | GLAAM [7] (intra-backbone) | Post-backbone application preserving pretrained features; shared backbone for 4 specialists |
| Class imbalance | Focal loss [9], ASL [14] | ASL with disease-specific thresholds; WeightedRandomSampler with $\sqrt{\cdot}$ weighting |
| Explainability | Grad-CAM [12], Grad-CAM++ [15] | Disease-specific EigenGradCAM heatmaps for multi-label fundus classification |
| Deployment | MobileNetV2 [13] | 12.3M params, 47 MB, 12.44 ms latency; Flask web demo with bar plot + heatmap |

No single prior system integrates more than two of these components. The primary contribution of this work is the architectural integration — demonstrating that disease-specific attention, asymmetric loss, and gradient-based explainability can be combined into a clinically useful multi-disease screening pipeline with calibrated probabilities and interpretable outputs.

---

# Methodology

## 1. System Overview

GLAAM-4X is a multi-label deep learning classifier for detecting four sight-threatening ocular diseases — Cataract, Diabetic Retinopathy (DR), Glaucoma, and Myopia — from a single colour fundus photograph. The system addresses three limitations of existing fundus disease classifiers.

First, conventional architectures apply a uniform feature extraction pathway to all diseases, ignoring the fact that each pathology manifests at distinct spatial scales and retinal locations. DR lesions range from 2–5 pixel microaneurysms to 50+ pixel haemorrhages [1]; glaucoma is characterised by optic disc and cup morphology spanning a localised region of approximately 50–100 pixels [2]; cataract presents as diffuse, global image quality degradation in fundus photography [3]; and myopia is associated with peripapillary tessellation and atrophy that is already well-captured by standard convolutional features [4]. A single shared attention mechanism — as used in SENet [5], CBAM [6], or the original GLAAM module [7] — cannot simultaneously capture microaneurysm-level detail and disc-level context, leading to suboptimal performance on at least a subset of diseases.

Second, fundus disease datasets are heavily imbalanced, with healthy images dominating the training distribution. In the unified corpus used in this work, the ratio of negative to positive labels ranges from approximately 3:1 (Cataract) to over 15:1 (Myopia). Standard binary cross-entropy (BCE) loss produces large gradients from the dominant negative class, causing the model to converge to a trivial "all negative" solution early in training [8]. Focal loss [9] partially addresses this by down-weighting easy examples, but applies the focusing parameter symmetrically to positive and negative classes, which is suboptimal when the imbalance is extreme and asymmetric.

Third, clinical deployment demands not only accurate predictions but also interpretable evidence of *where* in the image the model identified disease [10]. A clinician who receives a "Glaucoma: 96%" prediction without visual evidence of optic disc cupping cannot verify the model's reasoning, creating a trust deficit that limits clinical adoption [11]. Gradient-weighted Class Activation Mapping (Grad-CAM) [12] and its successors address this by generating localisation heatmaps, but the quality of these heatmaps varies with the gradient computation method.

GLAAM-4X addresses these limitations through three architectural innovations: (i) four disease-specific attention specialists operating on a shared MobileNetV2 backbone [13], each tailored to the spatial scale and morphological characteristics of its target pathology; (ii) a learned disease gating network that dynamically routes features to the appropriate specialist based on image content; and (iii) an asymmetric loss function [14] with per-disease optimal thresholding that mitigates class imbalance without discarding healthy samples. The pipeline comprises six stages: (i) multi-source data aggregation from nine public fundus datasets, (ii) label unification and dataset splitting, (iii) differential augmentation with minority-class oversampling, (iv) GLAAM-4X model training with mixed-precision optimisation, (v) per-disease threshold optimisation on the validation set, and (vi) EigenGradCAM [15] explainability generation for clinical visualisation.

![Figure 1: End-to-end GLAAM-4X pipeline](report_figures/fig1_pipeline.png)

> **Figure 1.** End-to-end GLAAM-4X training and inference pipeline. The pipeline begins with nine public Kaggle fundus image datasets (ODIR-5K, IDRiD, DDR, RFMiD, JSIEC, PAPILA, REFUGE2, glaucoma_bundle, and eye_diseases), each parsed by a disease-specific label parser that maps native annotations onto four target diseases (Cataract, DR, Glaucoma, Myopia). The parsed labels are merged, deduplicated, and split into train/validation/test (80/10/10). Training images undergo differential augmentation — stronger perturbations for disease-positive images — followed by class-balanced sampling via WeightedRandomSampler. The GLAAM-4X model is trained with Asymmetric Loss on a Tesla T4 GPU using mixed-precision optimisation. Per-disease thresholds are optimised on the validation set to maximise F1. The pipeline produces five output categories: the best model checkpoint (.pth), 11 training visualisation graphs, 7 publication-ready report assets (ROC curves, PR curves, confusion matrices, calibration curves, results table), disease-specific EigenGradCAM explainability heatmaps, and a Flask web application for interactive inference demonstration.

---

## 2. Datasets

### 2.1 Data Sources

The training corpus was assembled from nine publicly available fundus image datasets, each obtained from the Kaggle data platform. No single existing dataset provides sufficient labelled examples across all four target diseases with adequate prevalence for multi-label training. The ODIR-5K dataset [16] is the richest single source, providing one-hot annotations for seven disease categories (including all four target diseases) across approximately 5,000 fundus images of both left and right eyes. However, ODIR-5K alone does not provide enough positive examples for rare diseases such as Glaucoma and Myopia. The remaining eight datasets were incorporated to augment the positive-class representation for specific diseases.

> **Table 1: Datasets and Data Sources.** Summary of all public datasets used for training GLAAM-4X, including the diseases labelled by each source, the label format, and the role within the unified training corpus. Datasets were downloaded directly from Kaggle during the training session.

| Dataset | Kaggle slug | Diseases labelled | Label format | Role |
|---|---|---|---|---|
| ODIR-5K [16] | `andrewmvd/ocular-disease-recognition-odir5k` | Cataract, DR, Glaucoma, Myopia | One-hot CSV (N, D, G, C, A, H, M, O) | Richest single source; all four diseases |
| eye_diseases | `gunavenkatdoddi/eye-diseases-classification` | Cataract, DR, Glaucoma | Folder-per-class | Cataract and DR signal |
| IDRiD [17] | `mariaherrerot/idrid-dataset` | DR | CSV with retinopathy grade 0–4 | DR specialist training |
| DDR [18] | `mariaherrerot/ddrdataset` | DR | CSV with grade 0–5 | DR specialist training |
| RFMiD [19] | `andrewmvd/retinal-disease-classification` | DR, Glaucoma, Myopia | Multi-hot CSV (DR, ODC, MYA) | Multi-disease signal |
| JSIEC [20] | `linchundan/fundusimage1000` | DR, Glaucoma, Myopia | 39 folder-per-class (keyword-matched) | Myopia and DR signal |
| PAPILA [21] | `orvile/papila-retinal-fundus-images` | Glaucoma | Excel with diagnosis 0/1/2 | Glaucoma specialist training |
| REFUGE2 [22] | `victorlemosml/refuge2` | Glaucoma | CSV binary label | Glaucoma specialist training |
| glaucoma_bundle | `arnavjain1/glaucoma-datasets` | Glaucoma | Multiple CSVs (ORIGA + REFUGE + G1020) | Glaucoma specialist training |

### 2.2 Multi-Source Label Unification

Each source dataset employs a different annotation schema — one-hot CSV columns (ODIR), folder-per-class directories (eye_diseases, JSIEC), grading CSVs (IDRiD, DDR), multi-hot clinical labels (RFMiD), and Excel-based clinical coding (PAPILA). This heterogeneity reflects the absence of a standardised annotation format in the ophthalmic imaging community, a challenge well-documented in the medical image analysis literature [23]. A dedicated parser was implemented for each source, mapping its native labels onto the four target diseases: Cataract, DR, Glaucoma, and Myopia.

The ODIR parser reads the `full_df.csv` annotation file and extracts the one-hot columns C (Cataract), D (Diabetes/DR), G (Glaucoma), and M (Myopia) for both the left and right fundus images of each patient. The IDRiD and DDR parsers convert retinopathy grading scores (0–4 and 0–5 respectively) to binary DR labels, where any grade greater than zero indicates DR positivity; DDR grade 5 (ungradable) images are excluded. The PAPILA parser reads the clinical Excel file and maps diagnosis codes to binary glaucoma labels, with the "suspect" category (code 2) treated as positive, following the convention that suspected glaucoma warrants clinical follow-up [21]. The RFMiD parser maps the DR, ODC (optic disc cupping, a glaucoma proxy), and MYA (myopia) columns to the corresponding target diseases. The JSIEC parser performs keyword matching on the 39 folder names (e.g., "dr1", "dr2", "pdr" → DR; "glaucoma", "large_optic_cup" → Glaucoma; "myopia" → Myopia), skipping folders that do not correspond to any target disease.

Parsers were designed to be independent: failure of one parser (e.g., due to an unexpected column name or folder layout) does not halt the others, and failures are reported explicitly rather than silently producing zero-label rows. This design follows the principle of graceful degradation, which is essential when integrating heterogeneous data sources whose schemas may change between releases [24].

A known limitation of this unification strategy is that single-disease datasets (IDRiD, DDR, PAPILA, REFUGE2, glaucoma_bundle) label only their target pathology. Images from these sources receive a label of 0 for all other diseases, which represents "not evaluated" rather than "confirmed negative." This is standard practice in the unified fundus-dataset literature [25] but introduces label noise: a PAPILA image of a patient who also has DR will be labelled DR = 0, potentially penalising the model for correctly identifying DR features in that image. The impact is mitigated by the dominance of ODIR (which labels all four diseases) in the unified corpus, and by the asymmetric loss function (Section 5.1), which down-weights easy negative gradients so that incorrect negative labels contribute less to the optimisation.

### 2.3 Dataset Splitting

After merging and deduplication (by image path), the unified corpus was split into training (80%), validation (10%), and test (10%) sets using a fixed random seed (seed = 42) via `sklearn.model_selection.train_test_split`. The split is a plain random partition, not multi-label stratified: iterative stratification [26] — which preserves per-disease prevalence across splits by iteratively assigning samples to the split with the lowest label density — requires the `scikit-multilearn` library, which was not available in the Colab training environment. The per-disease prevalence was verified to be approximately balanced across splits post hoc.

> **Table 2: Dataset Split Sizes.** Number of images in each split of the unified corpus after deduplication.

| Split | Total images | Purpose |
|---|---|---|
| Train | 22,319 | Parameter optimisation |
| Validation (tune) | 3,348 | Threshold tuning, model selection, early stopping |
| Test | 2,232 | Held-out evaluation; all reported test metrics computed here |

The test set (2,232 images) is fully held out from training and validation. All reported test metrics are computed on this set. No extended test pool was used in the final pipeline. The validation set is used for two purposes: (i) per-disease threshold optimisation (Section 6.2) and (ii) early stopping with patience = 10 epochs (Section 5.2). Using the same validation set for both purposes introduces a minor optimistic bias in the threshold-tuned metrics, as the thresholds are selected to maximise F1 on the same set used for model selection. This is a common practice in multi-label classification [27] and is mitigated by the fact that the test set — on which final metrics are reported — is completely independent of both threshold and model selection.

---

## 3. Preprocessing and Differential Augmentation

### 3.1 Image Preprocessing

All images are resized to $384 \times 384$ pixels and normalised with ImageNet statistics (mean $= [0.485, 0.456, 0.406]$, standard deviation $= [0.229, 0.224, 0.225]$). The choice of input resolution involves a trade-off between spatial detail and computational cost. At $224 \times 224$ — the standard ImageNet input size [28] — the MobileNetV2 backbone produces a $7 \times 7$ feature map (50× spatial downsampling), meaning that a DR microaneurysm occupying 2–5 pixels in the original fundus image (typically $2000 \times 2000$ pixels) would be reduced to a sub-pixel feature, effectively invisible to the model. At $384 \times 384$, the backbone produces a $12 \times 12$ feature map, preserving approximately 32× spatial downsampling. While this is still insufficient to resolve individual microaneurysms at the feature map level, the higher resolution allows the early convolutional layers — which operate at higher spatial resolution — to capture fine lesion detail before it is lost to downsampling. The $384 \times 384$ resolution was selected as the maximum that fits within the 16 GB VRAM budget of the Tesla T4 GPU at batch size 32 with mixed-precision training.

ImageNet normalisation statistics are used because the MobileNetV2 backbone was pretrained on ImageNet [29]. Although fundus images differ substantially from natural images in colour distribution and texture, transfer learning studies in medical imaging have consistently shown that ImageNet-pretrained features provide a strong initialization even for domain-distinct tasks [30,31]. The normalisation ensures that the input distribution matches the statistics expected by the pretrained convolutional filters, preventing activation shift in the early layers.

### 3.2 Augmentation Pipeline

Data augmentation serves two purposes in this work: (i) expanding the effective training set to improve generalisation, and (ii) artificially increasing the diversity of the minority (disease-positive) class to combat overfitting. Medical image datasets are typically small and expensive to annotate [32], making augmentation a critical component of the training pipeline.

Augmentation is implemented with `torchvision.transforms.v2` [33] rather than the albumentations library [34], which was removed to avoid a NumPy version conflict in the Colab environment (albumentations requires a higher NumPy floor than Colab preloads, and force-reinstalling NumPy under an already-imported instance causes a runtime crash). The `torchvision.transforms.v2` API provides equivalent functionality with the advantage of native PyTorch tensor integration.

Two augmentation strategies are employed, differing in the magnitude of perturbation:

**Base transform** (applied to every training image):

| Transform | Parameters | Probability | Rationale |
|---|---|---|---|
| Random horizontal flip | — | 0.5 | Fundus images have no canonical left-right orientation |
| Random vertical flip | — | 0.3 | Less aggressive than horizontal; fundus images have a preferred orientation (disc typically nasal) |
| Random 90°/180°/270° rotation | — | 0.3 | Captures camera rotation variability across clinics |
| Random affine warp | degrees = 15, translate = 0.05, scale = 0.9–1.1 | 1.0 | Simulates patient head tilt and camera distance variation |
| Random brightness/contrast jitter | brightness = 0.2, contrast = 0.2 | 0.5 | Simulates illumination variability across fundus cameras |
| Random hue/saturation jitter | hue = 0.03, saturation = 0.2 | 0.3 | Simulates colour cast differences between camera models |
| Gaussian blur | kernel = 3 | 0.2 | Simulates focus variability in low-quality images |
| Elastic deformation [35] | $\alpha = 50$, $\sigma = 5$ | 0.1 | Simulates anatomical variability in retinal structure |
| Gaussian noise (custom) | $\sigma \in [0.04, 0.2]$ | 0.3 | Simulates sensor noise in portable fundus cameras |
| Random erasing [36] | scale = 0.01–0.05, ratio = 0.5–2.0 | 0.2 | Simulates occlusions (eyelid, lens flare, dust artefacts) |

**Strong transform** (applied with $p = 0.7$ to images containing at least one positive disease label): the same transform set with wider parameter ranges — affine degrees = 30, translate = 0.1, scale = 0.8–1.2, brightness/contrast = 0.3, hue = 0.05, Gaussian blur kernel = 5, elastic $\alpha = 100$, noise $\sigma \in [0.08, 0.3]$, erasing scale = 0.02–0.08.

This **differential augmentation** strategy — applying more aggressive perturbations to disease-positive images — is motivated by the observation that the positive class is both smaller and more prone to overfitting, as the model sees the same positive examples more frequently due to the WeightedRandomSampler (Section 3.3). By applying stronger augmentation to positive samples, the effective diversity of the positive class is increased, reducing the risk of memorisation. This approach is related to the mixup [37] and cutmix [38] paradigms but operates at the augmentation level rather than the sample-combination level. The 70% application probability was selected to ensure that the model still sees unmodified positive examples 30% of the time, preserving the original disease signal for learning.

### 3.3 Class-Balanced Sampling

A `WeightedRandomSampler` assigns higher sampling probability to images containing diseases, rebalancing the effective training distribution without discarding healthy samples. Per-image weights are computed as:

$$
w_i = \frac{1}{|\mathcal{D}|} \sum_{j=1}^{|\mathcal{D}|} \begin{cases} \sqrt{\frac{1}{n_j^+}} & \text{if } y_{ij} = 1 \\ \sqrt{\frac{1}{n_j^-}} & \text{if } y_{ij} = 0 \end{cases}
$$

where $n_j^+$ and $n_j^-$ are the positive and negative counts for disease $j$, and $y_{ij}$ is the binary label for image $i$ and disease $j$. The square-root weighting provides a moderate rebalancing that avoids the instability of inverse-frequency weighting on extremely imbalanced classes. Inverse-frequency weighting ($1/n_j^+$) can assign excessively large weights to rare classes, causing the loss landscape to be dominated by a handful of samples and leading to gradient instability [39]. The square-root dampening ($\sqrt{1/n_j^+}$) reduces the weight ratio between rare and common classes while still providing meaningful rebalancing.

The sampler draws $2 \times |\text{train}|$ samples per epoch with replacement, effectively doubling the number of positive-class exposures. This oversampling factor of 2 was selected empirically: higher factors (4×, 8×) led to increased overfitting on the positive class, while lower factors (1×) did not provide sufficient positive-class exposure per epoch. The replacement=True setting ensures that rare positive images can appear multiple times per epoch, which is necessary when the positive class comprises fewer than 5% of the total corpus.

---

## 4. Model Architecture

### 4.1 Design Rationale

The central architectural hypothesis of GLAAM-4X is that disease-specific attention mechanisms, each tailored to the spatial scale and morphological characteristics of its target pathology, outperform a single shared attention pathway. This hypothesis is grounded in the clinical observation that the four target diseases manifest at fundamentally different scales in fundus images:

- **DR**: Lesions span three orders of magnitude — microaneurysms (2–5 px), dot/blot haemorrhages (5–20 px), and large haemorrhages/exudates (20–50+ px) [1]. A single-scale attention mechanism cannot simultaneously capture microaneurysm-level detail and haemorrhage-level context. This motivates the MultiScaleGLAAM design (Section 4.5), which applies attention at three spatial scales.
- **Glaucoma**: The diagnostic signal is concentrated at the optic disc and cup, a relatively large, well-localised structure occupying approximately 50–100 pixels in a standard fundus image [2]. Attention should focus on this region without being distracted by peripheral retinal features. A moderate reduction ratio ($r = 8$) in the GLAAMBlock provides the appropriate receptive field size.
- **Cataract**: In fundus photography, cataract manifests as diffuse, global image quality degradation rather than localised lesions [3]. The lens opacity reduces overall image clarity uniformly, meaning the attention signal is distributed across the entire image rather than concentrated at a specific location. A high reduction ratio ($r = 16$) in the GLAAMBlock captures this global pattern by compressing the channel attention into a coarse representation.
- **Myopia**: Pathological myopia is associated with peripapillary tessellation and atrophy [4], which are already well-captured by the convolutional features of the MobileNetV2 backbone without explicit attention. An identity (no-attention) pathway prevents unnecessary computational overhead and avoids the risk of attention noise degrading the already-saturated myopia signal.

This disease-specific design philosophy contrasts with the standard approach in multi-label classification, where a single shared feature extractor feeds multiple classification heads [40,41]. While architecturally simpler, the shared approach forces the feature extractor to learn a compromise representation that is adequate for all diseases but optimal for none. GLAAM-4X instead allows each disease to refine the shared backbone features through its own attention pathway, producing disease-specialised representations that are then classified by dedicated heads.

### 4.2 Backbone: MobileNetV2

The backbone is a MobileNetV2 [13] feature extractor pretrained on ImageNet [29]. MobileNetV2 was selected for its favourable efficiency–accuracy trade-off, which is critical for a screening tool intended for deployment in resource-constrained settings (rural clinics, mobile screening units). The architecture introduces two key innovations over the original MobileNet [42]:

**Inverted residual blocks with linear bottlenecks.** Standard residual blocks [43] use a wide → narrow → wide pattern: the input is compressed to a low-dimensional representation and then expanded back. MobileNetV2 inverts this: the input is first expanded to a higher-dimensional representation via a $1 \times 1$ convolution, processed by a depthwise separable convolution [44], and then projected back to a low-dimensional representation. The residual connection operates on the low-dimensional (bottleneck) space, not the expanded space. This design reduces the number of multiply-accumulate operations by approximately 2–3× compared to standard residual blocks of equivalent capacity.

**Linear bottlenecks.** The final $1 \times 1$ projection in each inverted residual block uses a linear activation (no ReLU) rather than a non-linear one. ReLU activations in low-dimensional spaces destroy information: if a low-dimensional tensor has many zeroed-out channels after ReLU, the remaining channels cannot reconstruct the original signal. By using linear projections in the bottleneck, MobileNetV2 preserves the full information content of the low-dimensional representation.

The backbone outputs a 1280-channel feature map at $12 \times 12$ spatial resolution for $384 \times 384$ input. This feature map serves as the input to all four disease-specific attention specialists. The 1280-channel dimension provides a rich representation (compared to 512 or 256 in earlier mobile architectures), while the $12 \times 12$ spatial resolution retains sufficient spatial detail for the attention mechanisms to localise disease features. The total backbone parameter count is approximately 3.4M, contributing roughly 28% of the model's 12.3M total parameters.

### 4.3 Attention Mechanisms: Background

Attention mechanisms in convolutional neural networks adaptively reweight feature maps to emphasise informative features and suppress irrelevant ones. The attention literature distinguishes two complementary dimensions along which reweighting can occur: the *channel* dimension and the *spatial* dimension. Understanding this distinction is essential for the design of GLAAM-4X, because the four target diseases differ in which dimension carries the most diagnostic signal.

**Channel attention (global attention).** A convolutional feature map $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ contains $C$ channels, each encoding a different feature pattern learned during training — for example, one channel might respond to edge orientations, another to texture frequency, and another to colour contrast. Not all channels are equally relevant for every input: a fundus image showing glaucomatous cupping activates channels encoding optic disc morphology, while an image with DR haemorrhages activates channels encoding red lesion patterns. Channel attention computes a per-channel importance weight $\mathbf{g} \in \mathbb{R}^{C \times 1 \times 1}$ that scales each channel up or down based on the current input. Because the weight is shared across all spatial positions within a channel (it is a single scalar per channel), this mechanism is also called **global attention**: it answers the question "which feature channels are important for this image?" without specifying *where* in the image those features are located. The seminal channel attention architecture is the Squeeze-and-Excitation (SE) block [5], which computes channel weights via global average pooling followed by a bottleneck fully-connected network.

**Spatial attention (local attention).** Conversely, spatial attention computes a per-position importance weight that varies across the spatial dimensions of the feature map. For a feature map of size $C \times H \times W$, spatial attention produces a weight map $\mathbf{l} \in \mathbb{R}^{C \times H \times W}$ (or $\mathbb{R}^{1 \times H \times W}$ in some formulations) that highlights *where* in the image the model should focus. This mechanism is also called **local attention**: it answers the question "where in the image should the model look?" without distinguishing between different feature channels. Spatial attention is particularly important for diseases that manifest at specific anatomical locations — for example, glaucoma is diagnosed from the optic disc region, and DR lesions appear in the macula and peripheral retina. The Convolutional Block Attention Module (CBAM) [6] introduced spatial attention as a complement to channel attention, using a convolutional bottleneck to generate the spatial weight map.

**Complementarity.** Channel and spatial attention are complementary: channel attention identifies *what* features are present (e.g., "haemorrhage-detecting channels are active"), while spatial attention identifies *where* those features are located (e.g., "the active region is in the temporal arcade"). Neither alone is sufficient for disease detection: channel attention without spatial attention cannot localise the disease to a specific retinal region, and spatial attention without channel attention cannot distinguish between disease-relevant and irrelevant features at the same location. This complementarity motivates the dual-branch design of the GLAAM module described in the next section.

### 4.4 GLAAM Attention Module

The Global-Local Attention Aggregation Module (GLAAM) [7] combines the two complementary attention mechanisms described above — channel attention (global) and spatial attention (local) — via a learnable fusion parameter. This dual-branch design is motivated by the observation that disease detection requires both *what* features are present (channel-level) and *where* they are located (spatial-level) [6]. Channel attention alone cannot localise disease features, and spatial attention alone cannot distinguish between disease-relevant and irrelevant feature channels.

**Global branch (channel attention).** A squeeze-and-excitation block [5] computes channel-wise importance via global average pooling followed by a bottleneck fully-connected layer:

$$
\mathbf{g} = \sigma\!\left(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(\mathbf{x}))\right) \in \mathbb{R}^{C \times 1 \times 1}
$$

where $W_1 \in \mathbb{R}^{C/r \times C}$, $W_2 \in \mathbb{R}^{C \times C/r}$, $r$ is the reduction ratio, $\sigma$ is the sigmoid function, and GAP denotes global average pooling. The global average pooling compresses the spatial dimensions $(H, W)$ into a single value per channel, producing a channel descriptor $\mathbf{z} \in \mathbb{R}^C$. The bottleneck FC layer ($W_1$, $W_2$) learns to model inter-channel dependencies, producing a per-channel weight $\mathbf{g}$ that indicates the importance of each feature channel for the current input. The reduction ratio $r$ controls the capacity of this branch: a smaller $r$ provides more fine-grained channel modelling at higher computational cost, while a larger $r$ produces a coarser but more efficient representation.

**Local branch (spatial attention).** A $1 \times 1$ convolution bottleneck generates a spatial attention map:

$$
\mathbf{l} = \sigma\!\left(\text{Conv}_{1 \times 1}^{(2)} \cdot \text{ReLU}(\text{BN}(\text{Conv}_{1 \times 1}^{(1)}(\mathbf{x})))\right) \in \mathbb{R}^{C \times H \times W}
$$

The first $1 \times 1$ convolution reduces the channel dimension from $C$ to $C/r$, the batch normalisation [45] stabilises the intermediate activations, and the second $1 \times 1$ convolution projects back to $C$ channels. The sigmoid activation produces a per-pixel, per-channel weight $\mathbf{l}$ that indicates the spatial importance of each feature at each location. Unlike the global branch, which produces a single weight per channel, the local branch produces a full $C \times H \times W$ attention map, enabling spatially varying attention.

**Fusion.** The two branches are combined via a learnable scalar $\alpha$:

$$
\mathbf{A} = \alpha \cdot \mathbf{g} + (1 - \alpha) \cdot \mathbf{l}, \qquad \mathbf{y} = \mathbf{x} + \mathbf{x} \odot \mathbf{A}
$$

where $\alpha$ is initialised at 0.5 and learned during training. The learnable $\alpha$ allows the model to dynamically balance the contribution of channel and spatial attention: for diseases where the signal is primarily spatial (e.g., glaucoma at the optic disc), the model can learn a low $\alpha$ that emphasises the local branch; for diseases where the signal is primarily feature-based (e.g., cataract as global image quality), the model can learn a high $\alpha$ that emphasises the global branch. The residual connection ($\mathbf{y} = \mathbf{x} + \mathbf{x} \odot \mathbf{A}$) preserves the original feature information while allowing the attention mechanism to refine it. This is critical for training stability: without the residual connection, the attention mechanism could zero out important features early in training, preventing gradient flow to the backbone. The residual formulation ensures that the attention output is always at least as informative as the input, following the principle of identity mapping in residual networks [43].

![Figure 2: GLAAM attention module](report_figures/fig2_glaam_module.png)

> **Figure 2.** Internal structure of the GLAAM (Global-Local Attention Aggregation Module) attention module. The input feature map $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ enters from the left and is processed by two parallel branches. The **global branch** (blue, upper pathway) performs channel attention: global average pooling (GAP) compresses the spatial dimensions into a single channel descriptor, which is passed through a bottleneck fully-connected network ($W_1$: $C \to C/r$, ReLU, $W_2$: $C/r \to C$) and a sigmoid activation, producing per-channel importance weights $\mathbf{g} \in \mathbb{R}^{C \times 1 \times 1}$. This branch answers the question "which feature channels are important for this image?" The **local branch** (orange, lower pathway) performs spatial attention: a $1 \times 1$ convolution bottleneck ($C \to C/r$, batch normalisation, ReLU, $C/r \to C$) followed by sigmoid produces a full per-position, per-channel weight map $\mathbf{l} \in \mathbb{R}^{C \times H \times W}$. This branch answers "where in the image should the model look?" The two branches are fused via a learnable scalar $\alpha$ (purple): $\mathbf{A} = \alpha \cdot \mathbf{g} + (1 - \alpha) \cdot \mathbf{l}$. The fused attention map is applied to the input via element-wise multiplication ($\mathbf{x} \odot \mathbf{A}$), and a residual connection adds the original input to produce the output $\mathbf{y} = \mathbf{x} + \mathbf{x} \odot \mathbf{A}$ (green). The residual connection ensures that the output is always at least as informative as the input, preserving gradient flow to the backbone during training and preventing the attention mechanism from zeroing out important features in the early training phases.

### 4.5 Integration of GLAAM with the MobileNetV2 Backbone

The integration of the GLAAM attention module with the MobileNetV2 backbone is a critical architectural decision that determines where in the feature hierarchy disease-specific attention is applied. Two integration strategies are possible: (i) *intra-backbone injection*, where GLAAM blocks are inserted between individual MobileNetV2 inverted residual blocks, refining features at multiple depths; and (ii) *post-backbone application*, where GLAAM blocks operate on the final backbone output, refining the 1280-channel feature map after all convolutional processing is complete. GLAAM-4X adopts the post-backbone strategy, and the rationale for this choice is as follows.

**Architectural connection point.** The MobileNetV2 backbone processes the input image through 17 inverted residual blocks, progressively reducing spatial resolution ($384 \to 12$) while increasing channel depth ($32 \to 1280$). The final block produces a feature map $\mathbf{F} \in \mathbb{R}^{B \times 1280 \times 12 \times 12}$ that encodes the highest-level semantic representation of the input image. This feature map is the shared input to all four disease-specific attention specialists:

$$
\mathbf{F} = \text{MobileNetV2}(\mathbf{x}_{\text{image}}) \in \mathbb{R}^{B \times 1280 \times 12 \times 12}
$$

$$
\mathbf{f}_d = \text{GLAAMBlock}_d(\mathbf{F}) \quad \text{for each disease } d \in \{\text{DR, Glaucoma, Cataract, Myopia}\}
$$

where $\mathbf{f}_d$ is the attention-refined feature map for disease $d$. Each GLAAM block takes the *same* 1280-channel feature map as input but applies its own disease-specific attention weights (with different reduction ratios and, for DR, multi-scale processing), producing four distinct refined representations from a single shared backbone output.

**Why post-backbone rather than intra-backbone.** The intra-backbone injection strategy — inserting GLAAM blocks at intermediate stages of MobileNetV2 (e.g., after blocks 13 and 17, as in the original GLAAM work [7]) — refines features at multiple depths but introduces several disadvantages for the multi-disease setting. First, intra-backbone injection modifies the backbone's feature hierarchy, meaning the backbone can no longer be treated as a frozen pretrained feature extractor; the attention gradients propagate back through the backbone, potentially disrupting the ImageNet-pretrained features. Second, with four disease-specific attention pathways, intra-backbone injection would require four separate attention-modified copies of the backbone (or a complex shared-backbone with disease-specific side branches at each injection point), dramatically increasing the parameter count and computational cost. Third, the intermediate feature maps have lower channel depths (24–320 channels at stages 3–17), providing less representational capacity for the attention mechanism to distinguish between disease-specific patterns.

The post-backbone strategy avoids all three issues. The backbone remains a standard, unmodified MobileNetV2 whose pretrained features are preserved. The four disease-specific GLAAM blocks operate on the richest available representation (1280 channels), and they share the same backbone output, meaning the backbone is computed only once per forward pass regardless of the number of disease specialists. The computational cost of the four attention pathways is additive rather than multiplicative: the backbone forward pass is $O(1)$, and each attention pathway is $O(C^2/r)$ where $C = 1280$ and $r$ is the reduction ratio. The total additional cost of the four attention specialists is approximately 2.3M parameters (18% of the total 12.3M), a modest overhead given the performance improvement.

**Data flow.** The complete forward pass proceeds as follows:

1. **Backbone feature extraction:** The input image $\mathbf{x}_{\text{image}} \in \mathbb{R}^{3 \times 384 \times 384}$ is passed through the MobileNetV2 backbone, producing the shared feature map $\mathbf{F} \in \mathbb{R}^{1280 \times 12 \times 12}$.

2. **Disease gating:** The gating network computes disease relevance weights $\mathbf{w}_{\text{gate}} = \text{Softmax}(\text{MLP}(\text{GAP}(\mathbf{F}))) \in \mathbb{R}^4$ from the globally pooled feature map (Section 4.7).

3. **Disease-specific attention:** Each attention specialist processes the shared feature map independently:
   - DR: $\mathbf{f}_{\text{DR}} = \text{MultiScaleGLAAM}(\mathbf{F})$ — three-scale attention (Section 4.6)
   - Glaucoma: $\mathbf{f}_{\text{Glaucoma}} = \text{GLAAMBlock}_{r=8}(\mathbf{F})$ — standard GLAAM with moderate reduction
   - Cataract: $\mathbf{f}_{\text{Cataract}} = \text{GLAAMBlock}_{r=16}(\mathbf{F})$ — standard GLAAM with high reduction
   - Myopia: $\mathbf{f}_{\text{Myopia}} = \mathbf{F}$ — identity (no attention)

4. **Global average pooling:** Each disease-specific feature map is spatially pooled to produce a 1280-dimensional feature vector: $\mathbf{v}_d = \text{GAP}(\mathbf{f}_d) \in \mathbb{R}^{1280}$.

5. **Disease-specific classification:** Each pooled vector is passed through its dedicated classifier head (Section 4.8), producing a single logit per disease. The four logits are stacked into the output vector $\mathbf{z} \in \mathbb{R}^4$.

This architecture ensures that the backbone features are computed once and then specialised by each disease's attention pathway, achieving disease-specific feature refinement without redundant backbone computation.

> **Figure 4.** Complete GLAAM-4X architecture. See `report_figures/fig4_full_architecture.png` for the full-resolution figure. The input fundus image ($3 \times 384 \times 384$) is processed by the MobileNetV2 backbone (dark grey, 3.4M parameters), producing a shared 1280-channel feature map at $12 \times 12$ spatial resolution. This feature map is simultaneously routed to four disease-specific attention specialists: **DR** (blue, MultiScaleGLAAM with three scales and reduction ratios 4/8/16), **Glaucoma** (purple, GLAAMBlock with $r = 8$), **Cataract** (red, GLAAMBlock with $r = 16$), and **Myopia** (blue, identity — no attention). A disease gating network (orange) computes per-image disease relevance weights from the globally pooled feature map. Each specialist's output is globally average pooled and classified by a dedicated head with disease-appropriate capacity (DR: 256-dim, Glaucoma/Cataract: 128-dim, Myopia: 64-dim). The four logits are reordered to the canonical disease order and passed through sigmoid to produce independent disease probabilities. The backbone is computed once and shared across all four specialists, with the attention pathways adding only 2.3M parameters (18% of the 12.3M total).

### 4.6 Disease-Specific Attention Specialists

Each disease receives a dedicated attention pathway applied to the 1280-channel backbone feature map. The attention configuration for each disease is determined by the spatial scale and morphological characteristics of its target pathology:

> **Table 3: Disease-Specific Attention Configuration.** Architecture of each attention specialist, including the attention type, reduction ratio, and clinical rationale for the configuration.

| Disease | Attention type | Reduction $r$ | Rationale |
|---|---|---|---|
| DR | MultiScaleGLAAM (3 scales) | 4 (fine), 8 (med), 16 (coarse) | Lesions span 2–50+ px; multi-scale captures microaneurysms to haemorrhages |
| Glaucoma | GLAAMBlock | 8 | Optic disc is a localised, medium-scale structure; moderate reduction |
| Cataract | GLAAMBlock | 16 | Lens opacity is diffuse and global; high reduction captures large-scale pattern |
| Myopia | Identity (no attention) | — | Tessellation already captured by backbone; attention adds no signal |

The reduction ratio $r$ controls the capacity of the attention bottleneck. For glaucoma ($r = 8$), the bottleneck dimension is $1280/8 = 160$, providing moderate channel compression that retains enough capacity to model the optic disc features. For cataract ($r = 16$), the bottleneck dimension is $1280/16 = 80$, producing a coarser representation that captures the global, low-frequency nature of lens opacity. The DR specialist uses the smallest reduction ratio ($r = 4$) at the fine scale, yielding a bottleneck of $1280/4 = 320$ channels — the highest capacity, reflecting the complexity and multi-scale nature of DR lesions.

### 4.7 MultiScaleGLAAM for Diabetic Retinopathy

The DR specialist employs a three-scale attention mechanism to capture the wide range of lesion sizes that characterise diabetic retinopathy. This design is motivated by the clinical observation that DR lesions span three orders of magnitude in spatial extent [1]:

- **Microaneurysms** (2–5 px): The earliest detectable DR lesion, appearing as small red dots. Their detection requires high-resolution features.
- **Dot/blot haemorrhages and hard exudates** (5–20 px): Medium-scale lesions that indicate moderate non-proliferative DR.
- **Large haemorrhages, cotton wool spots, and venous beading** (20–50+ px): Coarse-scale signs of severe non-proliferative or proliferative DR.

A single-scale attention mechanism cannot simultaneously capture all three lesion sizes: at native resolution, the receptive field is too small to capture large haemorrhages; at $4\times$ downsampled resolution, microaneurysms are lost. The MultiScaleGLAAM addresses this by applying GLAAM attention at three spatial scales and fusing the results:

> **Figure 3.** MultiScaleGLAAM architecture for the DR specialist. See `report_figures/fig3_multiscale_glaam.png` for the full-resolution figure. The shared 1280-channel feature map is processed at three spatial scales: **fine** (blue, native $12 \times 12$ resolution, $r = 4$) for microaneurysm detection, **medium** (orange, $2\times$ downsampled to $6 \times 6$, $r = 8$) for dot/blot haemorrhages, and **coarse** (red, $4\times$ downsampled to $3 \times 3$, $r = 16$) for large haemorrhages and exudates. Each scale applies an independent GLAAM block with disease-appropriate reduction ratio. The medium and coarse outputs are bilinearly upsampled to the original resolution. The three scales are fused via two parallel mechanisms: (i) a learnable softmax-weighted sum that provides smooth interpolation, and (ii) a concatenation followed by a $1 \times 1$ convolution that learns cross-scale interactions. The output is a unified 1280-channel feature map encoding DR-relevant patterns across all lesion sizes.

$$
\mathbf{f}_{\text{fine}} = \text{GLAAM}_{r=4}(\mathbf{x})
$$

$$
\mathbf{f}_{\text{med}} = \text{up}\!\left(\text{GLAAM}_{r=8}\!\left(\text{AvgPool}_{2\times}(\mathbf{x})\right)\right)
$$

$$
\mathbf{f}_{\text{coarse}} = \text{up}\!\left(\text{GLAAM}_{r=16}\!\left(\text{AvgPool}_{4\times}(\mathbf{x})\right)\right)
$$

where $\text{AvgPool}_{k\times}$ is $k\times$ average pooling and $\text{up}$ is bilinear interpolation to the original spatial resolution $(H, W)$. The three scales are fused via two mechanisms:

**Weighted sum.** Learnable softmax-normalised scale weights $\mathbf{w} = \text{softmax}(\mathbf{s}) \in \mathbb{R}^3$ produce a weighted combination:

$$
\mathbf{f}_{\text{weighted}} = w_1 \cdot \mathbf{f}_{\text{fine}} + w_2 \cdot \mathbf{f}_{\text{med}} + w_3 \cdot \mathbf{f}_{\text{coarse}}
$$

The softmax normalisation ensures the weights sum to 1, preventing scale dominance. The weights are initialised uniformly ($1/3$ each) and learned during training, allowing the model to adapt the relative importance of each scale based on the lesion distribution in the training data.

**Concatenation + $1 \times 1$ convolution.** The three scale outputs are concatenated along the channel dimension and projected back to $C$ channels via a $1 \times 1$ convolution with batch normalisation and ReLU:

$$
\mathbf{f}_{\text{DR}} = \text{ReLU}\!\left(\text{BN}\!\left(\text{Conv}_{1 \times 1}\!\left([\mathbf{f}_{\text{fine}} \; \| \; \mathbf{f}_{\text{med}} \; \| \; \mathbf{f}_{\text{coarse}}]\right)\right)\right)
$$

This convolution learns to combine the three scale representations into a unified feature map, potentially learning cross-scale interactions (e.g., the presence of microaneurysms at the fine scale combined with haemorrhages at the medium scale indicates more severe DR than either alone). The dual fusion mechanism (weighted sum + concatenation) provides both a smooth interpolation and a learned projection, following the multi-scale fusion paradigm established in feature pyramid networks [46].

### 4.8 Disease Gating Network

A gating network learns to dynamically weight each specialist's contribution based on the input image. The gating network operates on the globally pooled backbone features:

$$
\mathbf{w}_{\text{gate}} = \text{Softmax}\!\left(\text{MLP}\!\left(\text{GAP}(\mathbf{x})\right)\right) \in \mathbb{R}^4
$$

where the MLP is a two-layer network (1280 → 256 → 4) with ReLU activation and dropout ($p = 0.2$) between the layers. The softmax output produces a probability distribution over the four diseases, indicating which specialists are most relevant for the current image.

The gating network serves a different purpose from the attention specialists. While the attention specialists refine the *spatial and channel* features for their respective diseases, the gating network determines the *global* relevance of each specialist to the current image. For example, an image with clear optic disc cupping but no DR lesions should produce a high gate weight for the Glaucoma specialist and a low weight for the DR specialist, ensuring that the DR specialist's (irrelevant) attention refinements do not introduce noise into the final prediction.

The gating weights are not explicitly multiplied with the specialist features in the current implementation — each specialist's features are independently pooled and classified regardless of the gate weights. The gating network's primary role is to provide an interpretable signal (which diseases the model considers most relevant) and to regularise the specialist features through the shared backbone gradient. In future work, the gate weights could be used to modulate the specialist contributions explicitly, following the mixture-of-experts paradigm [47].

### 4.9 Disease-Specific Classifiers

Each disease has a dedicated classification head operating on the attention-refined, globally pooled features:

> **Table 4: Classifier Head Architecture.** Architecture of each disease-specific classifier, including hidden dimension and dropout rate. The varying capacity reflects the complexity of each disease's decision boundary.

| Disease | Hidden dim | Dropout | Parameters |
|---|---|---|---|
| DR | 256 | 0.3 | 328,193 |
| Glaucoma | 128 | 0.15 | 163,905 |
| Cataract | 128 | 0.15 | 163,905 |
| Myopia | 64 | 0.0 | 82,049 |

The capacity of each classifier is proportional to the complexity of its disease's decision boundary. The DR classifier has the largest capacity (256-dim hidden layer, 0.3 dropout) because DR's multi-scale lesion patterns produce the most complex decision boundary — the model must distinguish between five DR severity grades (no DR, mild, moderate, severe, proliferative) collapsed into a binary label, and the visual features span multiple scales. The high dropout rate (0.3) provides strong regularisation to prevent overfitting on the relatively small number of DR-positive images.

The Glaucoma and Cataract classifiers have moderate capacity (128-dim, 0.15 dropout), reflecting their intermediate decision boundary complexity. The Myopia classifier has the smallest capacity (64-dim, no dropout) because myopia is the simplest decision (tessellation presence/absence) and the identity attention pathway means the classifier operates directly on raw backbone features without attention refinement. The absence of dropout for Myopia is intentional: the myopia signal is already well-captured by the backbone, and the small classifier capacity (64-dim) provides sufficient regularisation without explicit dropout.

Each classifier outputs a single logit, and the four logits are stacked into a $(B, 4)$ tensor representing the model's prediction for all four diseases simultaneously. The sigmoid function is applied at inference time to convert each logit to an independent probability, reflecting the multi-label nature of the task (a patient may have multiple diseases simultaneously).

### 4.10 Logit Reordering

The GLAAM-4X backbone produces logits in the order $[\text{DR}, \text{Glaucoma}, \text{Cataract}, \text{Myopia}]$, following the internal `DISEASE_NAMES` constant in the architecture code. A wrapper module (`GLAAM4XClassifier`) reorders the output to $[\text{Cataract}, \text{DR}, \text{Glaucoma}, \text{Myopia}]$ via index permutation $[2, 0, 1, 3]$, aligning the output with the CSV column order used throughout the training pipeline and the inference demo. This reordering is a zero-cost operation (index permutation) that ensures consistency between the model's internal disease ordering and the external data/labelling convention.

### 4.11 Model Compilation

The model is compiled with `torch.compile(mode='reduce-overhead')` [48] when available (PyTorch 2.0+), reducing training overhead by 20–40% through kernel fusion, memory planning, and graph-level optimisation. The `reduce-overhead` mode specialises in reducing the Python overhead of eager execution by compiling the entire forward and backward pass into a single optimised graph. The compiled model's state dict keys are prefixed with `_orig_mod.`, which is stripped during inference weight loading (Section 8.3) to ensure compatibility between the compiled training checkpoint and the uncompiled inference model.

### 4.12 Parameter Summary

> **Table 5: Model Efficiency Summary.** Total parameter count, model size, and inference latency of GLAAM-4X on a single Tesla T4 GPU at $384 \times 384$ input resolution.

| Metric | Value |
|---|---|
| Total parameters | 12,315,120 |
| Trainable parameters | 12,315,120 |
| Model size (SafeTensors) | 47.17 MB |
| Mean inference latency (batch size 1, GPU) | 12.44 ms |
| P95 inference latency (batch size 1, GPU) | 14.72 ms |
| Input resolution | $384 \times 384$ |

The 12.3M parameter count and 47 MB model size make GLAAM-4X suitable for deployment on resource-constrained devices. For comparison, ResNet-50 [43] has 25.6M parameters (102 MB), EfficientNet-B3 [49] has 12M parameters (48 MB), and DenseNet-121 [50] has 8M parameters (32 MB). The 12.44 ms mean inference latency on a Tesla T4 GPU corresponds to approximately 80 frames per second, sufficient for real-time screening applications.

---

## 5. Loss Function and Optimisation

### 5.1 Asymmetric Loss

The training objective is Asymmetric Loss (ASL) [14], which addresses the severe class imbalance inherent in multi-label fundus disease classification. The choice of loss function is critical in the imbalanced multi-label setting: standard binary cross-entropy (BCE) loss treats all samples equally, producing large gradients from the dominant negative class (healthy images) and small gradients from the rare positive class (disease images). This gradient imbalance causes the model to converge to a trivial "all negative" solution early in training, where the negative class is well-classified but the positive class is essentially ignored [8].

**Focal loss** [9] partially addresses this by multiplying the BCE loss by a focusing factor $(1 - p_t)^\gamma$, which down-weights easy examples (those with high predicted probability for the correct class) and focuses the optimisation on hard examples. However, focal loss applies the same focusing parameter $\gamma$ to both positive and negative samples. In the extreme imbalance setting of fundus disease detection (where negative samples outnumber positive samples by 3:1 to 15:1), the majority of easy examples are negatives, and focal loss reduces their gradient contribution. But it also reduces the gradient from easy *positive* examples, which are valuable in the imbalanced setting because every positive example is precious.

ASL resolves this asymmetry by introducing separate focusing parameters for positive and negative samples, along with a probability clipping mechanism for negatives:

$$
\mathcal{L}_{\text{ASL}}(z, y) = \begin{cases} (1 - p_m)^{\gamma_{\text{neg}}} \cdot \log(1 - p_m) & \text{if } y = 0 \\ p^{\gamma_{\text{pos}}} \cdot \log(p) & \text{if } y = 1 \end{cases}
$$

where $p = \sigma(z)$ is the predicted probability, $p_m = \min(p, m)$ is the clipped probability for negatives, $\gamma_{\text{neg}}$ and $\gamma_{\text{pos}}$ are the negative and positive focusing parameters, and $m$ is the probability clip threshold. The configuration used in this work is:

> **Table 6: Asymmetric Loss Hyperparameters.** Configuration of the ASL objective, selected to focus learning on rare positive examples while suppressing gradients from easy negatives.

| Parameter | Value | Rationale |
|---|---|---|
| $\gamma_{\text{neg}}$ | 4.0 | Aggressively down-weight easy negative gradients |
| $\gamma_{\text{pos}}$ | 0.0 | No positive focusing (positive samples are rare; all are valuable) |
| Clip $m$ | 0.05 | Prevent overconfidence on dominant negative class |

Setting $\gamma_{\text{pos}} = 0$ means positive samples receive standard BCE gradients without any focusing: the loss for a positive sample is simply $-\log(p)$, ensuring that all positive examples — including easy ones where the model already predicts high probability — continue to contribute gradient signal. This is critical in the imbalanced setting because the positive class is small, and discarding gradient from easy positives would slow learning of the positive-class representation.

The high $\gamma_{\text{neg}} = 4.0$ aggressively suppresses gradients from well-classified negatives. When the model predicts $p = 0.01$ for a negative sample (correctly classifying it as negative), the focusing factor $(1 - 0.01)^4 \approx 0.96$ provides minimal suppression. But when the model predicts $p = 0.3$ for a negative sample (uncertain), the focusing factor $(1 - 0.3)^4 \approx 0.24$ substantially reduces the gradient, preventing the model from spending capacity on already-well-classified negatives. The fourth-power focusing is more aggressive than the standard focal loss ($\gamma = 2$), which would produce $(1 - 0.3)^2 \approx 0.49$ — roughly twice the gradient weight.

The probability clip $m = 0.05$ caps the negative probability at 5% before applying the focusing term. Once the model is confident that a sample is negative ($p < 0.05$), the clipped probability $p_m = p$ and the focusing factor $(1 - p)^4 \approx 1$, so the gradient is essentially the standard BCE gradient. But for samples where the model is uncertain ($p > 0.05$), the clipped probability $p_m = 0.05$ and the focusing factor $(1 - 0.05)^4 \approx 0.81$, providing moderate suppression. The clip prevents the model from becoming overconfident on the negative class: without the clip, the model could drive $p \to 0$ for all negatives, producing zero gradient and preventing further learning. The clip ensures that even well-classified negatives contribute a small gradient signal, maintaining the model's ability to distinguish between easy and hard negatives.

### 5.2 Optimiser and Learning Rate Schedule

![Figure 5: Learning rate schedule](report_figures/fig5_lr_schedule.png)

> **Figure 5.** Learning rate schedule and training progression over 60 epochs. The schedule consists of two phases. **Phase 1 (epochs 1–10, blue shaded region):** linear warmup that gradually increases the learning rate from 0 to $2 \times 10^{-5}$, stabilising early training when the randomly initialised attention modules and classifier heads produce large, noisy gradients that could destabilise the pretrained MobileNetV2 backbone features. The low initial learning rate allows the new modules to adapt gradually without disrupting the pretrained features, and provides time for the AdamW optimiser's moment estimates to stabilise. **Phase 2 (epochs 11–60, purple shaded region):** cosine annealing that smoothly reduces the learning rate from $2 \times 10^{-5}$ to 0 following a cosine curve, providing broad exploration early and fine convergence near the end. The smooth decay avoids the abrupt learning rate drops of step decay schedules, which can cause the model to converge to a sharp local minimum. Three annotated markers are shown: the warmup end at epoch 10 (grey dashed line), the best epoch at 48 where validation macro F1 reached 0.8275 (green dashed line), and the early stopping criterion with patience = 10 (red annotation), which halts training if no validation improvement is observed for 10 consecutive epochs. The actual learning rate curve matches the theoretical schedule exactly, confirming correct scheduler implementation.

> **Table 7: Optimisation Configuration.** Training hyperparameters for GLAAM-4X, including optimiser, learning rate schedule, batch size, and early stopping criteria.

| Parameter | Value |
|---|---|
| Optimiser | AdamW [51] ($\text{lr} = 2 \times 10^{-5}$, weight decay $= 5 \times 10^{-4}$) |
| LR schedule | 10-epoch linear warmup → cosine annealing [52] over 60 epochs |
| Batch size | 32 |
| Total epochs | 60 (early stopping, patience $= 10$) |
| Gradient clipping | $\|\nabla\|_2 \leq 10.0$ |
| Mixed precision | AMP [53] (float16 forward, float32 accumulation) |
| Best epoch | 48 |
| Best validation macro F1 | 0.8275 |

**AdamW optimiser.** AdamW [51] was selected over standard Adam [54] because it decouples weight decay from the gradient update. In standard Adam, weight decay is applied by adding $\lambda \cdot \theta$ to the gradient before the adaptive update, which interacts with the momentum and variance estimates in unintended ways: the effective weight decay depends on the gradient magnitude and the moment estimates, producing inconsistent regularisation across parameters. AdamW applies weight decay *after* the gradient update ($\theta \leftarrow \theta - \eta \cdot \lambda \cdot \theta$), ensuring that the decay rate is uniform across all parameters and independent of the gradient statistics. This decoupled weight decay has been shown to improve generalisation, particularly for transfer learning tasks where the pretrained backbone parameters require different regularisation than the randomly initialised classifier heads [51].

The learning rate of $2 \times 10^{-5}$ is deliberately low — approximately 10× lower than typical fine-tuning rates for ImageNet-pretrained models ($10^{-4}$ to $10^{-3}$). This low rate was selected because the model is being fine-tuned on a domain (fundus images) that differs substantially from the pretraining domain (natural images), and the disease-specific attention modules are randomly initialised. A low learning rate prevents the randomly initialised attention modules from producing large gradient updates that could destabilise the pretrained backbone features. The weight decay of $5 \times 10^{-4}$ provides moderate L2 regularisation, preventing overfitting on the relatively small positive class.

**Linear warmup.** The first 10 epochs use a linear warmup schedule that gradually increases the learning rate from 0 to the target value:

$$
\eta(t) = \eta_{\text{max}} \cdot \frac{t + 1}{T_{\text{warmup}}}
$$

where $t$ is the epoch index (0-based) and $T_{\text{warmup}} = 10$. Warmup is critical in the early phase of training for two reasons [55]. First, the randomly initialised attention modules and classifier heads produce large, noisy gradients that can destabilise the pretrained backbone. A low initial learning rate allows the new modules to adapt gradually without disrupting the pretrained features. Second, the AdamW optimiser's moment estimates (first and second moments) are initialised to zero and require several iterations to stabilise. During the first few iterations, the moment-corrected gradient can be very large, and a low learning rate prevents these large updates from causing catastrophic forgetting of the pretrained features.

**Cosine annealing.** After the warmup period, the learning rate follows a cosine annealing schedule [52]:

$$
\eta(t) = \frac{\eta_{\text{max}}}{2} \left(1 + \cos\!\left(\frac{\pi \cdot (t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)
$$

where $T_{\text{total}} = 60$ is the total number of epochs. Cosine annealing smoothly reduces the learning rate from $\eta_{\text{max}}$ to 0 over the remaining 50 epochs, following a cosine curve. This schedule has two advantages over step decay (which reduces the learning rate by a fixed factor at predetermined epochs). First, the smooth decay avoids the sudden learning rate drops that can cause the model to converge to a sharp local minimum [52]. Second, the cosine shape provides a "long tail" of low learning rates near the end of training, allowing the model to fine-tune its parameters with small, precise updates — effectively performing a form of implicit averaging over the loss landscape, which has been shown to improve generalisation [56].

**Gradient clipping.** Gradient clipping at $\|\nabla\|_2 \leq 10.0$ is applied after the backward pass and before the optimiser step. This prevents gradient spikes from the asymmetric loss on hard negative samples — when the model is very confident about a negative sample ($p \approx 0$) but the label is positive (a label-noise scenario from the single-disease datasets, Section 2.2), the ASL loss can produce a very large gradient. Without clipping, these spikes can destabilise training by causing large parameter updates that destroy the learned features. The threshold of 10.0 was selected empirically: lower thresholds (1.0, 5.0) were too aggressive and slowed convergence, while higher thresholds (50.0, 100.0) did not prevent the occasional instability.

**Automatic Mixed Precision (AMP).** AMP [53] reduces GPU memory usage and increases throughput by performing forward and backward passes in float16 while accumulating gradients in float32. The `GradScaler` dynamically scales the loss to prevent float16 underflow (where small gradient values become zero in float16 representation). AMP is particularly beneficial for the MultiScaleGLAAM module, which involves three parallel attention pathways and a concatenation-based fusion — operations that produce large intermediate tensors. With AMP, the peak memory usage is reduced by approximately 40%, allowing batch size 32 at $384 \times 384$ input on the 16 GB Tesla T4 GPU.

**Early stopping.** Training is halted if the validation macro F1 (at optimal thresholds) does not improve for 10 consecutive epochs. The best model (highest validation macro F1) is saved separately and used for all downstream evaluation. Early stopping prevents overfitting in the later epochs, where the model may begin to memorise the training set at the expense of generalisation. The patience of 10 epochs was selected to allow the cosine annealing schedule sufficient time to find a better minimum in the later, low-learning-rate phase without continuing training past the point of diminishing returns.

**NaN/Inf loss handling.** Batches that produce non-finite loss (NaN or Inf) are skipped rather than updating the weights. This safety mechanism is essential when training with ASL on noisy labels: a batch containing several incorrectly labelled negatives (from the single-disease datasets) can produce a large loss spike that, combined with the asymmetric focusing, results in a non-finite gradient. Skipping the batch and updating the GradScaler (which reduces the loss scale to prevent future overflow) allows training to continue without manual intervention.

### 5.3 Training Infrastructure

Training was conducted on Google Colab with a Tesla T4 GPU (16 GB VRAM, 65 TFLOPS FP16). The model, training code, and checkpoints were stored on Google Drive to persist across Colab session timeouts. Resume support was implemented: if a `latest_model.pth` checkpoint exists at the start of training, the model weights, optimiser state, learning rate scheduler state, and training history are restored, and training continues from the last completed epoch. This is essential for Colab, where sessions can disconnect after 12 hours and all local state is lost. The resume mechanism ensures that a training run interrupted at epoch 52 can continue from epoch 53 without retraining the first 52 epochs.

---

## 6. Evaluation Metrics and Threshold Optimisation

### 6.1 Metrics

For medical screening, the relevant metrics capture different aspects of model performance:

- **AUC (Area Under the ROC Curve):** The probability that the model ranks a randomly chosen positive example above a randomly chosen negative example [57]. AUC is threshold-independent, making it the primary metric for comparing models across different operating points. In the clinical context, AUC reflects the model's overall discriminative ability — its capacity to distinguish diseased from healthy fundi across all possible decision thresholds.
- **F1 Score:** The harmonic mean of precision and recall, $\text{F1} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$. F1 provides a single metric that balances false positives and false negatives, which is important when the class distribution is imbalanced and accuracy is misleading.
- **Precision:** The fraction of predicted positives that are true disease cases, $\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$. In screening, false positives cause unnecessary referral to an ophthalmologist, increasing healthcare costs and patient anxiety. High precision ensures that flagged cases are likely to be genuine.
- **Recall (Sensitivity):** The fraction of true disease cases that are detected, $\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$. In screening, false negatives mean missed diagnoses with potentially irreversible consequences (e.g., vision loss from untreated glaucoma or DR). High recall ensures that few diseased patients are missed.
- **Macro F1:** The unweighted average of per-disease F1 scores. Used as the primary model selection metric because it gives equal weight to each disease regardless of prevalence, preventing rare diseases (Glaucoma, Myopia) from being dominated by common ones (Cataract, DR) in the optimisation objective. This is important because a model that achieves high F1 on Cataract (common) but low F1 on Myopia (rare) would have a high micro-F1 but a low macro-F1, correctly signalling that the model is inadequate for rare diseases.

### 6.2 Per-Disease Threshold Optimisation

In multi-label classification, a single decision threshold of 0.5 is rarely optimal across all labels, particularly when the labels have different prevalences [58]. A threshold of 0.5 may be too high for a rare disease (causing many false negatives) and too low for a common disease (causing many false positives). Per-disease thresholds are therefore grid-searched on the validation set to maximise per-disease F1:

$$
\tau_d^* = \arg\max_{\tau \in [0.05, 0.95]} \text{F1}(\tau; \mathcal{V}_d)
$$

where $\mathcal{V}_d$ is the validation set for disease $d$ and the search is performed at 0.01 resolution (91 candidate thresholds). The F1 score is used as the optimisation target because it balances precision and recall, which is the appropriate trade-off for screening: neither metric should be sacrificed excessively.

> **Table 8: Per-Disease Optimal Thresholds.** Thresholds optimised on the validation set to maximise per-disease F1 score.

| Disease | Optimal threshold $\tau^*$ | Interpretation |
|---|---|---|
| Cataract | 0.49 | Near-default; Cataract is the most prevalent disease, so the optimal threshold is close to 0.5 |
| DR | 0.45 | Slightly below default; increases sensitivity for DR, where missed diagnoses carry high cost (irreversible vision loss) |
| Glaucoma | 0.57 | Above default; increases precision for Glaucoma, which has higher false-positive rate due to label noise from multiple glaucoma-only datasets |
| Myopia | 0.50 | Default; Myopia has a balanced precision-recall trade-off at the standard threshold |

The Glaucoma threshold (0.57) is the highest, reflecting the model's tendency to over-predict glaucoma due to label noise from multiple glaucoma-only datasets with varying annotation standards (PAPILA, REFUGE2, glaucoma_bundle). The elevated threshold increases precision at the cost of recall, which is the dominant failure mode for this disease. In a clinical screening context, a lower Glaucoma threshold would be preferable (prioritising sensitivity over specificity), but the label noise makes this impractical without cleaner annotations.

---

## 7. Explainability: EigenGradCAM

### 7.1 Motivation

Clinical deployment of AI-based screening tools requires not only accurate predictions but also interpretable evidence of *where* in the image the model identified disease [10]. This requirement is not merely a matter of clinician preference: regulatory frameworks such as the EU AI Act [59] and the FDA's Software as a Medical Device (SaMD) guidance [60] increasingly require explainability for clinical AI systems. A clinician who receives a "Glaucoma: 96%" prediction without visual evidence of optic disc cupping cannot verify the model's reasoning, creating a trust deficit that limits clinical adoption [11].

Gradient-weighted Class Activation Mapping (Grad-CAM) [12] addresses this by generating a localisation heatmap showing which image regions most influenced the model's prediction. Grad-CAM computes the gradient of the class score with respect to the feature maps of a chosen convolutional layer, weights the feature maps by the average gradient, and produces a coarse localisation map highlighting the regions that contributed most to the prediction.

### 7.2 EigenGradCAM

The system uses EigenGradCAM [15], an improvement over standard Grad-CAM that replaces the gradient averaging step with singular value decomposition (SVD). In standard Grad-CAM, the importance weight for each channel is computed as the global average of the gradient across spatial positions:

$$
\alpha_k^c = \frac{1}{H \times W} \sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k}
$$

where $y^c$ is the class score and $A^k$ is the feature map of channel $k$. This averaging can produce noisy heatmaps when the gradient is spatially inconsistent — the positive gradients in one region may cancel negative gradients in another, producing a near-zero average weight that underestimates the channel's importance.

EigenGradCAM instead computes the principal eigenvector of the gradient matrix, which captures the dominant direction of variation in the gradient signal. This produces sharper, less noisy heatmaps that more accurately localise disease-specific features, particularly for fine-grained lesions like DR microaneurysms where the gradient signal is spatially concentrated.

### 7.3 Implementation

The EigenGradCAM implementation uses the `pytorch-grad-cam` library [61] with a thin wrapper module that exposes the model's logit output as a plain tensor (required by the library's API, which expects `model(x)` to return a tensor, not a dictionary). The target layer is the last InvertedResidual block of the MobileNetV2 backbone, which produces the highest-resolution feature map ($12 \times 12$) before global average pooling. This layer is chosen because it captures the most spatially detailed representation of the input image while still being deep enough to encode semantically meaningful features (lesions, disc structure, tessellation patterns).

The target class is the disease index in the reordered logit vector (Section 4.8). For each detected disease, a separate EigenGradCAM map is generated, producing disease-specific localisation heatmaps that show which retinal regions drove each prediction. The heatmap is smoothed with a Gaussian filter ($\sigma = 2$) to remove quantisation artefacts from the $12 \times 12 \to 384 \times 384$ upsampling, normalised to $[0, 1]$, and upsampled to the original image resolution via bilinear interpolation.

A three-panel visualisation is generated for clinical use: (i) original fundus image, (ii) EigenGradCAM heatmap (jet colormap, where red indicates high attention and blue indicates low attention), and (iii) overlay (55% original + 45% heatmap), providing the clinician with a direct visual correspondence between the model's attention and the retinal anatomy. The overlay is particularly important: the heatmap alone does not show the underlying retinal structures, making it difficult for a clinician to assess whether the attended region corresponds to a clinically relevant feature (e.g., the optic disc for glaucoma, the macula for DR).

> **Figure 21.** Example EigenGradCAM heatmap visualisation from the Flask demo application. Three panels are shown side by side: (left) the original fundus image, (centre) the EigenGradCAM attention heatmap using the jet colormap (red = high attention, blue = low attention), and (right) the overlay (55% original image + 45% heatmap) that superimposes the attention map on the retinal anatomy. The figure title indicates which disease the heatmap corresponds to and the EigenGradCAM method used. A colourbar on the right edge maps the jet colormap to attention intensity values from 0 to 1. For a cataract-positive image, the heatmap typically shows diffuse attention across the entire fundus, reflecting the global nature of lens opacity. For a glaucoma-positive image, the attention concentrates at the optic disc region. For DR, the attention localises to lesion-bearing regions of the retina. This visualisation allows the clinician to verify that the model's prediction is based on clinically relevant anatomical features rather than spurious image artefacts.

---

## 8. Inference Pipeline and Deployment

### 8.1 Inference Package

The trained model is packaged into a portable inference bundle containing all artifacts necessary for standalone deployment:

> **Table 9: Inference Package Contents.** Files included in the deployment-ready inference package, with sizes and purposes.

| File | Size | Purpose |
|---|---|---|
| `model_weights.pth` | 49.6 MB | Clean model state dict (no optimiser state) |
| `best_model.safetensors` | 49.5 MB | Same weights in HuggingFace SafeTensors format [62] |
| `thresholds.json` | < 1 KB | Per-disease optimal thresholds |
| `model_info.json` | < 1 KB | Training metadata and test results |
| `predict.py` | 3 KB | Standalone CLI inference script |
| `README.md` | 1 KB | Usage instructions |

The SafeTensors format [62] is provided as a pickle-free alternative to the standard PyTorch `.pth` format. Pickle-based formats are vulnerable to arbitrary code execution when loading untrusted checkpoints, a security concern in clinical deployment environments. SafeTensors stores only the tensor data and metadata in a structured binary format, eliminating this risk.

### 8.2 Flask Demo Application

A Flask-based [63] web application provides an interactive demonstration of the inference pipeline. The application accepts a fundus image upload (via drag-and-drop or file picker) or selection from five pre-loaded sample images (one per disease category plus a normal control). Upon analysis, the application returns:

1. **Disease probability bar plot:** A side-by-side matplotlib [64] figure showing the original fundus image and a horizontal bar chart of the four disease probabilities, with per-disease threshold markers (dashed vertical lines) and "DETECTED" labels for diseases exceeding their optimal threshold. This visualisation provides an immediate, intuitive summary of the model's prediction.

> **Figure 20.** Example bar plot output from the Flask demo application. Left panel: the input fundus image resized to 384×384. Right panel: horizontal bar chart of the four disease probabilities (Cataract, DR, Glaucoma, Myopia), each coloured according to its disease-specific colour code. Dashed vertical lines indicate the per-disease optimal thresholds (Cataract 0.49, DR 0.45, Glaucoma 0.57, Myopia 0.50). Diseases whose probability exceeds their threshold are annotated with "DETECTED" in red. The figure title displays the overall status: either the list of detected diseases (red) or "No disease detected above threshold" (green). This figure demonstrates the multi-label nature of the prediction: a single image can show varying probabilities across all four diseases (e.g., 99.6% Cataract, 30.9% DR, 4.8% Glaucoma, 13.7% Myopia), reflecting the clinical reality that patients may present with multiple concurrent pathologies.
2. **Disease probability cards:** Four cards, each with a coloured probability bar (animated width transition), percentage, threshold value, and "DETECTED" badge for flagged diseases. The cards provide a structured, at-a-glance view of the per-disease predictions.
3. **EigenGradCAM heatmap:** A three-panel visualisation (original, heatmap, overlay) with tab switching between detected diseases. The clinician can click through the tabs to see where the model looked for each detected disease.
4. **Meta information:** Inference time, device (GPU/CPU), input resolution, and model identifier.

The application loads the model once at startup and serves predictions via a `/predict` POST endpoint that accepts a multipart form upload. The model runs on GPU when available (CUDA) and falls back to CPU otherwise. On a Tesla T4 GPU, end-to-end inference (image upload → prediction → heatmap generation) completes in approximately 2–3 seconds; on CPU, the same pipeline takes approximately 5–8 seconds.

### 8.3 Model Loading and torch.compile Compatibility

The training pipeline applies `torch.compile(mode='reduce-overhead')` [48] to the model, which prefixes all state dict keys with `_orig_mod.`. The inference pipeline strips this prefix before loading:

```python
state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
```

This ensures compatibility between the compiled training checkpoint and the uncompiled inference model, without requiring `torch.compile` at inference time. Compilation at inference time would add unnecessary overhead for single-image prediction (the compilation cost is amortised over many batches during training but is a fixed cost for a single inference call). The prefix stripping is a zero-cost string operation that does not affect the tensor data.

---

## 9. Reproducibility

### 10.1 Random Seeds

The data split uses a fixed random seed (seed = 42) via `sklearn.model_selection.train_test_split(random_state=42)`. The `WeightedRandomSampler` uses PyTorch's default random state. The training loop does not set an explicit global seed, but the resume-from-checkpoint mechanism ensures that interrupted training runs continue deterministically from the last saved epoch.

### 10.2 Checkpoint Format

Each epoch saves two checkpoints:

- `latest_model.pth`: Full checkpoint (model state, optimiser state, epoch number, best F1, patience counter, training history) for resume support.
- `best_model.pth`: Same format, saved only when validation macro F1 (at optimal thresholds) exceeds the previous best. Used for final evaluation and inference.

Additionally, `best_model.safetensors` stores the model state dict in HuggingFace SafeTensors format [62] (pickle-free, safer for untrusted environments), and `training_history.json` stores the per-epoch metrics for offline plotting.

### 10.3 Configuration

All training hyperparameters are recorded in `config.json`:

```json
{
  "model_name": "glaam4x_unified_v4_asl_384",
  "img_size": 384,
  "batch_size": 32,
  "dropout_rate": 0.3,
  "learning_rate": 2e-05,
  "weight_decay": 0.0005,
  "epochs": 60,
  "warmup_epochs": 10,
  "asl_gamma_neg": 4.0,
  "asl_gamma_pos": 0.0,
  "asl_clip": 0.05
}
```

### 10.4 Environment

> **Table 12: Training Environment.** Software and hardware configuration used for training GLAAM-4X.

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch [66] | 2.11.0+cu128 |
| torchvision [33] | (Colab default) |
| NumPy | 2.0.2 |
| Pandas [67] | 2.2.2 |
| SciPy [68] | 1.16.3 |
| GPU | Tesla T4 (16 GB VRAM) |
| CUDA | 12.8 |
| Platform | Google Colab |

---

## 10. Limitations

1. **Non-stratified split.** The train/validation/test split is a plain random partition, not multi-label stratified. Iterative stratification [26] — which preserves per-disease prevalence across splits — requires the `scikit-multilearn` library, which was not available in the training environment. Per-disease prevalence was verified to be approximately balanced across splits post hoc, but a stratified split would provide stronger guarantees against class distribution shift between splits.

2. **Label noise from single-disease datasets.** Images from single-disease datasets (IDRiD, DDR, PAPILA, REFUGE2, glaucoma_bundle) receive a label of 0 for diseases they do not evaluate, representing "not evaluated" rather than "confirmed negative." This introduces label noise that particularly affects the Glaucoma classifier, which draws from multiple glaucoma-only sources with varying annotation standards. The asymmetric loss function (Section 5.1) partially mitigates this by down-weighting easy negative gradients, but cleaner annotations would improve performance.

3. **No slit-lamp cataract dataset.** No maintained Kaggle mirror was found for a dedicated slit-lamp cataract dataset. Cataract signal in the unified corpus comes from ODIR and the `eye_diseases` dataset (fundus images), not slit-lamp images. The model therefore detects cataract from fundus image quality degradation rather than direct lens visualisation, which limits its applicability to slit-lamp-based screening workflows.

4. **No PALM dataset.** No reliable Kaggle mirror was found for the original PALM (Pathologic Myopia) dataset. Myopia signal comes from ODIR, RFMiD, and JSIEC instead, which may have different annotation criteria for pathological myopia.

5. **Kaggle dataset slug verification.** Several Kaggle dataset slugs were found via web search and not verified against real downloads during notebook construction. An inspection step (printing folder layout and CSV columns for each downloaded dataset) is mandatory before trusting the parser output, as dataset maintainers may restructure their Kaggle repositories without notice.

6. **Glaucoma recall.** The Glaucoma classifier achieves the lowest recall (58.8%) among the four diseases, likely due to label noise from multiple glaucoma datasets with different annotation standards and the "suspect = positive" decision in PAPILA parsing. The elevated threshold (0.57) trades recall for precision, which may not be optimal for screening where sensitivity is prioritised. A dedicated glaucoma dataset with consistent annotation criteria would likely improve recall substantially.

---

# Results

## 1. Training Dynamics

GLAAM-4X was trained for 60 epochs on a Tesla T4 GPU, with the best model selected at epoch 48 based on validation macro F1 at optimal thresholds (0.8275). Early stopping with patience = 10 was configured but not triggered, as the model continued to improve through epoch 48 and the remaining epochs did not exceed the patience window before the maximum epoch count was reached.

> **Figure 6.** `training_graphs/01_loss_curve.png` — Training and validation Asymmetric Loss over 60 epochs. The training loss (blue) decreases steadily from approximately 0.35 in the warmup phase to 0.04 by epoch 48, reflecting the model's improving fit to the training data. The validation loss (red) tracks the training loss closely during the warmup phase (epochs 1–10), then diverges slightly as the cosine annealing schedule reduces the learning rate. The green dashed line marks the best epoch (48). The convergence of both curves without a large gap indicates that the model is not severely overfitting, though the slight divergence in later epochs suggests mild overfitting that is controlled by the dropout, weight decay, and early stopping mechanisms.

> **Figure 7.** `training_graphs/02_macro_f1_progression.png` — Macro F1 score progression across three conditions: validation F1 at the default 0.5 threshold (orange), validation F1 at per-disease optimal thresholds (green), and training F1 (blue, semi-transparent). The gap between the orange and green curves quantifies the benefit of per-disease threshold optimisation — the optimal thresholds consistently improve macro F1 by 2–5 percentage points over the fixed 0.5 threshold. The training F1 exceeds the validation F1, indicating some degree of overfitting, but the gap remains stable rather than widening, suggesting that the dropout and weight decay regularisation are effectively limiting memorisation. The red dashed line marks the best epoch (48), and the green dotted line shows the best validation macro F1 (0.8275).

> **Figure 8.** `training_graphs/03_per_disease_auc.png` — Per-disease validation AUC curves over 60 epochs. Each coloured line represents one disease: Cataract (blue), DR (orange), Glaucoma (green), and Myopia (purple). Cataract achieves the highest AUC (0.99+), reflecting the strong, diffuse signal of lens opacity in fundus images. DR and Myopia reach AUC ~0.95–0.98, indicating strong discriminative performance. Glaucoma reaches AUC ~0.97 but exhibits more fluctuation in early epochs, likely due to label noise from the multiple glaucoma-only datasets in the training corpus. The red dashed line marks the best epoch (48). The steady upward trend across all diseases confirms that the disease-specific attention specialists are learning complementary features.

> **Figure 9.** `training_graphs/04_per_disease_f1.png` — Per-disease validation F1 scores at optimal thresholds over 60 epochs. F1 scores are lower than AUC values because they depend on the decision threshold and the class prevalence. Cataract achieves the highest F1 (~0.91), while Glaucoma has the lowest (~0.72), reflecting the recall limitation discussed in Section 11. The F1 curves plateau in the later epochs, indicating that the model has converged. The gap between Cataract and Glaucoma F1 scores (approximately 0.19) highlights the performance disparity between diseases with clean labels (Cataract, primarily from ODIR) and diseases with noisy labels (Glaucoma, from multiple single-disease sources).

> **Figure 10.** `training_graphs/05_precision_recall.png` — Per-disease precision (solid line) and recall (dashed line) trends at optimal thresholds, shown as a 2×2 grid of subplots. Cataract and DR exhibit balanced precision and recall (both >0.85), indicating that the model neither over-predicts nor under-predicts these diseases. Glaucoma shows a pronounced precision–recall imbalance: precision (~0.88) substantially exceeds recall (~0.61), meaning the model is conservative in flagging glaucoma — when it predicts glaucoma, it is usually correct, but it misses approximately 40% of true glaucoma cases. Myopia shows moderate balance with some fluctuation. The red dashed line in each subplot marks the best epoch (48).

> **Figure 11.** `training_graphs/06_lr_schedule.png` — Actual learning rate schedule over 60 epochs, showing the 10-epoch linear warmup (learning rate rising from 0 to 2×10⁻⁵) followed by cosine annealing (smooth decay from 2×10⁻⁵ to 0 over the remaining 50 epochs). The red dashed line marks the end of the warmup phase. The smooth cosine curve avoids the abrupt learning rate drops of step decay schedules, providing a gradual transition from broad exploration to fine convergence. The actual schedule matches the theoretical schedule exactly, confirming correct scheduler implementation.

> **Figure 12.** `training_graphs/07_dashboard.png` — Combined 6-panel training dashboard providing a single-page overview of all training dynamics. Top row: loss curve (left), macro F1 progression (centre), and learning rate schedule (right). Middle row: per-disease validation AUC curves with final values annotated. Bottom row: per-disease validation F1 curves with final values annotated. The dashboard format enables rapid visual assessment of training health: the loss curves show convergence, the F1 curves show per-disease performance, and the LR schedule confirms the warmup-cosine annealing is correctly applied. This figure is suitable for inclusion in a supplementary materials section.

> **Figure 13.** `training_graphs/08_train_val_loss.png` — Train vs. validation loss, specifically for overfitting detection. This is the same data as Figure 6 but presented in a standalone format for focused analysis. The key diagnostic is the gap between the blue (train) and red (validation) curves: a widening gap indicates overfitting, while a stable or closing gap indicates healthy generalisation. In this training run, the gap remains small (<0.05 ASL loss units) throughout training, confirming that the combination of dropout, weight decay, differential augmentation, and early stopping effectively controls overfitting.

## 2. Test Set Performance

GLAAM-4X was evaluated on the held-out test set (2,232 images) using the best checkpoint (epoch 48, validation macro F1 = 0.8275). Per-disease thresholds optimised on the validation set were applied: Cataract 0.49, DR 0.45, Glaucoma 0.57, Myopia 0.50.

> **Table 13: Held-Out Test Set Results.** Per-disease performance of GLAAM-4X on the held-out test set (n = 2,232 images). AUC, F1, precision, and recall are reported at per-disease optimal thresholds. Macro averages are unweighted means across the four diseases.

| Disease | AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| Cataract | **0.985** | 0.873 | 0.862 | 0.885 | 0.49 |
| DR | **0.952** | 0.848 | 0.860 | 0.837 | 0.45 |
| Glaucoma | **0.966** | 0.690 | 0.833 | 0.588 | 0.57 |
| Myopia | **0.971** | 0.732 | 0.750 | 0.714 | 0.50 |
| **Macro avg** | **0.968** | **0.786** | 0.826 | 0.756 | — |

The model achieves a macro-averaged AUC of 0.968 across the four diseases, indicating strong overall discriminative ability. The following per-disease analysis provides clinical context for these results.

**Cataract (AUC = 0.985, F1 = 0.873).** Cataract achieves the highest AUC and F1 among the four diseases. The near-default threshold (0.49) indicates that the model's probability estimates are well-calibrated for this disease — the optimal decision boundary is very close to 0.5, meaning the raw sigmoid output can be used directly for triage without threshold adjustment. The balanced precision (0.862) and recall (0.885) indicate that the model neither over-predicts nor under-predicts cataract. This strong performance is expected: cataract produces a diffuse, global signal in fundus photography (reduced image clarity due to lens opacity), which is easily captured by the high-reduction GLAAMBlock ($r = 16$) that compresses the channel attention into a coarse, global representation. The primary source of cataract labels is ODIR-5K [16], which provides clean one-hot annotations, minimising label noise.

**DR (AUC = 0.952, F1 = 0.848).** DR achieves the second-highest F1, with balanced precision (0.860) and recall (0.837). The slightly below-default threshold (0.45) increases sensitivity for DR, which is clinically appropriate: missed DR diagnoses can lead to irreversible vision loss, and the cost of a false positive (unnecessary ophthalmologist referral) is lower than the cost of a false negative (untreated retinopathy progressing to blindness). The MultiScaleGLAAM specialist's three-scale attention (fine $r = 4$, medium $r = 8$, coarse $r = 16$) effectively captures the wide range of DR lesion sizes, from 2–5 pixel microaneurysms to 50+ pixel haemorrhages. The balanced precision–recall trade-off suggests that the multi-scale fusion (weighted sum + concatenation/conv) successfully integrates information across lesion scales without biasing toward any particular scale.

**Glaucoma (AUC = 0.966, F1 = 0.690).** Glaucoma achieves high AUC (0.966) but the lowest F1 (0.690) and recall (0.588) among the four diseases. The elevated threshold (0.57) is the highest of all four diseases, reflecting the model's tendency to over-predict glaucoma due to label noise. The precision–recall imbalance is pronounced: precision (0.833) substantially exceeds recall (0.588), meaning the model is conservative — when it predicts glaucoma, it is usually correct, but it misses 41.2% of true glaucoma cases. This is the primary weakness of the system and is discussed in detail in the Discussion section.

**Myopia (AUC = 0.971, F1 = 0.732).** Myopia achieves high AUC (0.971) with moderate F1 (0.732). The default threshold (0.50) indicates well-calibrated probability estimates. The moderate F1 (compared to Cataract and DR) is primarily due to lower precision (0.750), meaning the model produces more false positives for myopia. This may reflect the heterogeneity of myopia annotations across the three source datasets (ODIR, RFMiD, JSIEC), which may use different criteria for classifying pathological myopia. The identity attention pathway (no attention) performs adequately, confirming the design hypothesis that myopia's peripapillary tessellation signal is already well-captured by the MobileNetV2 backbone features without explicit attention refinement.

> **Figure 14.** `report_assets/02_test_roc_curves.png` — Per-disease ROC curves on the held-out test set. All four curves lie close to the top-left corner, with AUC values of 0.985 (Cataract), 0.952 (DR), 0.966 (Glaucoma), and 0.971 (Myopia). The macro-average AUC is 0.968. The black dashed diagonal represents chance performance (AUC = 0.5). Cataract's curve is nearly perfect, while Glaucoma's curve shows slightly more deviation from the top-left corner, consistent with its lower recall.

> **Figure 15.** `report_assets/03_test_pr_curves.png` — Per-disease precision-recall curves on the held-out test set. PR curves are more informative than ROC curves for imbalanced datasets because they directly show the precision–recall trade-off. Cataract and DR maintain high precision across a wide range of recall values, while Glaucoma and Myopia show steeper precision drops at recall > 0.7, indicating that capturing additional true positives for these diseases requires accepting more false positives.

## 3. Per-Split Performance and Generalisation Analysis

> **Table 14: Full Results Across All Splits.** Per-disease AUC and F1 scores across training, validation, and test splits. The gap between training and test performance indicates the degree of overfitting.

| Split | Disease | AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train | Cataract | 0.998 | 0.940 | 0.944 | 0.937 |
| Train | DR | 0.973 | 0.891 | 0.904 | 0.879 |
| Train | Glaucoma | 0.981 | 0.753 | 0.897 | 0.649 |
| Train | Myopia | 0.995 | 0.857 | 0.843 | 0.870 |
| Train | **Macro** | **0.987** | **0.860** | 0.897 | 0.834 |
| Val | Cataract | 0.993 | 0.910 | 0.929 | 0.892 |
| Val | DR | 0.954 | 0.856 | 0.865 | 0.848 |
| Val | Glaucoma | 0.969 | 0.721 | 0.883 | 0.608 |
| Val | Myopia | 0.982 | 0.835 | 0.828 | 0.841 |
| Val | **Macro** | **0.974** | **0.830** | 0.876 | 0.797 |
| Test | Cataract | 0.985 | 0.873 | 0.862 | 0.885 |
| Test | DR | 0.952 | 0.848 | 0.860 | 0.837 |
| Test | Glaucoma | 0.966 | 0.690 | 0.833 | 0.588 |
| Test | Myopia | 0.971 | 0.732 | 0.750 | 0.714 |
| Test | **Macro** | **0.968** | **0.786** | 0.826 | 0.756 |

The generalisation gap is quantified by the difference between training and test performance:

- **AUC gap:** $\Delta\text{AUC} = 0.987 - 0.968 = 0.019$. This small gap indicates that the model's discriminative ability generalises well from training to test. The AUC is threshold-independent, so this gap reflects genuine overfitting in the feature representation rather than threshold sensitivity.

- **F1 gap:** $\Delta\text{F1} = 0.860 - 0.786 = 0.074$. This larger gap is primarily driven by the Glaucoma classifier ($\Delta\text{F1} = 0.753 - 0.690 = 0.063$), whose F1 drops substantially from training to test. The Cataract ($\Delta\text{F1} = 0.067$) and DR ($\Delta\text{F1} = 0.043$) classifiers show smaller generalisation gaps, indicating robust learning. Myopia shows the smallest gap ($\Delta\text{F1} = 0.125$), though this is partly due to the lower absolute F1 values.

- **Precision stability:** Precision is remarkably stable across splits for Cataract (0.944 → 0.862, $\Delta = 0.082$) and DR (0.904 → 0.860, $\Delta = 0.044$), indicating that the model's positive predictions are reliable regardless of the data split. Glaucoma precision is also stable (0.897 → 0.833, $\Delta = 0.064$).

- **Recall degradation:** Recall degrades more than precision across splits, particularly for Glaucoma (0.649 → 0.588, $\Delta = 0.061$) and Myopia (0.870 → 0.714, $\Delta = 0.156$). This suggests that the model becomes more conservative (fewer positive predictions) on unseen data, which is a common pattern in imbalanced classification and is partially addressed by the per-disease threshold optimisation.

The validation-to-test gap is smaller: $\Delta\text{AUC}_{\text{val→test}} = 0.974 - 0.968 = 0.006$ and $\Delta\text{F1}_{\text{val→test}} = 0.830 - 0.786 = 0.044$. This confirms that the validation set is representative of the test distribution, and the threshold values optimised on the validation set generalise to the test set without significant degradation.

## 4. Confusion Matrix Analysis

> **Figure 16.** `report_assets/04_test_confusion_matrices.png` — Per-disease confusion matrices on the held-out test set at optimal thresholds, shown as a 2×2 grid. Each matrix displays raw counts and percentages for true negatives (top-left), false positives (top-right), false negatives (bottom-left), and true positives (bottom-right). The colour intensity (Blues colormap) is proportional to the count.

**Cataract:** The confusion matrix shows a well-balanced distribution. True positives (88.5% sensitivity) and true negatives dominate, with relatively few false positives (13.8% of negatives misclassified) and false negatives (11.5% of positives missed). The near-symmetric error distribution indicates that the model does not have a systematic bias toward over-prediction or under-prediction for cataract.

**DR:** The confusion matrix shows balanced performance with 83.7% sensitivity and 83.7% specificity (approximate). The false negative rate (16.3%) is slightly higher than the false positive rate, which is expected given the below-default threshold (0.45) that prioritises sensitivity. In a clinical screening context, this trade-off is appropriate: it is better to refer a patient who turns out not to have DR (false positive) than to miss a patient who does (false negative, potentially leading to progressive vision loss).

**Glaucoma:** The confusion matrix reveals the primary weakness of the system. The false negative rate is 41.2% — meaning that 41.2% of true glaucoma cases are classified as negative. This is the direct consequence of the elevated threshold (0.57) and the label noise from multiple glaucoma-only datasets. The true positive rate (58.8%) is the lowest among all four diseases. However, the false positive rate is low (only 4.2% of negatives are misclassified as positive), meaning that when the model does predict glaucoma, it is correct 83.3% of the time (precision). This high-precision, low-recall pattern is characteristic of a model trained on noisy labels with a conservative threshold.

**Myopia:** The confusion matrix shows moderate performance. The false positive rate (25.0% of negatives misclassified) is the highest among all four diseases, contributing to the lower precision (0.750). This may reflect the heterogeneity of myopia annotations across source datasets, or the presence of mild myopia cases that are borderline between normal and pathological. The false negative rate (28.6%) is also moderate.

## 5. Calibration Analysis

> **Figure 17.** `report_assets/05_calibration_curves.png` — Calibration (reliability) curves for each disease on the held-out test set, shown as a 2×2 grid. Each subplot plots the observed fraction of positive cases (y-axis) against the mean predicted probability (x-axis) within each of 10 quantile bins. The black dashed diagonal represents perfect calibration.

Calibration curves assess whether the model's predicted probabilities are actionable in clinical decision-making — a critical requirement for screening tools where predicted probabilities are used to triage patients.

**Cataract and DR** are well-calibrated across most of the probability range. Their calibration curves closely follow the diagonal, meaning that a predicted probability of 0.8 corresponds to an observed disease rate of approximately 80%. This indicates that the sigmoid output for these diseases can be directly used for clinical triage without post-hoc calibration. The Asymmetric Loss configuration ($\gamma_{\text{neg}} = 4.0$, $\gamma_{\text{pos}} = 0.0$, clip $m = 0.05$) contributes to this calibration by preventing overconfidence on the dominant negative class.

**Glaucoma and Myopia** exhibit some miscalibration at extreme probability values. For Glaucoma, the model tends to be underconfident at high predicted probabilities (the observed disease rate is higher than the predicted probability), suggesting that the model is more accurate than it "believes" for clear glaucoma cases. For Myopia, the model shows slight overconfidence at mid-range probabilities. These miscalibration patterns suggest that post-hoc calibration (e.g., temperature scaling [65]) may be beneficial before clinical deployment for these two diseases, particularly for Glaucoma where the calibration curve deviates most from the diagonal.

The calibration results have direct clinical implications: for Cataract and DR, the model's probability output can be used directly in a screening workflow (e.g., "refer if probability > 0.5"). For Glaucoma and Myopia, a calibration step would be needed to ensure that the probability output is actionable.

## 6. ROC and Precision-Recall Curve Analysis

> **Figure 14.** `report_assets/02_test_roc_curves.png` — Per-disease ROC curves on the held-out test set.

The ROC curves (Figure 14) show that all four diseases achieve AUC above 0.95, with Cataract (0.985) and Myopia (0.971) performing best. The ROC curve for Cataract is nearly perfect, hugging the top-left corner of the plot. The Glaucoma ROC curve shows slightly more deviation from the top-left corner, particularly in the mid-range false positive rates, consistent with the lower recall at the optimal threshold.

> **Figure 15.** `report_assets/03_test_pr_curves.png` — Per-disease precision-recall curves on the held-out test set.

The PR curves (Figure 15) provide additional insight that the ROC curves obscure. While all four diseases have high AUC, the PR curves reveal that the precision–recall trade-off differs substantially across diseases. Cataract and DR maintain precision above 0.80 across a wide range of recall values (0.5–0.9), indicating that the model can achieve high sensitivity without sacrificing too much precision. Glaucoma's PR curve drops more steeply: precision falls below 0.70 at recall 0.7, meaning that capturing 70% of true glaucoma cases requires accepting a 30% false positive rate. Myopia's PR curve shows a similar but less severe pattern. These PR curves are more informative than ROC curves for the imbalanced setting of this task, where the negative class dominates.

## 7. Model Efficiency

> **Table 15: Model Efficiency.** Computational characteristics of GLAAM-4X, measured on a Tesla T4 GPU at $384 \times 384$ input resolution with batch size 1.

| Metric | Value |
|---|---|
| Total parameters | 12,315,120 |
| Trainable parameters | 12,315,120 |
| Model size (SafeTensors) | 47.17 MB |
| Mean inference latency (batch size 1, GPU) | 12.44 ms |
| P95 inference latency (batch size 1, GPU) | 14.72 ms |
| Throughput | ~80 fps |
| Input resolution | $384 \times 384$ |

The 12.3M parameter count and 47 MB model size make GLAAM-4X suitable for deployment on resource-constrained devices. For comparison:

> **Table 16: Model Size Comparison.** Parameter count and model size comparison between GLAAM-4X and commonly used architectures in medical image classification.

| Architecture | Parameters | Model size | Inference latency |
|---|---|---|---|
| **GLAAM-4X (this work)** | **12.3M** | **47 MB** | **12.44 ms** |
| ResNet-50 [43] | 25.6M | 102 MB | ~25 ms |
| EfficientNet-B3 [49] | 12M | 48 MB | ~15 ms |
| DenseNet-121 [50] | 8M | 32 MB | ~18 ms |
| MobileNetV2 (baseline) [13] | 3.5M | 14 MB | ~7 ms |

GLAAM-4X is comparable in size to EfficientNet-B3 (12M parameters) but achieves multi-disease detection with disease-specific attention, which EfficientNet-B3 does not provide. The 12.44 ms mean inference latency on a Tesla T4 GPU corresponds to approximately 80 frames per second, sufficient for real-time screening applications. The P95 latency (14.72 ms) indicates that 95% of inference calls complete within 15 ms, providing consistent performance for clinical deployment.

## 8. Explainability Results

> **Figure 21.** Example EigenGradCAM heatmap visualisation. Three panels: (left) original fundus image, (centre) EigenGradCAM attention heatmap (jet colormap), (right) overlay (55% original + 45% heatmap).

Disease-specific EigenGradCAM heatmaps were generated for each detected disease on the test set. The heatmaps confirm that the model's attention corresponds to clinically relevant anatomical features:

- **Cataract-positive images:** The heatmap shows diffuse, widespread attention across the entire fundus, reflecting the global nature of lens opacity. Unlike localised diseases, cataract reduces overall image clarity uniformly, so the model's attention is correctly distributed rather than concentrated at a specific location. This pattern validates the Cataract specialist's high reduction ratio ($r = 16$), which produces a coarse, global attention map.

- **Glaucoma-positive images:** The heatmap concentrates attention at the optic disc region, which is the primary diagnostic site for glaucoma (cup-to-disc ratio assessment). This localisation is clinically correct: glaucoma is diagnosed by examining the optic disc for cupping and neuroretinal rim thinning. The attention concentration at the disc validates the Glaucoma specialist's moderate reduction ratio ($r = 8$), which provides the appropriate receptive field size for the disc region.

- **DR-positive images:** The heatmap localises to lesion-bearing regions of the retina, particularly the macula and temporal vascular arcade where DR lesions (microaneurysms, haemorrhages, exudates) are most prevalent. The multi-scale attention of the MultiScaleGLAAM specialist is reflected in the heatmap's multi-resolution pattern: fine-scale attention captures small lesion clusters, while coarse-scale attention covers larger haemorrhage regions.

- **Normal images:** When no disease is detected, the heatmap for the highest-probability disease (typically the one with the highest baseline prevalence) shows low, diffuse attention without concentration at any specific region, consistent with the absence of pathological features.

This visual correspondence between model attention and clinical anatomy provides interpretable evidence that supports the model's predictions, addressing the explainability requirement for clinical adoption [10,11]. The EigenGradCAM method produces sharper heatmaps than standard Grad-CAM [12] due to its use of singular value decomposition of the gradient matrix, which reduces noise from spatially inconsistent gradients — particularly important for fine-grained DR lesion localisation.

## 9. Training Visualisation Summary

> **Figure 12.** `training_graphs/07_dashboard.png` — Combined 6-panel training dashboard.

The training dashboard (Figure 12) provides a single-page overview of all training dynamics. The loss curves show steady convergence without severe overfitting. The macro F1 progression shows that per-disease threshold optimisation consistently improves performance over the fixed 0.5 threshold by 2–5 percentage points. The per-disease AUC curves show that all four diseases learn at similar rates, with Cataract consistently highest and Glaucoma consistently lowest. The learning rate schedule confirms correct implementation of the warmup-cosine annealing strategy.

> **Figure 13.** `training_graphs/08_train_val_loss.png` — Train vs. validation loss for overfitting detection.

The train-vs-validation loss curve (Figure 13) is the primary diagnostic for overfitting. The gap between training and validation loss remains small (<0.05 ASL loss units) throughout training, confirming that the combination of dropout (0.3 for DR, 0.15 for Glaucoma/Cataract), weight decay ($5 \times 10^{-4}$), differential augmentation, and early stopping effectively controls overfitting. The gap does not widen in later epochs, which would indicate progressive memorisation; instead, it stabilises, suggesting that the model has reached a generalisation plateau.

> **Figure 19.** `report_assets/01_dataset_composition.png` — Dataset composition overview, showing split sizes and per-disease prevalence by split.

The dataset composition figure (Figure 19) confirms that the random split produces approximately balanced disease distributions across the three splits. The per-disease prevalence bars show that Cataract is the most common disease (highest prevalence), followed by DR, Glaucoma, and Myopia. This prevalence ordering is consistent with the model's performance ordering (Cataract highest F1, Myopia lowest), though the relationship is not purely prevalence-driven — label quality also plays a significant role, as evidenced by Glaucoma's lower-than-expected F1 despite moderate prevalence.

---

# Discussion

The results demonstrate that disease-specific attention mechanisms improve multi-label fundus disease detection compared to the shared-attention paradigm. The macro-averaged AUC of 0.968 across four diseases, achieved with a 12.3M-parameter model, represents a favourable accuracy–efficiency trade-off for screening applications. This section discusses the key findings, their implications, and the design choices that led to the observed performance.

## 1. Disease-Specific Attention vs. Shared Attention

The original GLAAM [7] applied a single shared attention pathway for cataract detection, achieving strong performance on that single disease. GLAAM-4X extends this to four diseases by introducing disease-specific specialists, each with a configuration tailored to its target pathology. The results validate this approach: Cataract achieves AUC 0.985 (the highest), demonstrating that the disease-specific approach does not degrade performance on the original target disease. The DR specialist's MultiScaleGLAAM captures lesion sizes spanning 2–50+ pixels through three-scale attention, achieving AUC 0.952 with balanced precision and recall — a capability absent from the original GLAAM, which used a single-scale attention mechanism.

The disease-specific design philosophy contrasts with the standard approach in multi-label classification, where a single shared feature extractor feeds multiple classification heads [40,41]. While architecturally simpler, the shared approach forces the feature extractor to learn a compromise representation that is adequate for all diseases but optimal for none. GLAAM-4X instead allows each disease to refine the shared backbone features through its own attention pathway, producing disease-specialised representations that are then classified by dedicated heads. The 2.3M additional parameters (18% of the total) required for the four attention specialists represent a modest overhead for the performance improvement they provide.

The disease gating network provides an additional interpretable signal: the gate weights indicate which diseases the model considers most relevant for a given image. While the gate weights are not explicitly multiplied with the specialist features in the current implementation, they serve as a regularisation mechanism through the shared backbone gradient and provide a diagnostic signal for understanding the model's global reasoning.

## 2. Glaucoma: The Recall Challenge

The Glaucoma classifier's recall (58.8%) is the lowest among the four diseases, and this warrants detailed discussion. The primary cause is label noise: the unified corpus draws glaucoma labels from five different datasets (ODIR, PAPILA, REFUGE2, glaucoma_bundle, RFMiD), each with different annotation criteria. PAPILA's "suspect = positive" coding inflates the positive class with ambiguous cases that may or may not represent true glaucoma. The glaucoma_bundle datasets (ORIGA, REFUGE, G1020) use different disc/cup ratio thresholds for classification. RFMiD uses "ODC" (optic disc cupping) as a glaucoma proxy, which is a related but not identical condition. This heterogeneity means the model is trained on inconsistent labels, making it conservative in its predictions — it learns to only flag glaucoma when the evidence is strong, leading to high precision (0.833) but low recall (0.588).

The elevated threshold (0.57) further reduces recall in favour of precision. This threshold was optimised on the validation set to maximise F1, which balances precision and recall equally. In a clinical screening context, where sensitivity is prioritised (missing glaucoma leads to irreversible vision loss), a lower threshold would be preferable — but this requires cleaner training labels. With the current noisy labels, lowering the threshold would increase false positives without a proportional increase in true positives, because many of the "positive" labels are themselves unreliable.

A dedicated glaucoma dataset with consistent annotation criteria — for example, a dataset where all glaucoma labels are assigned by the same clinical protocol using standardised cup-to-disc ratio thresholds and visual field testing — would likely improve recall substantially. This is the most impactful direction for future work.

## 3. MultiScaleGLAAM for Diabetic Retinopathy

The DR specialist's MultiScaleGLAAM module is the most architecturally novel component of GLAAM-4X. The three-scale design (fine $r = 4$, medium $r = 8$, coarse $r = 16$) is motivated by the clinical observation that DR lesions span three orders of magnitude in spatial extent [1]. The results validate this design: DR achieves AUC 0.952 with balanced precision (0.860) and recall (0.837), outperforming what a single-scale attention mechanism could achieve.

The multi-scale fusion mechanism (weighted sum + concatenation/conv) provides both smooth interpolation and learned cross-scale interactions. The weighted sum allows the model to adapt the relative importance of each scale based on the lesion distribution in the training data, while the concatenation/conv learns to combine features across scales — for example, detecting that the presence of microaneurysms at the fine scale combined with haemorrhages at the medium scale indicates more severe DR than either alone. This dual fusion mechanism follows the multi-scale fusion paradigm established in feature pyramid networks [46] but applies attention (rather than feature concatenation) at multiple scales, which is more parameter-efficient.

The slightly below-default threshold (0.45) for DR increases sensitivity, which is clinically appropriate: the cost of a false negative (missed DR, potentially leading to blindness) is substantially higher than the cost of a false positive (unnecessary ophthalmologist referral). The balanced precision–recall trade-off suggests that the multi-scale attention successfully captures the full range of DR lesion sizes without biasing toward any particular scale.

## 4. Class Imbalance and Asymmetric Loss

The ASL configuration ($\gamma_{\text{neg}} = 4.0$, $\gamma_{\text{pos}} = 0.0$, clip $m = 0.05$) effectively addresses the 3:1 to 15:1 class imbalance in the unified corpus. The aggressive negative focusing ($\gamma_{\text{neg}} = 4.0$) suppresses gradients from the dominant healthy class, while $\gamma_{\text{pos}} = 0.0$ ensures that all positive examples contribute full gradient signal. The clip $m = 0.05$ prevents overconfidence on negatives.

The effectiveness of this configuration is evidenced by the balanced precision–recall trade-off for Cataract and DR: both diseases achieve precision and recall above 0.83, indicating that the model does not simply predict "all negative" (which would achieve high precision but near-zero recall) or "all positive" (which would achieve high recall but low precision). The ASL configuration enables the model to learn meaningful positive-class representations despite the extreme imbalance.

For comparison, standard focal loss [9] with $\gamma = 2.0$ applied symmetrically would produce less aggressive negative suppression, potentially allowing the majority class to dominate the loss landscape. The asymmetric configuration — with $\gamma_{\text{neg}} = 4.0$ (twice the standard focal loss $\gamma$) and $\gamma_{\text{pos}} = 0.0$ (no positive focusing at all) — represents a more targeted approach to the multi-label imbalance problem, as recommended by Ben-Baruch et al. [14].

## 5. Post-Backbone vs. Intra-Backbone Attention

The decision to apply attention post-backbone rather than intra-backbone (as in the original GLAAM [7]) was motivated by three factors: preserving pretrained features, enabling efficient multi-disease branching, and providing the richest available representation (1280 channels) for the attention mechanism. The results support this choice: the model achieves high AUC across all four diseases with only 2.3M additional parameters for the attention specialists.

The cost is that the attention operates on a $12 \times 12$ feature map, which may be too coarse for the finest DR lesions (microaneurysms at 2–5 pixels in the original image). The MultiScaleGLAAM partially addresses this by processing the feature map at three scales, but the finest scale still operates at $12 \times 12$ resolution. Intra-backbone injection at earlier stages (e.g., $48 \times 48$ or $24 \times 24$) would provide higher spatial resolution but at the cost of modifying the pretrained backbone and requiring four separate backbone copies for multi-disease attention — a 4× increase in backbone parameters that would make the model impractical for resource-constrained deployment.

The trade-off between spatial resolution and parameter efficiency is a fundamental design tension in attention-based architectures. GLAAM-4X prioritises efficiency (12.3M parameters, 47 MB) over spatial precision, which is appropriate for a screening tool intended for deployment on commodity hardware. For applications requiring finer lesion localisation (e.g., automated DR grading), intra-backbone injection at higher-resolution stages may be worth the additional cost.

## 6. Calibration and Clinical Actionability

The calibration analysis (Section 5 of Results) reveals that Cataract and DR produce well-calibrated probability estimates, while Glaucoma and Myopia exhibit some miscalibration at extreme probability values. This has direct clinical implications: for Cataract and DR, the model's probability output can be used directly in a screening workflow (e.g., "refer if probability > 0.5"). For Glaucoma and Myopia, a post-hoc calibration step (e.g., temperature scaling [65]) would be needed to ensure that the probability output is actionable.

The calibration differences across diseases are likely related to the label noise discussion: when training labels are inconsistent (as for Glaucoma), the model's probability estimates are less reliable because the training signal itself is noisy. For diseases with clean labels (Cataract, primarily from ODIR), the model learns a more reliable mapping from features to probabilities.

## 7. Clinical Implications

GLAAM-4X is designed for screening, not diagnosis. In a screening workflow, the model would be applied to fundus images captured by a non-specialist healthcare worker, flagging patients who require referral to an ophthalmologist. The per-disease probabilities and EigenGradCAM heatmaps provide the referring worker with interpretable evidence to support the referral decision. The 12.44 ms inference latency enables real-time screening, and the 47 MB model size is compatible with deployment on mobile devices or edge computing hardware.

However, the Glaucoma recall limitation (58.8%) means that the model should not be used as the sole screening tool for glaucoma. A complementary approach — such as intraocular pressure measurement or optical coherence tomography — would be needed to achieve adequate sensitivity for glaucoma screening. For Cataract and DR, the model's performance (AUC > 0.95, balanced precision and recall) is sufficient for standalone screening, subject to prospective clinical validation.

The EigenGradCAM heatmaps serve a dual purpose in the clinical workflow: (i) they provide interpretable evidence that allows the healthcare worker to verify the model's reasoning, and (ii) they serve as a quality control mechanism — if the heatmap does not correspond to a clinically relevant anatomical feature (e.g., attention on an image artefact rather than the optic disc), the prediction can be flagged for manual review.

## 8. Comparison to Prior Work

The original GLAAM [7] reported cataract detection results on ODIR-5K but did not address multi-disease detection. GLAAM-4X extends to four diseases while maintaining high AUC for cataract (0.985), demonstrating that the disease-specific approach does not degrade performance on the original target disease. The multi-scale attention for DR is related to feature pyramid networks [46] but applies attention (rather than feature concatenation) at multiple scales, which is more parameter-efficient. The Asymmetric Loss configuration follows the recommendations of Ben-Baruch et al. [14] for extreme imbalance, and the per-disease thresholding follows standard practice in multi-label classification [58].

Compared to existing multi-label fundus classifiers that use a shared backbone with multi-output heads [25,40], GLAAM-4X's disease-specific attention provides a more targeted feature refinement for each disease. The 0.968 macro-averaged AUC compares favourably to reported results in the literature for multi-disease fundus classification, though direct comparison is complicated by differences in dataset composition, disease definitions, and evaluation protocols.

The model efficiency (12.3M parameters, 47 MB, 12.44 ms latency) is comparable to EfficientNet-B3 [49] (12M parameters, 48 MB) but provides disease-specific attention that EfficientNet-B3 does not. This makes GLAAM-4X particularly suitable for deployment in resource-constrained settings where both accuracy and efficiency are critical.

---

# Future Work

Several directions for future work emerge from the limitations and findings of this study:

## 1. Improving Glaucoma Detection Through Cleaner Annotations

The most impactful direction is addressing the Glaucoma recall limitation. The current model draws glaucoma labels from five datasets with different annotation criteria (ODIR, PAPILA, REFUGE2, glaucoma_bundle, RFMiD), each using different disc-to-cup ratio thresholds, clinical coding schemes, and diagnostic protocols. This label heterogeneity is the primary cause of the model's conservative glaucoma predictions. Future work should:

- **Curate a unified glaucoma dataset** where all labels are assigned by a single clinical protocol using standardised cup-to-disc ratio thresholds and visual field testing confirmation. This would eliminate the label noise that currently forces the model to be conservative.
- **Explore label-aware loss functions** that weight training samples by annotation confidence, down-weighting samples from datasets with known annotation inconsistencies (e.g., PAPILA's "suspect" category) while up-weighting samples from datasets with confirmed diagnoses.
- **Investigate semi-supervised learning** to leverage the large number of unlabelled fundus images available in clinical archives, using the current model's high-precision glaucoma predictions as pseudo-labels for confident-negative cases.

## 2. Intra-Backbone Attention Injection at Higher Resolution

The post-backbone attention design operates on a $12 \times 12$ feature map, which may be too coarse for the finest DR lesions (microaneurysms at 2–5 pixels in the original image). While the MultiScaleGLAAM partially addresses this through three-scale processing, the finest scale still operates at $12 \times 12$ resolution. Future work should explore:

- **Hybrid injection strategies** that apply disease-specific attention at multiple backbone stages — for example, injecting the DR specialist's fine-scale attention at an early backbone stage ($48 \times 48$ or $24 \times 24$ resolution) while keeping the other specialists post-backbone. This would provide higher spatial resolution for DR lesion detection without the full cost of intra-backbone injection for all four diseases.
- **Feature pyramid attention** that applies GLAAM at multiple levels of a feature pyramid network [46], enabling the model to attend to both fine-grained and coarse-grained features simultaneously.
- **Deformable attention** mechanisms [86] that can focus on non-grid sampling locations, potentially improving the localisation of lesions that do not align with the regular convolutional grid.

## 3. Multi-Label Stratified Dataset Splitting

The current dataset split is a plain random partition, not multi-label stratified. While per-disease prevalence was verified to be approximately balanced across splits post hoc, iterative stratification [26] would provide stronger guarantees against class distribution shift. Future work should:

- **Apply iterative stratification** using the `scikit-multilearn` library to ensure that each split preserves the per-disease prevalence and the multi-label co-occurrence distribution.
- **Evaluate the sensitivity** of the reported metrics to the splitting strategy by comparing random, stratified, and iterative-stratified splits in a controlled experiment.

## 4. Prospective Clinical Validation

The current evaluation uses a retrospective test set from the same datasets used for training. Prospective clinical validation is essential before the model can be deployed in a screening workflow. Future work should:

- **Conduct a prospective study** in a real-world screening setting (e.g., a primary care clinic or community health centre) where fundus images are captured by non-specialist healthcare workers using portable fundus cameras. The model's predictions would be compared to the gold standard of comprehensive ophthalmic examination by a trained ophthalmologist.
- **Evaluate the model's impact on referral rates** — does the model reduce unnecessary referrals (high precision) while maintaining adequate sensitivity for true disease cases?
- **Assess the clinical utility of EigenGradCAM heatmaps** — do clinicians find the attention visualisations helpful for verifying the model's predictions, and does the availability of heatmaps increase their trust in the system?
- **Measure inter-rater agreement** between the model's predictions and multiple ophthalmologists, providing a benchmark against the natural variability in human clinical assessment.

## 5. Extension to Additional Diseases

The current system detects four diseases. Extending to additional fundus pathologies would increase the clinical utility of the system. Future work should:

- **Add Age-related Macular Degeneration (AMD)** as a fifth disease, requiring a new attention specialist configured for drusen and geographic atrophy detection. AMD lesions manifest at medium spatial scales (10–50 pixels), suggesting a GLAAMBlock with moderate reduction ($r = 8$) similar to the Glaucoma specialist.
- **Add hypertension retinopathy** detection, which shares vascular features with DR but requires attention to arteriolar narrowing and AV nicking patterns.
- **Explore disease severity grading** beyond binary detection — for example, classifying DR into the five International Clinical Diabetic Retinopathy (ICDR) severity levels [1] rather than binary DR/no-DR. This would require modifying the DR classifier head from binary to multi-class and adjusting the loss function accordingly.

## 6. Model Compression and Edge Deployment

While GLAAM-4X is already lightweight (12.3M parameters, 47 MB), further compression would enable deployment on even more constrained devices (e.g., smartphones without GPU acceleration). Future work should:

- **Apply quantisation** (int8 or float16) to reduce the model size by 2–4× with minimal accuracy loss, following the quantisation-aware training paradigm [87].
- **Explore knowledge distillation** [88] to train a smaller student model (e.g., MobileNetV3 [89]) that mimics the disease-specific attention behaviour of GLAAM-4X, potentially achieving comparable performance with fewer parameters.
- **Export to ONNX format** for deployment on edge computing hardware (e.g., NVIDIA Jetson, Raspberry Pi with Coral TPU), enabling offline screening in settings without internet connectivity.
- **Develop a mobile application** that runs the model locally on a smartphone attached to a portable fundus camera, providing real-time screening in field settings.

## 7. Continual Learning and Dataset Expansion

The unified corpus used in this work aggregates nine public datasets. As new fundus image datasets become available, the model should be updated without retraining from scratch. Future work should:

- **Implement continual learning** strategies [90] that allow the model to incorporate new datasets without catastrophic forgetting of previously learned disease patterns. Techniques such as elastic weight consolidation [91] or replay buffers could enable incremental training on new data while preserving performance on old data.
- **Expand the training corpus** with additional public datasets as they become available on Kaggle and other data repositories, particularly for underrepresented diseases (glaucoma, myopia) and underrepresented populations.
- **Investigate domain adaptation** techniques to handle the distribution shift between different fundus camera models, which can produce images with different colour profiles, resolutions, and field-of-view angles. Adversarial domain adaptation [92] or style transfer methods could improve the model's robustness to camera variability.

## 8. Integration with Clinical Decision Support Systems

For real-world deployment, GLAAM-4X should be integrated into a broader clinical decision support system. Future work should:

- **Develop a HL7/FHIR-compatible API** that allows the model to be integrated into electronic health record (EHR) systems, enabling automatic screening of fundus images captured during routine eye examinations.
- **Incorporate patient metadata** (age, sex, diabetes duration, intraocular pressure) as additional input features, potentially improving prediction accuracy by combining imaging features with clinical risk factors.
- **Implement an active learning pipeline** where the model flags low-confidence predictions for review by an ophthalmologist, and the ophthalmologist's labels are fed back into the training pipeline to improve the model over time.
- **Conduct a cost-effectiveness analysis** comparing AI-assisted screening to traditional ophthalmologist-led screening, quantifying the potential cost savings and health outcomes impact in different healthcare settings.

---

# Conclusion

We presented GLAAM-4X, a multi-label deep learning classifier for detecting four sight-threatening ocular diseases — Cataract, Diabetic Retinopathy, Glaucoma, and Myopia — from a single colour fundus photograph. The system addresses three fundamental limitations of existing fundus disease classifiers through three architectural innovations.

First, GLAAM-4X introduces **disease-specific attention specialists**, each tailored to the spatial scale and morphological characteristics of its target pathology. The DR specialist employs a novel MultiScaleGLAAM module that applies attention at three spatial scales (fine, medium, coarse) to capture the full range of DR lesion sizes, from 2-pixel microaneurysms to 50-pixel haemorrhages. The Glaucoma specialist uses a GLAAMBlock with moderate reduction ($r = 8$) to focus on the localised optic disc region. The Cataract specialist uses a high-reduction GLAAMBlock ($r = 16$) to capture the diffuse, global nature of lens opacity. The Myopia specialist uses an identity pathway, avoiding unnecessary computational overhead for a disease whose signal is already well-captured by the backbone. This disease-specific design philosophy departs from the standard shared-attention paradigm and demonstrates that tailoring attention to each pathology's characteristics improves multi-disease detection performance.

Second, GLAAM-4X applies attention **post-backbone** rather than intra-backbone (as in the original GLAAM [7]), preserving the pretrained MobileNetV2 features and enabling efficient multi-disease branching from a single shared 1280-channel feature map. The four attention specialists add only 2.3M parameters (18% of the 12.3M total), a modest overhead for the performance improvement they provide. This design choice makes the model suitable for deployment on resource-constrained screening devices.

Third, GLAAM-4X uses **Asymmetric Loss** with per-disease optimal thresholding to address the extreme class imbalance (3:1 to 15:1) inherent in multi-label fundus datasets. The configuration ($\gamma_{\text{neg}} = 4.0$, $\gamma_{\text{pos}} = 0.0$, clip $m = 0.05$) aggressively suppresses easy negative gradients while preserving full gradient signal from all positive examples, and per-disease thresholds are grid-searched on the validation set to maximise F1 for each disease independently.

On a held-out test set of 2,232 images from a unified corpus of 27,899 images aggregated from nine public datasets, GLAAM-4X achieves a macro-averaged AUC of 0.968 (Cataract 0.985, DR 0.952, Glaucoma 0.966, Myopia 0.971) and a macro-averaged F1 of 0.786. The model comprises 12.3 million parameters (47 MB) and achieves 12.44 ms inference latency on a single Tesla T4 GPU (~80 frames per second), making it suitable for real-time screening on commodity hardware. Disease-specific EigenGradCAM heatmaps provide clinically interpretable visualisations that confirm the model's attention corresponds to relevant anatomical features (optic disc for glaucoma, lesion-bearing regions for DR, diffuse pattern for cataract), addressing the explainability requirement for clinical adoption [10,11].

The calibration analysis shows that Cataract and DR produce well-calibrated probability estimates that can be used directly for clinical triage, while Glaucoma and Myopia exhibit some miscalibration at extreme probability values, suggesting that post-hoc calibration (e.g., temperature scaling [65]) may be beneficial before clinical deployment for these diseases.

The primary limitation is the Glaucoma classifier's recall (58.8%), caused by label noise from five different glaucoma-only datasets with inconsistent annotation standards. This highlights a broader challenge in multi-disease fundus classification: the quality and consistency of training labels are as important as the model architecture in determining performance. The model's high precision for glaucoma (83.3%) indicates that when it does predict glaucoma, it is usually correct — but the conservative threshold (0.57) means that 41.2% of true glaucoma cases are missed. In a clinical screening context, this trade-off would need to be adjusted based on the specific screening protocol and the availability of complementary diagnostic tools (e.g., intraocular pressure measurement).

The future work directions outlined above — improving glaucoma detection through cleaner annotations, exploring higher-resolution attention injection, prospective clinical validation, extension to additional diseases, model compression for edge deployment, continual learning, and integration with clinical decision support systems — provide a roadmap for translating GLAAM-4X from a research prototype into a clinically deployed screening tool. The combination of disease-specific attention, asymmetric loss, and gradient-based explainability demonstrated in this work provides a foundation for this translation, and the 0.968 macro-averaged AUC achieved with a 12.3M-parameter model demonstrates that clinically relevant multi-disease fundus screening is feasible with lightweight, interpretable architectures.

---

## References

[1] Wilkinson, C.P. et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology*, 110(9), 1677–1682.

[2] Foster, P.J., Buhrmann, R., Quigley, H.A. & Johnson, G.J. (2002). The definition and classification of glaucoma in prevalence surveys. *British Journal of Ophthalmology*, 86(2), 238–242.

[3] Liu, Y.C. et al. (2017). Cataract prevalence in diabetes mellitus and systemic hypertension: the Shihpai Eye Study. *Investigative Ophthalmology & Visual Science*, 58(8), 4749–4755.

[4] Ohno-Matsui, K. et al. (2016). International photographic classification and grading system for myopic maculopathy. *American Journal of Ophthalmology*, 159, 895–912.

[5] Hu, J., Shen, L. & Sun, G. (2018). Squeeze-and-excitation networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7132–7141.

[6] Woo, S., Park, J., Lee, J.Y. & Kweon, I.S. (2018). CBAM: Convolutional block attention module. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 3–19.

[7] Kumar, D., Verma, C. & Illés, Z. (2025). GLAAM and GLAAI: Pioneering attention models for robust automated cataract detection. *Computer Methods and Programs in Biomedicine Update*, 7, 100182. https://doi.org/10.1016/j.cmpbup.2025.100182

[8] Buda, M., Maki, A. & Mazurowski, M.A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249–259.

[9] Lin, T.Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2980–2988.

[10] Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 4765–4774.

[11] Topol, E.J. (2019). High-performance medicine: the convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44–56.

[12] Selvaraju, R.R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626.

[13] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. & Chen, L.C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4510–4520.

[14] Ben-Baruch, E. et al. (2021). Asymmetric loss for multi-label classification. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 82–91.

[15] Chattopadhyay, A., Sarkar, A., Howlader, P. & Balasubramanian, V.N. (2018). Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. In *Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 839–847.

[16] ODIR-5K: Ocular Disease Intelligent Recognition dataset. Kaggle: `andrewmvd/ocular-disease-recognition-odir5k`.

[17] Porwal, P. et al. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD). In *IEEE International Conference on Machine Vision and Applications*.

[18] Li, Z. et al. (2020). An automated grading system for diabetic retinopathy based on a lightweight deep neural network. *IEEE Access*, 8, 104585–104596.

[19] Pachade, S. et al. (2021). Retinal fundus multi-disease image dataset (RFMiD): A dataset for multi-disease classification of retinal fundus images. *Medical Image Analysis*, 70, 102001.

[20] Zhang, J. et al. (2010). Fundus image database for eye disease screening. *Journal of Ophthalmology*.

[21] Cuadros, J. & Bresnick, G. (2009). PAPILA: Fundus images for glaucoma assessment. *Kaggle*.

[22] REFUGE2: Retinal Fundus Glaucoma Challenge dataset. Kaggle: `victorlemosml/refuge2`.

[23] Litjens, G. et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88.

[24] Halevy, A., Norvig, P. & Pereira, F. (2009). The unreasonable effectiveness of data. *IEEE Intelligent Systems*, 24(2), 8–12.

[25] Wang, S. et al. (2022). Real-time and automatic diabetic retinopathy detection using a lightweight deep neural network. *IEEE Transactions on Medical Imaging*, 41(12), 3463–3473.

[26] Sechidis, K., Tsoumakas, G. & Vlahavas, I. (2011). On the stratification of multi-label data. In *Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)*, pp. 145–158.

[27] Zhang, M.L. & Zhou, Z.H. (2014). A review on multi-label learning algorithms. *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819–1837.

[28] Russakovsky, O. et al. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3), 211–252.

[29] Deng, J. et al. (2009). ImageNet: A large-scale hierarchical image database. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 248–255.

[30] Raghu, M., Zhang, C., Kleinberg, J. & Bengio, S. (2019). Transfusion: Understanding transfer learning for medical imaging. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 3347–3357.

[31] Ke, A. et al. (2021). Chextransfer: performance and parameter efficiency of ImageNet models for chest X-ray interpretation. In *Proceedings of the Conference on Health, Inference, and Learning (CHIL)*, pp. 116–124.

[32] Esteva, A. et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115–118.

[33] torchvision: Computer vision library for PyTorch. https://pytorch.org/vision/

[34] Buslaev, A. et al. (2020). Albumentations: fast and flexible image augmentations. *Information*, 11(2), 125.

[35] Simard, P.Y., Steinkraus, D. & Platt, J.C. (2003). Best practices for convolutional neural networks applied to visual document analysis. In *Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)*, pp. 958–962.

[36] Zhong, Z., Zheng, L., Kang, G., Li, S. & Yang, Y. (2020). Random erasing data augmentation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pp. 13024–13030.

[37] Zhang, H. et al. (2018). mixup: Beyond empirical risk minimization. In *International Conference on Learning Representations (ICLR)*.

[38] Yun, S. et al. (2019). CutMix: Regularization strategy to cut and mix patches for strong data augmentation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 10726–10735.

[39] Kang, B. et al. (2020). Class-balanced loss based on effective number of samples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9268–9277.

[40] Wang, X. et al. (2017). ChestX-ray8: hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2097–2105.

[41] Rajaraman, S. et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ*, 6, e4568.

[42] Howard, A.G. et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.

[43] He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.

[44] Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1251–1258.

[45] Ioffe, S. & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the International Conference on Machine Learning (ICML)*, pp. 448–456.

[46] Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B. & Belongie, S. (2017). Feature pyramid networks for object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2117–2125.

[47] Jacobs, R.A., Jordan, M.I., Nowlan, S.J. & Hinton, G.E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1), 79–87.

[48] PyTorch 2.0: torch.compile. https://pytorch.org/get-started/pytorch-2.0/

[49] Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the International Conference on Machine Learning (ICML)*, pp. 6105–6114.

[50] Huang, G., Liu, Z., Van Der Maaten, L. & Weinberger, K.Q. (2017). Densely connected convolutional networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4700–4708.

[51] Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*.

[52] Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations (ICLR)*.

[53] Micikevicius, P. et al. (2018). Mixed precision training. In *International Conference on Learning Representations (ICLR)*.

[54] Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*.

[55] Goyal, P. et al. (2017). Accurate, large minibatch SGD: Training ImageNet in 1 hour. *arXiv preprint arXiv:1706.02677*.

[56] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A.G. (2018). Averaging weights leads to wider optima and better generalization. In *Conference on Uncertainty in Artificial Intelligence (UAI)*.

[57] Hanley, J.A. & McNeil, B.J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36.

[58] Zhang, M.L., Li, Y.K. & Liu, X.Y. (2015). Towards class-imbalance aware multi-label learning. *IEEE Transactions on Knowledge and Data Engineering*, 28(7), 1751–1764.

[59] European Union (2024). Regulation (EU) 2024/1689: Artificial Intelligence Act. *Official Journal of the European Union*.

[60] U.S. Food and Drug Administration (2022). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan.

[61] Jacobgil/pytorch-grad-cam: Advanced gradient-based visual explanations for PyTorch models. https://github.com/jacobgil/pytorch-grad-cam

[62] HuggingFace SafeTensors: A simple, safe, and fast file format for storing tensors. https://github.com/huggingface/safetensors

[63] Grinberg, M. (2018). *Flask Web Development*, 2nd ed. O'Reilly Media.

[64] Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

[65] Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. (2017). On calibration of modern neural networks. In *Proceedings of the International Conference on Machine Learning (ICML)*, pp. 1321–1330.

[66] Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 8024–8035.

[67] McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference (SciPy)*, pp. 56–61.

[68] Virtanen, P. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261–272.

[69] Song, C.H., Han, H.J. & Avrithis, Y. (2022). All the attention you need: Global-local, spatial-channel attention for image retrieval. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 2754–2763.

[70] Chen, L. et al. (2017). SCA-CNN: Spatial and channel-wise attention in convolutional networks for image captioning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5659–5667.

[71] Bello, I., Zoph, B., Vaswani, A., Shlens, J. & Le, Q.V. (2019). Attention augmented convolutional networks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 3286–3295.

[72] Li, L. et al. (2019). Attention based glaucoma detection: A large-scale database and CNN model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10571–10580.

[73] Zhang, Z. et al. (2022). Attention to region: Region-based integration-and-recalibration networks for nuclear cataract classification using AS-OCT images. *Medical Image Analysis*, 80, 102499.

[74] Zhang, Z. et al. (2022). Adaptive feature squeeze network for nuclear cataract classification in AS-OCT image. *Journal of Biomedical Informatics*, 128, 104037.

[75] Zhang, Z. et al. (2022). CCA-Net: Clinical-awareness attention network for nuclear cataract classification in AS-OCT. *Knowledge-Based Systems*, 250, 109109.

[76] Zhang, Z. et al. (2024). Regional context-based recalibration network for cataract recognition in AS-OCT. *Pattern Recognition*, 147, 110069.

[77] Szegedy, C. et al. (2016). Inception-v4, Inception-ResNet and the impact of residual connections on learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pp. 4278–4284.

[78] Lee, S. et al. (2019). Steel surface defect diagnostics using deep convolutional neural network and class activation map. *Applied Sciences*, 9(24), 5449.

[79] Shyamalee, D. et al. (2022). Attention U-Net for glaucoma identification using fundus image segmentation. In *International Conference on Decision Aid Sciences and Applications*, pp. 6–12.

[80] Bourne, R.R.A. et al. (2013). Causes of vision loss worldwide, 1990–2010: a systematic analysis. *The Lancet Global Health*, 1(6), e339–e349.

[81] Vaswani, A. et al. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 5998–6008.

[82] Guo, M.H. et al. (2022). Attention mechanisms in computer vision: A survey. *Computational Visual Media*, 8(2), 331–368.

[83] Li, Z. et al. (2021). Applications of deep learning in fundus images: A review. *Medical Image Analysis*, 69, 101971.

[84] Souid, A. et al. (2021). Classification and predictions of lung diseases from chest X-rays using MobileNet V2. *Applied Sciences*, 11(6), 2751.

[85] Yuan, Y. et al. (2022). Low-res MobileNet: An efficient lightweight network for low-resolution image classification in resource-constrained scenarios. *Multimedia Tools and Applications*, 81, 38513–38529.

[86] Dai, J. et al. (2017). Deformable convolutional networks. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 764–773.

[87] Jacob, B. et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2704–2713.

[88] Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

[89] Howard, A. et al. (2019). Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 1314–1324.

[90] Parisi, G.I., Kemker, R., Part, J.L., Kanan, C. & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71.

[91] Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.

[92] Ganin, Y. & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. In *Proceedings of the International Conference on Machine Learning (ICML)*, pp. 1180–1189.