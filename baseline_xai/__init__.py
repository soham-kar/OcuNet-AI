# Baseline XAI Package
"""
MobileNetV2 Baseline for Cataract Detection with XAI Tools.

This package provides:
- MobileNetBaseline: Simple classifier without attention
- XAI Tools: Grad-CAM, LIME, SHAP for explainability
- Training: Simplified training script

Usage:
    from baseline_xai.models import MobileNetBaseline
    from baseline_xai.evaluation import GradCAM, lime_explain
"""

from .models import MobileNetBaseline, SimpleLoss
from .evaluation import GradCAM, lime_explain, shap_explain, full_xai_analysis

__all__ = [
    'MobileNetBaseline',
    'SimpleLoss',
    'GradCAM',
    'lime_explain',
    'shap_explain',
    'full_xai_analysis'
]
