"""
Models for cataract detection.
"""

# Commented out to avoid cv2 dependency for GLAAM-ODIR integration
# from .hybrid_model import HybridCataractModel, MultiTaskLoss
from .attention import GLAAM, GLAAMBlock, GLAAI, GLAAIBlock
from .backbones import get_backbone, MobileNetV2WithAttention, InceptionV3WithAttention
# from .yolo import LensDetector, LensDetectorDataset
from .glaam_bayesian import GLAAM_Bayesian, DualViewGLAAM_Bayesian, MCDropout

__all__ = [
    # Main models
    'HybridCataractModel',
    'MultiTaskLoss',
    
    # Bayesian models
    'GLAAM_Bayesian',
    'DualViewGLAAM_Bayesian',
    'MCDropout',
    
    # Attention modules
    'GLAAM',
    'GLAAMBlock',
    'GLAAI',
    'GLAAIBlock',
    
    # Backbones
    'get_backbone',
    'MobileNetV2WithAttention',
    'InceptionV3WithAttention',
    
    # YOLO
    'LensDetector',
    'LensDetectorDataset'
]

