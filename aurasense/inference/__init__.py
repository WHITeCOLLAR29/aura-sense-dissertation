"""
Pretrained inference package for AuraSense.
Provides facial analysis (DeepFace + RetinaFace), vocal analysis (Wav2Vec2 AutoProcessor),
and configurable multimodal emotion fusion with temporal stabilization.
"""

from .face_analyzer import PretrainedFaceAnalyzer
from .fusion import (
    FusionRuntimeSettings,
    MultimodalEmotionFusionEngine,
    normalize_distribution,
    to_percentage_dict,
)
from .model_registry import (
    PretrainedModelRegistry,
    get_cached_model_registry,
    get_model_registry,
    select_compute_device,
)
from .voice_analyzer import PretrainedVoiceAnalyzer

__all__ = [
    "PretrainedModelRegistry",
    "get_model_registry",
    "get_cached_model_registry",
    "select_compute_device",
    "PretrainedFaceAnalyzer",
    "PretrainedVoiceAnalyzer",
    "FusionRuntimeSettings",
    "MultimodalEmotionFusionEngine",
    "normalize_distribution",
    "to_percentage_dict",
]
