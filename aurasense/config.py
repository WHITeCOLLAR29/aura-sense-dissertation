"""
Configuration constants, Material Design 3 emotion-adaptive pastel palettes,
dual-model facial gender presentation thresholds, and strict face quality gates.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Exact uncertainty message required when models disagree or confidence/quality is insufficient
GENDER_UNCERTAIN_MESSAGE: str = "Uncertain – Please adjust camera position"

# Internal unified 7-class emotion keys shared between facial and vocal distributions
UNIFIED_EMOTIONS: List[str] = [
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Surprised",
    "Disgust",
    "Neutral",
]

# Exact 7 horizontal score bar labels & order for the dashboard:
# Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral
DISPLAY_EMOTION_ORDER: List[Tuple[str, str]] = [
    ("Happy", "Happy"),
    ("Sad", "Sad"),
    ("Angry", "Angry"),
    ("Fearful", "Fear"),
    ("Surprised", "Surprise"),
    ("Disgust", "Disgust"),
    ("Neutral", "Neutral"),
]

DEEPFACE_EMOTION_LABELS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

WAV2VEC2_EMOTION_LABELS: List[str] = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "sad",
    "surprised",
]

# Material Design 3 Pastel Palettes from AuraSense
PASTEL_PALETTES: Dict[str, Dict[str, str]] = {
    "Neutral": {
        "bg": "#F6F4FA",
        "card": "#ECE8F6",
        "accent": "#7C6FA6",
        "pill": "#E0D9F2",
        "emoji": "😌",
    },
    "Calm": {
        "bg": "#F3F8F5",
        "card": "#DFF0E6",
        "accent": "#4A8B6F",
        "pill": "#CCE6D8",
        "emoji": "🌿",
    },
    "Happy": {
        "bg": "#FFF9F2",
        "card": "#FFE8D2",
        "accent": "#D47A46",
        "pill": "#FFD8BE",
        "emoji": "☀️",
    },
    "Sad": {
        "bg": "#F2F7FC",
        "card": "#DCEAF8",
        "accent": "#4D7FB3",
        "pill": "#C8DFF5",
        "emoji": "🌧️",
    },
    "Angry": {
        "bg": "#FDF4F5",
        "card": "#FADCE0",
        "accent": "#C45664",
        "pill": "#F5C6CC",
        "emoji": "🔥",
    },
    "Fearful": {
        "bg": "#F8F4FC",
        "card": "#ECDCF8",
        "accent": "#8759AD",
        "pill": "#DFC7F2",
        "emoji": "🫧",
    },
    "Disgust": {
        "bg": "#F6F8F2",
        "card": "#E4ECD8",
        "accent": "#6E854B",
        "pill": "#D5E2C3",
        "emoji": "🍃",
    },
    "Surprised": {
        "bg": "#FFFBF0",
        "card": "#FFF0C2",
        "accent": "#C2922E",
        "pill": "#FFE69E",
        "emoji": "✨",
    },
}


@dataclass(frozen=True)
class LiveInferenceConfig:
    """Runtime parameters for dual-model facial gender presentation & multimodal emotion analysis."""

    # Audio capture & rolling 4.0-second speech buffer (within 3–5s window)
    target_sample_rate: int = 16000
    rolling_window_sec: float = 4.0
    min_speech_window_sec: float = 2.5
    min_voiced_speech_sec: float = 0.65
    inference_interval_sec: float = 1.8
    silence_rms_threshold: float = 0.0065
    low_volume_rms_threshold: float = 0.016
    min_voiced_ratio: float = 0.16

    # Video capture & periodic dual-model facial analysis
    max_video_buffer_frames: int = 10
    frame_sample_stride: int = 6
    max_sampled_frames: int = 2
    buffalo_det_size: Tuple[int, int] = (640, 640)

    # Strict Facial Gender Quality Gate Thresholds (avoids predicting on degraded faces)
    min_det_score_for_gender: float = 0.65
    min_face_size_px: int = 72
    min_face_area_ratio: float = 0.020
    min_sharpness_laplacian_var: float = 28.0
    min_lighting_luma: float = 45.0
    max_lighting_luma: float = 235.0
    min_edge_margin_px: int = 6
    max_yaw_asymmetry_ratio: float = 0.42
    max_roll_degrees: float = 28.0
    min_pitch_nose_mouth_ratio: float = 0.18
    max_pitch_nose_mouth_ratio: float = 0.84
    max_face_displacement_ratio: float = 0.22

    # Dual-Model Facial Gender Presentation Confidence Thresholds (0–100% per-model softmax)
    # Never blindly averages Model 1 and Model 2 — requires BOTH models to agree AND clear thresholds.
    min_buffalo_gender_confidence_pct: float = 65.0
    min_fairface_gender_confidence_pct: float = 65.0
    vocal_presentation_uncertainty_threshold_pct: float = 65.0

    # Multimodal Emotion Fusion defaults (configurable at runtime)
    facial_fusion_weight: float = 0.50
    vocal_fusion_weight: float = 0.50
    adapt_weights_by_reliability: bool = True
    min_facial_score_threshold: float = 24.0
    min_vocal_score_threshold: float = 24.0
    min_combined_score_threshold: float = 24.0
    smoothing_mode: str = "ema"
    ema_alpha: float = 0.45
    rolling_average_window: int = 4
    switch_hysteresis_margin: float = 6.0
    min_consecutive_switch_windows: int = 2

    # Pretrained Model Identifiers
    insightface_pack_name: str = "buffalo_l"
    fairface_repo_id: str = "wmpscc/StyleGene"
    fairface_filename: str = "res34_fair_align_multi_7_20190809.pt"
    voice_emotion_model_id: str = "Dpngtm/wav2vec2-emotion-recognition"
    voice_presentation_model_id: str = "prithivMLmods/Common-Voice-Gender-Detection"
    enable_voice_presentation: bool = True
