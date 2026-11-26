"""
Cached Pretrained Model Registry for AuraSense.

Loads and caches all pretrained inference models strictly once via `@st.cache_resource`:
1. Facial Gender Model 1 + 5-Point Detector: **InsightFace `buffalo_l`** (`det_10g` SCRFD + `genderage.onnx`)
2. Facial Gender Model 2: **FairFace `ResNet34`** (`res34_fair_align_multi_7_20190809.pt`)
3. Facial Emotion Model: **DeepFace** pretrained facial expression CNN (run on the aligned face crop)
4. Voice Emotion Model: **`Dpngtm/wav2vec2-emotion-recognition`** (`AutoProcessor` + `AutoModelForAudioClassification`)
5. Vocal Gender Presentation Model (kept strictly separate): **`prithivMLmods/Common-Voice-Gender-Detection`**
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
)

from aurasense.config import LiveInferenceConfig

logger = logging.getLogger("AuraSense.ModelRegistry")


def select_compute_device(preference: Optional[str] = None) -> torch.device:
    """Selects optimal PyTorch compute device supporting CUDA and CPU (plus MPS when available)."""
    requested = (preference or os.getenv("AURASENSE_DEVICE", "auto")).strip().lower()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested via AURASENSE_DEVICE, but torch.cuda.is_available() is False; using CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_onnx_providers(device: torch.device) -> List[str]:
    """Selects ONNX Runtime execution providers compatible with the active compute device."""
    avail = ort.get_available_providers()
    if device.type == "cuda" and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class PretrainedModelRegistry:
    """Thread-safe singleton holder for InsightFace Buffalo_L, FairFace ResNet34, DeepFace, and Wav2Vec2."""

    def __init__(
        self,
        config: Optional[LiveInferenceConfig] = None,
        device_preference: Optional[str] = None,
    ) -> None:
        self.config = config or LiveInferenceConfig()
        self.device: torch.device = select_compute_device(device_preference)
        self.onnx_providers: List[str] = select_onnx_providers(self.device)

        self._buffalo_lock = threading.Lock()
        self._fairface_lock = threading.Lock()
        self._deepface_lock = threading.Lock()
        self._torch_lock = threading.Lock()
        self._init_lock = threading.Lock()

        # 1. InsightFace Buffalo_L (SCRFD 5-point detector + genderage attribute head)
        self._buffalo_app: Optional[Any] = None
        self._buffalo_ready: bool = False
        self._buffalo_error: Optional[str] = None
        self._buffalo_load_ms: float = 0.0
        self._buffalo_loads: int = 0

        # 2. FairFace ResNet34 (independent facial gender model)
        self._fairface_model: Optional[nn.Module] = None
        self._fairface_transform: Optional[Any] = None
        self._fairface_ready: bool = False
        self._fairface_error: Optional[str] = None
        self._fairface_load_ms: float = 0.0
        self._fairface_loads: int = 0

        # 3. DeepFace Emotion CNN (runs on cropped face with detector_backend='skip')
        self._deepface_ready: bool = False
        self._deepface_error: Optional[str] = None
        self._deepface_load_ms: float = 0.0
        self._deepface_loads: int = 0

        # 4. Voice Emotion Wav2Vec2
        self._voice_emo_processor: Optional[Any] = None
        self._voice_emo_model: Optional[Any] = None
        self._voice_emo_error: Optional[str] = None
        self._voice_emo_load_ms: float = 0.0
        self._voice_emo_loads: int = 0

        # 5. Vocal Presentation Wav2Vec2 (kept strictly separate from facial gender)
        self._voice_pres_processor: Optional[Any] = None
        self._voice_pres_model: Optional[Any] = None
        self._voice_pres_error: Optional[str] = None
        self._voice_pres_load_ms: float = 0.0
        self._voice_pres_loads: int = 0

    def ensure_buffalo_loaded(self) -> Optional[Any]:
        """Loads InsightFace `buffalo_l` (`detection` + `genderage` modules) once."""
        if self._buffalo_ready and self._buffalo_app is not None:
            return self._buffalo_app
        with self._init_lock:
            if self._buffalo_ready and self._buffalo_app is not None:
                return self._buffalo_app
            t0 = time.perf_counter()
            try:
                from insightface.app import FaceAnalysis

                ctx_id = 0 if "CUDAExecutionProvider" in self.onnx_providers else -1
                app = FaceAnalysis(
                    name=self.config.insightface_pack_name,
                    allowed_modules=["detection", "genderage"],
                    providers=self.onnx_providers,
                )
                app.prepare(ctx_id=ctx_id, det_size=self.config.buffalo_det_size)
                self._buffalo_app = app
                self._buffalo_ready = True
                self._buffalo_error = None
                self._buffalo_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._buffalo_loads += 1
                logger.info("Loaded InsightFace '%s' in %.1f ms.", self.config.insightface_pack_name, self._buffalo_load_ms)
                return app
            except Exception as exc:
                self._buffalo_ready = False
                self._buffalo_error = f"InsightFace Buffalo_L initialization failed: {exc}"
                logger.error(self._buffalo_error)
                return None

    def ensure_fairface_loaded(self) -> Tuple[Optional[nn.Module], Optional[Any]]:
        """Loads pretrained FairFace ResNet34 (`res34_fair_align_multi_7_20190809.pt`) once."""
        if self._fairface_ready and self._fairface_model is not None:
            return self._fairface_model, self._fairface_transform
        with self._init_lock:
            if self._fairface_ready and self._fairface_model is not None:
                return self._fairface_model, self._fairface_transform
            t0 = time.perf_counter()
            try:
                ckpt_path = hf_hub_download(
                    repo_id=self.config.fairface_repo_id,
                    filename=self.config.fairface_filename,
                )
                model = torchvision.models.resnet34(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 18)
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()

                transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                self._fairface_model = model
                self._fairface_transform = transform
                self._fairface_ready = True
                self._fairface_error = None
                self._fairface_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._fairface_loads += 1
                logger.info("Loaded FairFace ResNet34 in %.1f ms on %s.", self._fairface_load_ms, self.device)
                return model, transform
            except Exception as exc:
                self._fairface_ready = False
                self._fairface_error = f"FairFace ResNet34 initialization failed: {exc}"
                logger.error(self._fairface_error)
                return None, None

    def ensure_deepface_loaded(self) -> bool:
        """Loads DeepFace pretrained facial expression CNN once (uses `detector_backend='skip'` on aligned crops)."""
        if self._deepface_ready:
            return True
        with self._init_lock:
            if self._deepface_ready:
                return True
            t0 = time.perf_counter()
            try:
                from deepface import DeepFace

                dummy = np.full((112, 112, 3), 128, dtype=np.uint8)
                with self._deepface_lock:
                    DeepFace.analyze(
                        img_path=dummy,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="skip",
                        silent=True,
                    )
                self._deepface_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._deepface_loads += 1
                self._deepface_ready = True
                self._deepface_error = None
                return True
            except Exception as exc:
                self._deepface_ready = False
                self._deepface_error = f"DeepFace emotion model initialization failed: {exc}"
                logger.error(self._deepface_error)
                return False

    def ensure_voice_emotion_loaded(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Loads `Dpngtm/wav2vec2-emotion-recognition` via AutoProcessor + AutoModelForAudioClassification."""
        if self._voice_emo_model is not None and self._voice_emo_processor is not None:
            return self._voice_emo_processor, self._voice_emo_model
        with self._init_lock:
            if self._voice_emo_model is not None and self._voice_emo_processor is not None:
                return self._voice_emo_processor, self._voice_emo_model
            model_id = self.config.voice_emotion_model_id
            t0 = time.perf_counter()
            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForAudioClassification.from_pretrained(model_id)
                model.to(self.device)
                model.eval()
                self._voice_emo_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._voice_emo_loads += 1
                self._voice_emo_processor = processor
                self._voice_emo_model = model
                self._voice_emo_error = None
                return processor, model
            except OSError as os_exc:
                self._voice_emo_error = (
                    f"Missing model files or failed Hugging Face download for '{model_id}': {os_exc}"
                )
                logger.error(self._voice_emo_error)
                return None, None
            except Exception as exc:
                self._voice_emo_error = f"Failed to initialize voice emotion model '{model_id}': {exc}"
                logger.error(self._voice_emo_error)
                return None, None

    def ensure_voice_presentation_loaded(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Loads verified pretrained vocal presentation model (`prithivMLmods/Common-Voice-Gender-Detection`)."""
        if not self.config.enable_voice_presentation or not self.config.voice_presentation_model_id:
            return None, None
        if self._voice_pres_model is not None and self._voice_pres_processor is not None:
            return self._voice_pres_processor, self._voice_pres_model
        with self._init_lock:
            if self._voice_pres_model is not None and self._voice_pres_processor is not None:
                return self._voice_pres_processor, self._voice_pres_model
            model_id = self.config.voice_presentation_model_id
            t0 = time.perf_counter()
            try:
                try:
                    processor = AutoProcessor.from_pretrained(model_id)
                except Exception:
                    processor = AutoFeatureExtractor.from_pretrained(model_id)
                model = AutoModelForAudioClassification.from_pretrained(model_id)
                model.to(self.device)
                model.eval()
                self._voice_pres_load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self._voice_pres_loads += 1
                self._voice_pres_processor = processor
                self._voice_pres_model = model
                self._voice_pres_error = None
                return processor, model
            except Exception as exc:
                self._voice_pres_error = f"Voice presentation model '{model_id}' not available: {exc}"
                logger.warning(self._voice_pres_error)
                return None, None

    def warmup_all(self) -> Dict[str, Any]:
        """Preloads all facial and vocal models once at startup."""
        self.ensure_buffalo_loaded()
        self.ensure_fairface_loaded()
        self.ensure_deepface_loaded()
        self.ensure_voice_emotion_loaded()
        self.ensure_voice_presentation_loaded()
        return self.status()

    @property
    def buffalo_lock(self) -> threading.Lock:
        return self._buffalo_lock

    @property
    def fairface_lock(self) -> threading.Lock:
        return self._fairface_lock

    @property
    def deepface_lock(self) -> threading.Lock:
        return self._deepface_lock

    @property
    def torch_lock(self) -> threading.Lock:
        return self._torch_lock

    def critical_errors(self) -> List[str]:
        errs: List[str] = []
        if self._buffalo_error:
            errs.append(self._buffalo_error)
        if self._fairface_error:
            errs.append(self._fairface_error)
        if self._deepface_error:
            errs.append(self._deepface_error)
        if self._voice_emo_error:
            errs.append(self._voice_emo_error)
        return errs

    def status(self) -> Dict[str, Any]:
        """Returns structured health, device, load count, and timing metadata for UI & Diagnostics."""
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "device": str(self.device).upper(),
            "onnx_providers": list(self.onnx_providers),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": cuda_name,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "buffalo_ready": self._buffalo_ready,
            "buffalo_model": f"InsightFace {self.config.insightface_pack_name} (SCRFD + genderage.onnx)",
            "buffalo_load_ms": self._buffalo_load_ms,
            "buffalo_loads": self._buffalo_loads,
            "buffalo_error": self._buffalo_error,
            "fairface_ready": self._fairface_ready,
            "fairface_model": f"FairFace ResNet34 ({self.config.fairface_filename})",
            "fairface_load_ms": self._fairface_load_ms,
            "fairface_loads": self._fairface_loads,
            "fairface_error": self._fairface_error,
            "deepface_ready": self._deepface_ready,
            "face_detector": "SCRFD 10G (InsightFace 5-pt Landmarks)",
            "deepface_load_ms": self._deepface_load_ms,
            "deepface_loads": self._deepface_loads,
            "deepface_error": self._deepface_error,
            "voice_emotion_ready": self._voice_emo_model is not None,
            "voice_emotion_model": self.config.voice_emotion_model_id,
            "voice_emotion_load_ms": self._voice_emo_load_ms,
            "voice_emotion_loads": self._voice_emo_loads,
            "voice_emotion_error": self._voice_emo_error,
            "voice_presentation_ready": self._voice_pres_model is not None,
            "voice_presentation_model": (
                self.config.voice_presentation_model_id
                if self.config.enable_voice_presentation
                else "Not available"
            ),
            "voice_presentation_load_ms": self._voice_pres_load_ms,
            "voice_presentation_loads": self._voice_pres_loads,
            "voice_presentation_error": self._voice_pres_error,
            "critical_errors": self.critical_errors(),
            "all_core_ready": bool(
                self._buffalo_ready
                and self._fairface_ready
                and self._deepface_ready
                and self._voice_emo_model is not None
            ),
        }


_GLOBAL_REGISTRY: Optional[PretrainedModelRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_model_registry(
    config: Optional[LiveInferenceConfig] = None,
    device_preference: Optional[str] = None,
) -> PretrainedModelRegistry:
    """Returns process-wide singleton PretrainedModelRegistry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = PretrainedModelRegistry(
                    config=config,
                    device_preference=device_preference,
                )
    return _GLOBAL_REGISTRY


@st.cache_resource(show_spinner="Loading InsightFace Buffalo_L, FairFace ResNet34, DeepFace & Wav2Vec2...")
def get_cached_model_registry() -> PretrainedModelRegistry:
    """Streamlit resource-cached entrypoint ensuring all 5 models load strictly once across reruns."""
    registry = get_model_registry(LiveInferenceConfig())
    registry.warmup_all()
    return registry
