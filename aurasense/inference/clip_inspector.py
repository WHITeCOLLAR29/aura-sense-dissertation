"""
Clip Inspector — Offline Video Clip Analysis reusing AuraSense's cached pretrained models.

Supports MP4, MOV, and WEBM uploads:
- Samples video frames at configurable intervals (default 1.5s)
- Extracts mono 16 kHz audio and slices 4.0s rolling windows
- Reuses the already-loaded `PretrainedModelRegistry` (zero duplicate models)
- Cleans up temporary files and intermediate arrays immediately in `finally:`
"""

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import av
import cv2
import librosa
import numpy as np

from aurasense.config import GENDER_UNCERTAIN_MESSAGE, LiveInferenceConfig
from .fusion import MultimodalEmotionFusionEngine
from .model_registry import PretrainedModelRegistry


SUPPORTED_CLIP_EXTENSIONS = (".mp4", ".mov", ".webm")


def analyze_uploaded_clip(
    file_bytes: bytes,
    filename: str,
    registry: PretrainedModelRegistry,
    config: Optional[LiveInferenceConfig] = None,
    sample_interval_sec: float = 1.5,
    max_duration_sec: float = 60.0,
    max_file_mb: float = 50.0,
) -> Dict[str, Any]:
    """
    Analyzes an uploaded MP4/MOV/WEBM video clip in sampled segments using the
    already-loaded `PretrainedModelRegistry`.
    """
    t0 = time.perf_counter()
    cfg = config or registry.config

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED_CLIP_EXTENSIONS:
        return {
            "available": False,
            "error": f"Unsupported format '{ext}'. Please upload MP4, MOV, or WEBM.",
        }

    size_mb = len(file_bytes) / (1024.0 * 1024.0)
    if size_mb > max_file_mb:
        return {
            "available": False,
            "error": f"File size ({size_mb:.1f} MB) exceeds the {max_file_mb:.0f} MB limit.",
        }

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        container = av.open(tmp_path)
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {
            "available": False,
            "error": f"Could not decode video file (corrupt or unsupported): {exc}",
        }

    sampled_frames: List[tuple[float, np.ndarray]] = []
    audio_chunks: List[np.ndarray] = []
    audio_sr: int = cfg.target_sample_rate
    has_audio = False
    duration_sec = 0.0

    try:
        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            return {"available": False, "error": "No video stream found in uploaded file."}

        v_stream = video_streams[0]
        if v_stream.duration and v_stream.time_base:
            duration_sec = float(v_stream.duration * v_stream.time_base)

        next_sample_ts = 0.0
        for frame in container.decode(video=0):
            ts = float(frame.pts * frame.time_base) if (frame.pts is not None and frame.time_base) else next_sample_ts
            if ts > max_duration_sec:
                break
            if ts + 1e-6 >= next_sample_ts:
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                if w > 640:
                    img = cv2.resize(img, (640, int(h * (640.0 / w))), interpolation=cv2.INTER_AREA)
                sampled_frames.append((round(ts, 2), img))
                next_sample_ts = ts + max(0.5, float(sample_interval_sec))
            duration_sec = max(duration_sec, ts)

        audio_streams = [s for s in container.streams if s.type == "audio"]
        if audio_streams:
            has_audio = True
            src_sr = int(audio_streams[0].rate or 48000)
            for aframe in container.decode(audio=0):
                ats = float(aframe.pts * aframe.time_base) if (aframe.pts is not None and aframe.time_base) else 0.0
                if ats > max_duration_sec:
                    break
                arr = aframe.to_ndarray()
                if np.issubdtype(arr.dtype, np.integer):
                    wav = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
                else:
                    wav = arr.astype(np.float32)
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=0) if wav.shape[0] <= 8 else np.mean(wav, axis=-1)
                wav = wav.reshape(-1)
                if wav.size > 0:
                    if src_sr != cfg.target_sample_rate:
                        wav = librosa.resample(wav, orig_sr=src_sr, target_sr=cfg.target_sample_rate).astype(np.float32)
                    audio_chunks.append(wav)
    except Exception as exc:
        return {"available": False, "error": f"Error reading media streams: {exc}"}
    finally:
        container.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not sampled_frames:
        return {"available": False, "error": "No video frames could be decoded from the clip."}

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)
    del audio_chunks

    engine = MultimodalEmotionFusionEngine(registry=registry, config=cfg)
    win_samples = int(cfg.rolling_window_sec * audio_sr)

    timeline: List[Dict[str, Any]] = []
    face_emotions: List[str] = []
    voice_emotions: List[str] = []
    combined_emotions: List[str] = []
    combined_confs: List[float] = []
    presentation_votes: List[str] = []
    any_face_detected = False
    multiple_faces_seen = False

    for ts, frame_bgr in sampled_frames:
        end_idx = int(ts * audio_sr)
        start_idx = max(0, end_idx - win_samples)
        if full_audio.size >= int(cfg.min_speech_window_sec * audio_sr):
            audio_win = full_audio[start_idx:end_idx] if end_idx > start_idx else full_audio[:win_samples]
        else:
            audio_win = np.zeros(0, dtype=np.float32)

        res = engine.analyze_live_window(
            frames_bgr=[frame_bgr],
            audio_waveform=audio_win,
            audio_sr=audio_sr,
        )
        fa = res.get("face_analysis") or {}
        va = res.get("voice_analysis") or {}
        dg = res.get("dual_facial_gender") or {}

        if fa.get("face_detected"):
            any_face_detected = True
        if (fa.get("diagnostics") or {}).get("multiple_faces"):
            multiple_faces_seen = True

        if fa.get("available") and fa.get("predicted_emotion"):
            face_emotions.append(str(fa["predicted_emotion"]))
        if va.get("available") and va.get("predicted_emotion"):
            voice_emotions.append(str(va["predicted_emotion"]))
        if res.get("available") and res.get("dominant_emotion"):
            combined_emotions.append(str(res["dominant_emotion"]))
            if res.get("emotion_confidence") is not None:
                combined_confs.append(float(res["emotion_confidence"]))
        if not dg.get("is_uncertain", True) and dg.get("short_label"):
            presentation_votes.append(str(dg["short_label"]))

        timeline.append(
            {
                "timestamp_sec": ts,
                "facial_presentation": dg.get("final_label") or GENDER_UNCERTAIN_MESSAGE,
                "facial_emotion": fa.get("predicted_emotion") if fa.get("available") else "No Face",
                "vocal_emotion": va.get("predicted_emotion") if va.get("available") else (va.get("display_state") or "No Speech"),
                "combined_emotion": res.get("dominant_emotion") if res.get("available") else "Insufficient data",
                "confidence_pct": res.get("emotion_confidence"),
                "fusion_mode": res.get("fusion_mode", "insufficient_data"),
            }
        )

    del sampled_frames, full_audio

    def _majority(items: List[str], fallback: str) -> str:
        if not items:
            return fallback
        return max(set(items), key=items.count)

    overall_presentation = (
        f"{_majority(presentation_votes, '')} (Perceived Presentation)"
        if presentation_votes
        else (GENDER_UNCERTAIN_MESSAGE if any_face_detected else "No face detected")
    )

    return {
        "available": True,
        "filename": filename,
        "duration_sec": round(duration_sec, 2),
        "segments_analyzed": len(timeline),
        "has_audio_track": has_audio,
        "any_face_detected": any_face_detected,
        "multiple_faces_seen": multiple_faces_seen,
        "overall_facial_presentation": overall_presentation,
        "overall_facial_emotion": _majority(face_emotions, "No face detected"),
        "overall_vocal_emotion": _majority(voice_emotions, "No speech detected"),
        "overall_combined_emotion": _majority(combined_emotions, "Insufficient data"),
        "avg_confidence_pct": round(float(np.mean(combined_confs)), 1) if combined_confs else None,
        "timeline": timeline,
        "processing_time_sec": round(time.perf_counter() - t0, 2),
    }
