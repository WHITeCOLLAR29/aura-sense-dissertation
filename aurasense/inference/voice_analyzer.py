"""
Pretrained Voice Emotion & Voice Presentation Analyzer using Hugging Face Transformers.

Implementation guarantees:
- Voice Emotion Model: `Dpngtm/wav2vec2-emotion-recognition` loaded via `AutoProcessor` and
  `AutoModelForAudioClassification`, executed strictly inside `torch.inference_mode()`.
- Converts incoming microphone audio to mono and resamples to 16,000 Hz.
- Evaluates rolling 3–5 second windows (default 4.0s) — never individual WebRTC audio frames.
- Handles silence and steady background noise via Butterworth band-limiting + frame-level
  voiced speech activity detection (`librosa.effects.split` + RMS + spectral flatness).
- When speech is insufficient or silent, explicitly returns status `"waiting_for_speech"`
  with display label `"Waiting for speech"` and never fabricates predictions.
- Returns all seven emotion probabilities (`angry`, `calm`, `disgust`, `fearful`, `happy`,
  `sad`, `surprised` as well as unified 7-class taxonomy probabilities), dominant emotion, and confidence.
- Voice Presentation: Uses verified `prithivMLmods/Common-Voice-Gender-Detection` when available;
  if unavailable or disabled, explicitly returns `"Not available"`.
"""

import time
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
from scipy import signal
import torch

from aurasense.config import LiveInferenceConfig, UNIFIED_EMOTIONS, WAV2VEC2_EMOTION_LABELS
from .model_registry import PretrainedModelRegistry, get_model_registry

WAV2VEC2_EMO_MAP: Dict[str, str] = {
    "neutral": "Neutral",
    "calm": "Neutral",  # Model card: neutral & calm merged under index 1 ('calm')
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "fearful": "Fearful",
    "disgust": "Disgust",
    "surprise": "Surprised",
    "surprised": "Surprised",
}


class PretrainedVoiceAnalyzer:
    """Runs real-time Wav2Vec2 speech emotion inference on 3–5s rolling 16 kHz mono speech windows."""

    def __init__(
        self,
        registry: Optional[PretrainedModelRegistry] = None,
        config: Optional[LiveInferenceConfig] = None,
    ) -> None:
        self.config = config or LiveInferenceConfig()
        self.registry = registry or get_model_registry(self.config)

    def preprocess_waveform(
        self,
        waveform: np.ndarray,
        input_sr: int,
    ) -> np.ndarray:
        """
        1. Converts multi-channel microphone audio to mono float32.
        2. Resamples to 16,000 Hz using Librosa.
        3. Applies 75–7600 Hz speech band-pass Butterworth filter (SciPy) to suppress hum and hiss.
        """
        if waveform is None or waveform.size == 0:
            return np.zeros(0, dtype=np.float32)

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0 if wav.shape[0] <= wav.shape[-1] else -1)
        wav = wav.reshape(-1)

        if not np.all(np.isfinite(wav)):
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        # Remove DC offset
        wav = wav - float(np.mean(wav))

        # Resample to 16,000 Hz
        target_sr = self.config.target_sample_rate
        if input_sr != target_sr and wav.size > 32:
            wav = librosa.resample(wav, orig_sr=input_sr, target_sr=target_sr).astype(np.float32)

        # Speech band-pass filter (75 Hz - 7600 Hz at 16 kHz) to attenuate room rumble & high-freq noise
        if wav.size > 512:
            nyq = 0.5 * target_sr
            sos = signal.butter(
                4,
                [75.0 / nyq, min(7600.0, nyq * 0.96) / nyq],
                btype="bandpass",
                output="sos",
            )
            wav = signal.sosfilt(sos, wav).astype(np.float32)

        return wav

    def detect_speech_activity(self, wav_16k: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Distinguishes active human speech from silence or stationary background fan/room noise
        over a 3–5 second rolling buffer.
        """
        sr = self.config.target_sample_rate
        duration_sec = round(wav_16k.size / float(sr), 2)
        if wav_16k.size < int(self.config.min_speech_window_sec * sr):
            return False, {
                "duration_sec": duration_sec,
                "voiced_sec": 0.0,
                "voiced_ratio": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "spectral_flatness": 1.0,
                "is_silent": True,
                "is_background_noise": False,
                "is_low_volume": False,
                "quality_score": 0.0,
            }

        rms = float(np.sqrt(np.mean(np.square(wav_16k))))
        peak = float(np.max(np.abs(wav_16k)))

        # Frame-wise energy envelope & spectral flatness
        frame_rms = librosa.feature.rms(y=wav_16k, frame_length=512, hop_length=160)[0]
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav_16k + 1e-7, n_fft=512, hop_length=160)))

        noise_floor = float(np.percentile(frame_rms, 20)) if frame_rms.size > 0 else 0.0
        speech_gate = max(self.config.silence_rms_threshold, noise_floor * 2.1)
        voiced_frames = frame_rms >= speech_gate
        voiced_ratio = float(np.mean(voiced_frames)) if voiced_frames.size > 0 else 0.0

        # Non-silent voiced intervals using librosa.effects.split
        intervals = librosa.effects.split(wav_16k, top_db=26, frame_length=512, hop_length=160)
        voiced_samples = int(sum(end - start for start, end in intervals))
        voiced_sec = round(voiced_samples / float(sr), 2)

        # White/fan noise exhibits high spectral flatness (> 0.42) and flat envelope modulation
        dynamic_crest = float(np.percentile(frame_rms, 92) / max(noise_floor, 1e-5)) if frame_rms.size > 0 else 1.0
        is_noise_only = bool(flatness > 0.42 and dynamic_crest < 1.8)
        is_silent = bool(rms < self.config.silence_rms_threshold)

        has_sufficient_speech = bool(
            (not is_silent)
            and (not is_noise_only)
            and voiced_sec >= self.config.min_voiced_speech_sec
            and voiced_ratio >= self.config.min_voiced_ratio
        )

        is_low_volume = bool(rms < self.config.low_volume_rms_threshold)
        snr_factor = float(np.clip((rms - self.config.silence_rms_threshold) / 0.045, 0.30, 1.0))
        quality_score = (
            round(snr_factor * (0.75 if is_low_volume else 1.0), 3)
            if has_sufficient_speech
            else 0.0
        )

        return has_sufficient_speech, {
            "duration_sec": duration_sec,
            "voiced_sec": voiced_sec,
            "voiced_ratio": round(voiced_ratio, 3),
            "rms": round(rms, 5),
            "peak": round(peak, 4),
            "spectral_flatness": round(flatness, 4),
            "is_silent": is_silent,
            "is_background_noise": is_noise_only,
            "is_low_volume": is_low_volume,
            "quality_score": quality_score,
        }

    def analyze_waveform(self, waveform: np.ndarray, input_sr: int = 16000) -> Dict[str, Any]:
        """
        Runs Wav2Vec2 emotion inference on ~3–5 seconds of 16 kHz mono speech.
        Returns `"Waiting for speech"` status if buffered speech is insufficient or silent,
        and surfaces explicit errors if model loading or inference fails.
        """
        t0 = time.perf_counter()
        wav_16k = self.preprocess_waveform(waveform, input_sr=input_sr)

        # Keep most recent `rolling_window_sec` (4.0s, within 3–5s range)
        max_samples = int(self.config.rolling_window_sec * self.config.target_sample_rate)
        if wav_16k.size > max_samples:
            wav_16k = wav_16k[-max_samples:]

        emo_processor, emo_model = self.registry.ensure_voice_emotion_loaded()
        if emo_processor is None or emo_model is None:
            err = (
                self.registry.status().get("voice_emotion_error")
                or f"Model '{self.config.voice_emotion_model_id}' could not be loaded."
            )
            return self._non_active_result(
                status="error",
                display_state="Model error",
                reason=err,
                diagnostics={
                    "duration_sec": round(wav_16k.size / float(self.config.target_sample_rate), 2),
                    "voiced_sec": 0.0,
                    "rms": 0.0,
                    "peak": 0.0,
                    "is_silent": True,
                    "is_background_noise": False,
                    "is_low_volume": False,
                    "quality_score": 0.0,
                },
                t0=t0,
                error=err,
            )

        has_speech, diag = self.detect_speech_activity(wav_16k)
        if not has_speech:
            if diag["duration_sec"] < self.config.min_speech_window_sec:
                detail = (
                    f"Waiting for speech ({diag['duration_sec']:.1f}s / "
                    f"{self.config.min_speech_window_sec:.1f}s buffered)"
                )
            elif diag["is_background_noise"]:
                detail = "Waiting for speech (background noise filtered)"
            else:
                detail = "Waiting for speech"
            return self._non_active_result(
                status="waiting_for_speech",
                display_state="Waiting for speech",
                reason=detail,
                diagnostics=diag,
                t0=t0,
            )

        # Normalize active speech waveform cleanly before AutoProcessor
        peak = max(float(np.max(np.abs(wav_16k))), 1e-4)
        norm_wav = (wav_16k / peak * 0.9).astype(np.float32)

        try:
            inputs = emo_processor(
                norm_wav,
                sampling_rate=self.config.target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                k: v.to(self.registry.device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            with self.registry.torch_lock, torch.inference_mode():
                logits = emo_model(**model_inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        except Exception as exc:
            err_msg = f"Voice emotion inference failed: {exc}"
            return self._non_active_result(
                status="error",
                display_state="Inference error",
                reason=err_msg,
                diagnostics=diag,
                t0=t0,
                error=err_msg,
            )

        id2label = getattr(emo_model.config, "id2label", {})
        emo_probs: Dict[str, float] = {e: 0.0 for e in UNIFIED_EMOTIONS}
        raw_seven_probs: Dict[str, float] = {lbl: 0.0 for lbl in WAV2VEC2_EMOTION_LABELS}

        for idx, p in enumerate(probs):
            raw_lbl = str(id2label.get(idx, id2label.get(str(idx), f"class_{idx}"))).strip().lower()
            pct_val = round(float(p) * 100.0, 2)
            raw_seven_probs[raw_lbl] = pct_val
            mapped = WAV2VEC2_EMO_MAP.get(raw_lbl)
            if mapped:
                emo_probs[mapped] += float(p) * 100.0

        total_p = sum(emo_probs.values())
        if total_p > 0:
            emo_probs = {k: round((v / total_p) * 100.0, 2) for k, v in emo_probs.items()}

        top_raw_label, top_raw_prob = max(raw_seven_probs.items(), key=lambda kv: kv[1])
        dominant_emotion = max(emo_probs.items(), key=lambda kv: kv[1])[0]
        dominant_prob = emo_probs[dominant_emotion]

        voice_pres = self._run_voice_presentation(norm_wav)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "available": True,
            "status": "ready",
            "display_state": dominant_emotion,
            "error": None,
            "model_id": self.config.voice_emotion_model_id,
            "predicted_emotion": dominant_emotion,
            "raw_model_label": top_raw_label,
            "raw_probability_pct": top_raw_prob,
            "probability_pct": dominant_prob,
            "probabilities": emo_probs,
            "raw_probabilities": raw_seven_probs,
            "presentation_estimate": voice_pres,
            "diagnostics": diag,
            "latency_ms": latency_ms,
        }

    def _run_voice_presentation(self, norm_wav: np.ndarray) -> Dict[str, Any]:
        """
        Runs verified pretrained voice presentation classifier if loaded;
        otherwise explicitly returns `'Not available'` without fabricating any score.
        """
        pres_processor, pres_model = self.registry.ensure_voice_presentation_loaded()
        if pres_processor is None or pres_model is None:
            return {
                "available": False,
                "status": "not_available",
                "model_id": None,
                "label": "Not available",
                "confidence_pct": None,
                "probabilities": {},
                "is_uncertain": True,
                "disclaimer": "No verified voice presentation model active.",
            }

        try:
            inputs = pres_processor(
                norm_wav,
                sampling_rate=self.config.target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = {
                k: v.to(self.registry.device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            with self.registry.torch_lock, torch.inference_mode():
                logits = pres_model(**model_inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()

            id2label = getattr(pres_model.config, "id2label", {})
            fem_pct = 0.0
            masc_pct = 0.0
            for idx, p in enumerate(probs):
                lbl = str(id2label.get(idx, id2label.get(str(idx), ""))).strip().lower()
                if "female" in lbl or "woman" in lbl:
                    fem_pct += float(p) * 100.0
                elif "male" in lbl or "man" in lbl:
                    masc_pct += float(p) * 100.0

            s = fem_pct + masc_pct
            if s <= 0:
                return {
                    "available": False,
                    "status": "not_available",
                    "model_id": self.config.voice_presentation_model_id,
                    "label": "Not available",
                    "confidence_pct": None,
                    "probabilities": {},
                    "is_uncertain": True,
                    "disclaimer": "Voice presentation model did not return valid class scores.",
                }

            fem_pct = round((fem_pct / s) * 100.0, 2)
            masc_pct = round((masc_pct / s) * 100.0, 2)
            top_pct = max(fem_pct, masc_pct)
            margin_pct = abs(fem_pct - masc_pct)

            if top_pct < self.config.vocal_presentation_uncertainty_threshold_pct or margin_pct < 15.0:
                return {
                    "available": True,
                    "status": "uncertain",
                    "model_id": self.config.voice_presentation_model_id,
                    "label": "Uncertain",
                    "confidence_pct": None,
                    "probabilities": {
                        "Masculine-presenting": masc_pct,
                        "Feminine-presenting": fem_pct,
                    },
                    "is_uncertain": True,
                    "disclaimer": (
                        "Acoustic voice presentation estimate (Common Voice) — "
                        "does not determine actual gender identity."
                    ),
                }

            label = "Masculine-presenting" if masc_pct >= fem_pct else "Feminine-presenting"
            return {
                "available": True,
                "status": "estimated",
                "model_id": self.config.voice_presentation_model_id,
                "label": label,
                "confidence_pct": top_pct,
                "probabilities": {
                    "Masculine-presenting": masc_pct,
                    "Feminine-presenting": fem_pct,
                },
                "is_uncertain": False,
                "disclaimer": (
                    "Acoustic voice presentation estimate (Common Voice) — "
                    "does not determine actual gender identity."
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "not_available",
                "model_id": self.config.voice_presentation_model_id,
                "label": "Not available",
                "confidence_pct": None,
                "probabilities": {},
                "is_uncertain": True,
                "error": str(exc),
                "disclaimer": "Voice presentation inference failed; showing Not available.",
            }

    def _non_active_result(
        self,
        status: str,
        display_state: str,
        reason: str,
        diagnostics: Dict[str, Any],
        t0: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        pres_ready = self.registry.status().get("voice_presentation_ready", False)
        pres_label = "Waiting for speech" if (status == "waiting_for_speech" and pres_ready) else "Not available"
        return {
            "available": False,
            "status": status,
            "display_state": display_state,
            "reason": reason,
            "error": error,
            "model_id": self.config.voice_emotion_model_id,
            "predicted_emotion": None,
            "raw_model_label": None,
            "raw_probability_pct": None,
            "probability_pct": None,
            "probabilities": {},
            "raw_probabilities": {},
            "presentation_estimate": {
                "available": False,
                "status": status if pres_ready else "not_available",
                "model_id": self.config.voice_presentation_model_id if pres_ready else None,
                "label": pres_label,
                "confidence_pct": None,
                "probabilities": {},
                "is_uncertain": True,
                "disclaimer": (
                    "Acoustic voice presentation estimate — does not determine actual gender identity."
                ),
            },
            "diagnostics": diagnostics,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
