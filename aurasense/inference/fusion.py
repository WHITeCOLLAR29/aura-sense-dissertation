"""
Multimodal Emotion Fusion Module for AuraSense.

Implements strict, transparent fusion of real predictions from:
1. Facial Emotion Model (`DeepFace` 7-class distribution)
2. Vocal Emotion Model (`Dpngtm/wav2vec2-emotion-recognition` 7-class distribution)

Fusion Steps & Methodological Guarantees:
1. Extracts the 7-class emotion distribution from the voice model and the 7-class distribution
   from the facial model over `['Happy', 'Sad', 'Angry', 'Fearful', 'Surprised', 'Disgust', 'Neutral']`.
2. Strictly normalizes both 7-class distributions so each sums to `1.0` (`100.0%`).
3. Evaluates modality validity and per-modality confidence thresholds (`min_facial_score_threshold`,
   `min_vocal_score_threshold`) plus signal quality diagnostics — never assuming both modalities
   are equally reliable.
4. When both modalities are valid and reliable, computes the configurable weighted sum
   (default baseline `0.5 × Facial + 0.5 × Vocal`, optionally reliability-modulated when signal
   quality differs between face and voice).
5. When only one modality is valid/reliable, uses that available modality alone and explicitly
   records unimodal contribution (`facial_only` or `vocal_only`).
6. When neither modality has a valid, above-threshold prediction, returns `"Insufficient data"`
   without fabricating any scores.
7. Applies configurable temporal smoothing (`"ema"` exponential moving average or `"rolling"`
   window average) AND hysteresis dwell-time stabilization to prevent rapid emotion flickering.
8. Surfaces all scores strictly as uncalibrated **model output scores** for an **estimated expressed
   emotion**, never claiming calibrated probabilities of a person's internal emotional state.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from aurasense.config import LiveInferenceConfig, PASTEL_PALETTES, UNIFIED_EMOTIONS
from .face_analyzer import PretrainedFaceAnalyzer
from .model_registry import PretrainedModelRegistry, get_model_registry
from .voice_analyzer import PretrainedVoiceAnalyzer


@dataclass
class FusionRuntimeSettings:
    """Mutable user-configurable fusion parameters controlled live from the dashboard."""

    facial_weight: float = 0.50
    vocal_weight: float = 0.50
    adapt_by_reliability: bool = True
    min_modality_score_threshold: float = 24.0
    min_combined_score_threshold: float = 24.0
    smoothing_mode: str = "ema"  # "ema" | "rolling" | "none"
    ema_alpha: float = 0.45
    rolling_window_size: int = 4
    switch_hysteresis_margin: float = 6.0
    min_consecutive_switch_windows: int = 2

    @classmethod
    def from_config(cls, cfg: LiveInferenceConfig) -> "FusionRuntimeSettings":
        return cls(
            facial_weight=cfg.facial_fusion_weight,
            vocal_weight=cfg.vocal_fusion_weight,
            adapt_by_reliability=cfg.adapt_weights_by_reliability,
            min_modality_score_threshold=cfg.min_facial_score_threshold,
            min_combined_score_threshold=cfg.min_combined_score_threshold,
            smoothing_mode=cfg.smoothing_mode,
            ema_alpha=cfg.ema_alpha,
            rolling_window_size=cfg.rolling_average_window,
            switch_hysteresis_margin=cfg.switch_hysteresis_margin,
            min_consecutive_switch_windows=cfg.min_consecutive_switch_windows,
        )


def normalize_distribution(raw_scores: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Normalizes an arbitrary non-negative 7-class emotion score dictionary onto `[0.0, 1.0]`
    summing strictly to `1.0`. Returns `None` if input is empty or zero-sum (never fabricates).
    """
    if not raw_scores:
        return None
    vec = np.array(
        [max(0.0, float(raw_scores.get(lbl, 0.0))) for lbl in UNIFIED_EMOTIONS],
        dtype=np.float64,
    )
    total = float(np.sum(vec))
    if not np.isfinite(total) or total <= 1e-9:
        return None
    normed = vec / total
    return {lbl: float(normed[i]) for i, lbl in enumerate(UNIFIED_EMOTIONS)}


def to_percentage_dict(unit_dist: Dict[str, float]) -> Dict[str, float]:
    """Converts a normalized `[0.0, 1.0]` distribution to display percentages summing to `100.0`."""
    vec = np.array([max(0.0, float(unit_dist.get(lbl, 0.0))) for lbl in UNIFIED_EMOTIONS], dtype=np.float64)
    s = float(np.sum(vec))
    if s <= 1e-9:
        return {}
    pct = (vec / s) * 100.0
    return {lbl: round(float(pct[i]), 2) for i, lbl in enumerate(UNIFIED_EMOTIONS)}


def jensen_shannon_divergence_unit(p_unit: Dict[str, float], q_unit: Dict[str, float]) -> float:
    """Computes normalized base-2 Jensen-Shannon Divergence in `[0.0, 1.0]` between unit distributions."""
    eps = 1e-8
    p = np.array([max(eps, float(p_unit.get(k, 0.0))) for k in UNIFIED_EMOTIONS], dtype=np.float64)
    q = np.array([max(eps, float(q_unit.get(k, 0.0))) for k in UNIFIED_EMOTIONS], dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
    return round(float(np.clip(jsd, 0.0, 1.0)), 4)


class MultimodalEmotionFusionEngine:
    """
    Combines normalized 7-class facial and vocal emotion distributions with configurable weights,
    reliability gating, confidence thresholds, temporal smoothing, and anti-switching hysteresis.
    """

    def __init__(
        self,
        registry: Optional[PretrainedModelRegistry] = None,
        config: Optional[LiveInferenceConfig] = None,
        settings: Optional[FusionRuntimeSettings] = None,
    ) -> None:
        self.config = config or LiveInferenceConfig()
        self.registry = registry or get_model_registry(self.config)
        self.settings = settings or FusionRuntimeSettings.from_config(self.config)
        self.face_analyzer = PretrainedFaceAnalyzer(self.registry, self.config)
        self.voice_analyzer = PretrainedVoiceAnalyzer(self.registry, self.config)

        # Temporal smoothing & anti-switching state
        self._ema_unit: Optional[Dict[str, float]] = None
        self._rolling_history: Deque[Dict[str, float]] = deque(maxlen=10)
        self._locked_emotion: Optional[str] = None
        self._candidate_emotion: Optional[str] = None
        self._candidate_streak: int = 0

    def reset_smoothing(self) -> None:
        """Clears temporal smoothing buffers and emotion-lock state on session reset/stop."""
        self._ema_unit = None
        self._rolling_history.clear()
        self._locked_emotion = None
        self._candidate_emotion = None
        self._candidate_streak = 0

    def analyze_live_window(
        self,
        frames_bgr: List[np.ndarray],
        audio_waveform: np.ndarray,
        audio_sr: int = 16000,
        enable_ema: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Runs facial + vocal pretrained models on the live window and fuses their distributions."""
        t0 = time.perf_counter()
        if enable_ema is not None:
            if enable_ema and self.settings.smoothing_mode == "none":
                self.settings.smoothing_mode = "ema"
            elif not enable_ema and self.settings.smoothing_mode == "ema":
                self.settings.smoothing_mode = "none"

        face_res = self.face_analyzer.analyze_frames(frames_bgr)
        voice_res = self.voice_analyzer.analyze_waveform(audio_waveform, input_sr=audio_sr)

        fused = self.fuse_distributions(face_res, voice_res)
        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        errors: List[str] = []
        if face_res.get("error"):
            errors.append(str(face_res["error"]))
        if voice_res.get("error"):
            errors.append(str(voice_res["error"]))

        dom_emo = fused.get("dominant_emotion")
        theme_key = dom_emo or "Neutral"
        if (
            dom_emo == "Neutral"
            and voice_res.get("available")
            and voice_res.get("raw_model_label") == "calm"
        ):
            theme_key = "Calm"

        return {
            "available": fused["available"],
            "display_label": fused["display_label"],
            "timestamp": datetime.now().astimezone().isoformat(),
            "fusion_mode": fused["fusion_mode"],
            "contributing_modalities": fused["contributing_modalities"],
            "contribution_banner": fused["contribution_banner"],
            "dominant_emotion": dom_emo,
            "theme_key": theme_key,
            "pastel_theme": PASTEL_PALETTES.get(theme_key, PASTEL_PALETTES["Neutral"]),
            "emotion_confidence": fused["emotion_confidence"],
            "raw_dominant_emotion": fused.get("raw_dominant_emotion"),
            "raw_emotion_confidence": fused.get("raw_emotion_confidence"),
            "emotion_scores": fused["emotion_scores"],
            "combined_probabilities": fused["combined_probabilities"],
            "normalized_facial_probabilities": fused["normalized_facial_probabilities"],
            "normalized_vocal_probabilities": fused["normalized_vocal_probabilities"],
            "modality_weights": fused["modality_weights"],
            "disagreement": fused["disagreement"],
            "face_analysis": face_res,
            "voice_analysis": voice_res,
            "dual_facial_gender": face_res.get("dual_gender", {}),
            "facial_presentation": face_res.get("presentation_estimate", {}),
            "voice_presentation": voice_res.get("presentation_estimate", {}),
            "warnings": fused["warnings"],
            "errors": errors,
            "smoothing_applied": fused["smoothing_applied"],
            "smoothing_mode": self.settings.smoothing_mode,
            "switch_stabilized": fused["switch_stabilized"],
            "expressed_emotion_disclaimer": (
                "Estimated expressed emotion derived from observable facial and vocal cues. "
                "Model output scores are uncalibrated softmax scores — not calibrated probabilities "
                "of a person's internal feelings."
            ),
            "total_latency_ms": total_latency_ms,
        }

    def fuse_distributions(
        self,
        face_res: Dict[str, Any],
        voice_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Executes the 7-step multimodal emotion fusion algorithm with reliability checks,
        confidence thresholds, temporal smoothing, and anti-switching hysteresis.
        """
        s = self.settings
        warnings: List[str] = []

        # Steps 1–3: Extract & normalize both 7-class distributions
        face_unit = (
            normalize_distribution(face_res.get("probabilities") or {})
            if face_res.get("available")
            else None
        )
        voice_unit = (
            normalize_distribution(voice_res.get("probabilities") or {})
            if voice_res.get("available")
            else None
        )

        face_pct = to_percentage_dict(face_unit) if face_unit is not None else {}
        voice_pct = to_percentage_dict(voice_unit) if voice_unit is not None else {}

        # Step 4: Check modality validity & confidence thresholds
        face_top_score = max(face_pct.values()) if face_pct else 0.0
        voice_top_score = max(voice_pct.values()) if voice_pct else 0.0

        face_qual = float(face_res.get("diagnostics", {}).get("quality_score", 0.0)) if face_unit else 0.0
        voice_qual = float(voice_res.get("diagnostics", {}).get("quality_score", 0.0)) if voice_unit else 0.0

        face_valid = bool(
            face_unit is not None
            and face_top_score >= s.min_modality_score_threshold
            and face_qual >= 0.20
        )
        voice_valid = bool(
            voice_unit is not None
            and voice_top_score >= s.min_modality_score_threshold
            and voice_qual >= 0.20
        )

        if face_unit is not None and not face_valid:
            warnings.append(
                f"Facial emotion score ({face_top_score:.1f}%) is below threshold "
                f"({s.min_modality_score_threshold:.0f}%); excluded from fusion."
            )
        elif face_unit is not None:
            f_diag = face_res.get("diagnostics") or {}
            if f_diag.get("low_lighting"):
                warnings.append("Low lighting detected on face; CLAHE applied & visual fusion weight moderated.")
            elif f_diag.get("poor_lighting"):
                warnings.append("Harsh/overexposed facial lighting detected; visual fusion weight moderated.")
            if f_diag.get("rapid_motion_or_blur"):
                warnings.append("Rapid face movement or motion blur detected; visual fusion weight moderated.")
            if f_diag.get("multiple_faces"):
                n_f = f_diag.get("detected_face_count", 2)
                warnings.append(f"Multiple faces detected ({n_f}); analyzing primary foreground face.")
        elif not face_res.get("available") and face_res.get("reason"):
            warnings.append(str(face_res["reason"]))

        if voice_unit is not None and not voice_valid:
            warnings.append(
                f"Vocal emotion score ({voice_top_score:.1f}%) is below threshold "
                f"({s.min_modality_score_threshold:.0f}%); excluded from fusion."
            )
        elif not voice_res.get("available") and voice_res.get("reason"):
            warnings.append(str(voice_res["reason"]))

        # Step 7: Neither modality is valid/reliable -> "Insufficient data"
        if not face_valid and not voice_valid:
            return self._insufficient_data_result(
                warnings=warnings,
                face_pct=face_pct,
                voice_pct=voice_pct,
                reason="Neither facial nor vocal modality produced a reliable above-threshold prediction.",
            )

        # Normalize user-configured base weights (default 0.50 Facial + 0.50 Vocal)
        cfg_wf = max(0.0, float(s.facial_weight))
        cfg_wv = max(0.0, float(s.vocal_weight))
        if cfg_wf + cfg_wv <= 1e-6:
            cfg_wf, cfg_wv = 0.5, 0.5
        base_wf = cfg_wf / (cfg_wf + cfg_wv)
        base_wv = cfg_wv / (cfg_wf + cfg_wv)

        # Steps 5 & 6: Compute weighted multimodal or unimodal unit distribution
        if face_valid and voice_valid and face_unit is not None and voice_unit is not None:
            if s.adapt_by_reliability:
                # Modulate configured base weights by live modality reliability (quality * peak clarity)
                rel_f = face_qual * (0.5 + 0.5 * (face_top_score / 100.0))
                rel_v = voice_qual * (0.5 + 0.5 * (voice_top_score / 100.0))
                raw_wf = base_wf * max(0.20, rel_f)
                raw_wv = base_wv * max(0.20, rel_v)
                eff_wf = raw_wf / (raw_wf + raw_wv)
                eff_wv = raw_wv / (raw_wf + raw_wv)
                weight_method = (
                    f"Configured ({base_wf:.2f} Face / {base_wv:.2f} Voice) adjusted by live signal reliability"
                )
            else:
                # Exact user-configured linear combination: e.g. (0.5 * Facial) + (0.5 * Vocal)
                eff_wf = base_wf
                eff_wv = base_wv
                weight_method = f"Fixed weighted fusion ({eff_wf:.2f} × Facial + {eff_wv:.2f} × Vocal)"

            instant_unit = {
                emo: eff_wf * face_unit[emo] + eff_wv * voice_unit[emo]
                for emo in UNIFIED_EMOTIONS
            }
            instant_unit = normalize_distribution(instant_unit) or face_unit

            fusion_mode = "multimodal"
            contributing = ["facial", "vocal"]
            contribution_banner = (
                f"Multimodal Fusion ({eff_wf * 100:.0f}% Facial + {eff_wv * 100:.0f}% Vocal)"
            )

            f_dom = max(face_pct.items(), key=lambda kv: kv[1])[0]
            v_dom = max(voice_pct.items(), key=lambda kv: kv[1])[0]
            agree = bool(f_dom == v_dom)
            jsd = jensen_shannon_divergence_unit(face_unit, voice_unit)
            disagreement = {
                "modalities_agree": agree,
                "js_divergence": jsd,
                "status": "Modalities Concordant" if agree else (
                    "High Cross-Modal Divergence" if jsd >= 0.22 else "Moderate Modality Difference"
                ),
                "message": (
                    f"Facial ({f_dom}) and Vocal ({v_dom}) expressions align (JSD={jsd:.3f})."
                    if agree
                    else f"Facial expresses {f_dom} while Vocal expresses {v_dom} (JSD={jsd:.3f})."
                ),
            }
        elif face_valid and face_unit is not None:
            # Step 6a: Facial modality only
            eff_wf, eff_wv = 1.0, 0.0
            instant_unit = dict(face_unit)
            fusion_mode = "facial_only"
            contributing = ["facial"]
            v_state = voice_res.get("display_state") or "Unavailable"
            contribution_banner = f"Unimodal: Facial Emotion Only (Vocal: {v_state})"
            weight_method = "Unimodal fallback — only Facial modality had a valid prediction"
            disagreement = {
                "modalities_agree": None,
                "js_divergence": None,
                "status": "Facial Modality Only",
                "message": f"Only facial expression contributed ({v_state}).",
            }
        else:
            # Step 6b: Vocal modality only
            assert voice_unit is not None
            eff_wf, eff_wv = 0.0, 1.0
            instant_unit = dict(voice_unit)
            fusion_mode = "vocal_only"
            contributing = ["vocal"]
            contribution_banner = "Unimodal: Vocal Emotion Only (No Reliable Face Detected)"
            weight_method = "Unimodal fallback — only Vocal modality had a valid prediction"
            disagreement = {
                "modalities_agree": None,
                "js_divergence": None,
                "status": "Vocal Modality Only",
                "message": "Only vocal expression contributed (no reliable face in frame).",
            }

        instant_pct = to_percentage_dict(instant_unit)
        raw_dom, raw_top_score = max(instant_pct.items(), key=lambda kv: kv[1])

        # Apply Temporal Smoothing (Rolling Average or Exponential Moving Average)
        smoothed_unit, smoothing_applied = self._apply_temporal_smoothing(instant_unit)
        smoothed_pct = to_percentage_dict(smoothed_unit)

        smoothed_top_emo, smoothed_top_score = max(smoothed_pct.items(), key=lambda kv: kv[1])

        # Enforce Combined Score Threshold -> "Insufficient data" if distribution is too flat/ambiguous
        if smoothed_top_score < s.min_combined_score_threshold:
            return self._insufficient_data_result(
                warnings=warnings
                + [
                    f"Top combined score ({smoothed_top_score:.1f}%) is below combined confidence "
                    f"threshold ({s.min_combined_score_threshold:.0f}%)."
                ],
                face_pct=face_pct,
                voice_pct=voice_pct,
                reason="Combined distribution is below minimum confidence threshold.",
            )

        # Apply Hysteresis & Dwell-Time Stabilization to prevent rapid emotion switching
        stabilized_emo, switch_stabilized = self._stabilize_dominant_emotion(
            smoothed_pct=smoothed_pct,
            candidate_top=smoothed_top_emo,
        )
        stabilized_score = smoothed_pct[stabilized_emo]

        emotion_scores = [
            {
                "emotion": emo,
                "probability": smoothed_pct[emo],
                "raw_probability": instant_pct[emo],
                "face_probability": face_pct.get(emo) if face_valid else None,
                "voice_probability": voice_pct.get(emo) if voice_valid else None,
            }
            for emo in sorted(UNIFIED_EMOTIONS, key=lambda k: smoothed_pct[k], reverse=True)
        ]

        return {
            "available": True,
            "display_label": stabilized_emo,
            "fusion_mode": fusion_mode,
            "contributing_modalities": contributing,
            "contribution_banner": contribution_banner,
            "dominant_emotion": stabilized_emo,
            "emotion_confidence": stabilized_score,
            "raw_dominant_emotion": raw_dom,
            "raw_emotion_confidence": raw_top_score,
            "combined_probabilities": smoothed_pct,
            "normalized_facial_probabilities": face_pct if face_valid else {},
            "normalized_vocal_probabilities": voice_pct if voice_valid else {},
            "emotion_scores": emotion_scores,
            "modality_weights": {
                "face_pct": round(eff_wf * 100.0, 1),
                "voice_pct": round(eff_wv * 100.0, 1),
                "configured_face_weight": round(base_wf, 2),
                "configured_voice_weight": round(base_wv, 2),
                "face_quality": face_qual,
                "voice_quality": voice_qual,
                "method": weight_method,
            },
            "disagreement": disagreement,
            "smoothing_applied": smoothing_applied,
            "switch_stabilized": switch_stabilized,
            "warnings": warnings,
        }

    def _apply_temporal_smoothing(
        self,
        instant_unit: Dict[str, float],
    ) -> tuple[Dict[str, float], bool]:
        """Applies either EMA or Rolling Window Average over normalized unit distributions."""
        mode = (self.settings.smoothing_mode or "ema").lower()
        if mode == "rolling":
            win_n = max(2, int(self.settings.rolling_window_size))
            if self._rolling_history.maxlen != win_n:
                self._rolling_history = deque(list(self._rolling_history)[-win_n:], maxlen=win_n)
            self._rolling_history.append(dict(instant_unit))
            avg = {
                emo: float(np.mean([h[emo] for h in self._rolling_history]))
                for emo in UNIFIED_EMOTIONS
            }
            return normalize_distribution(avg) or instant_unit, len(self._rolling_history) > 1

        if mode == "ema":
            alpha = float(np.clip(self.settings.ema_alpha, 0.10, 0.95))
            if self._ema_unit is None:
                self._ema_unit = dict(instant_unit)
                return dict(self._ema_unit), False
            blended = {
                emo: alpha * instant_unit[emo] + (1.0 - alpha) * self._ema_unit[emo]
                for emo in UNIFIED_EMOTIONS
            }
            self._ema_unit = normalize_distribution(blended) or instant_unit
            return dict(self._ema_unit), True

        return dict(instant_unit), False

    def _stabilize_dominant_emotion(
        self,
        smoothed_pct: Dict[str, float],
        candidate_top: str,
    ) -> tuple[str, bool]:
        """
        Prevents rapid emotion flickering between consecutive windows:
        Switching away from `_locked_emotion` requires either:
        - `candidate_top` leading `_locked_emotion` by >= `2 * switch_hysteresis_margin` immediately, OR
        - `candidate_top` leading `_locked_emotion` by >= `switch_hysteresis_margin` for
          `min_consecutive_switch_windows` consecutive windows.
        """
        if self._locked_emotion is None or self._locked_emotion not in smoothed_pct:
            self._locked_emotion = candidate_top
            self._candidate_emotion = None
            self._candidate_streak = 0
            return candidate_top, False

        if candidate_top == self._locked_emotion:
            self._candidate_emotion = None
            self._candidate_streak = 0
            return self._locked_emotion, False

        cur_locked_score = float(smoothed_pct.get(self._locked_emotion, 0.0))
        cand_score = float(smoothed_pct.get(candidate_top, 0.0))
        lead = cand_score - cur_locked_score
        margin = max(0.0, float(self.settings.switch_hysteresis_margin))
        req_streak = max(1, int(self.settings.min_consecutive_switch_windows))

        # Decisive expression shift (strong margin) switches immediately
        if lead >= max(margin * 2.2, 16.0):
            self._locked_emotion = candidate_top
            self._candidate_emotion = None
            self._candidate_streak = 0
            return candidate_top, False

        # Otherwise require sustained margin across `req_streak` consecutive windows
        if lead >= margin:
            if self._candidate_emotion == candidate_top:
                self._candidate_streak += 1
            else:
                self._candidate_emotion = candidate_top
                self._candidate_streak = 1

            if self._candidate_streak >= req_streak:
                self._locked_emotion = candidate_top
                self._candidate_emotion = None
                self._candidate_streak = 0
                return candidate_top, False

        else:
            self._candidate_emotion = None
            self._candidate_streak = 0

        # Hold previous locked expressed emotion to prevent single-window jitter
        return self._locked_emotion, True

    @staticmethod
    def _insufficient_data_result(
        warnings: List[str],
        face_pct: Dict[str, float],
        voice_pct: Dict[str, float],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "available": False,
            "display_label": "Insufficient data",
            "fusion_mode": "insufficient_data",
            "contributing_modalities": [],
            "contribution_banner": "Insufficient data — neither modality has a reliable prediction",
            "dominant_emotion": None,
            "emotion_confidence": None,
            "raw_dominant_emotion": None,
            "raw_emotion_confidence": None,
            "combined_probabilities": {},
            "normalized_facial_probabilities": face_pct,
            "normalized_vocal_probabilities": voice_pct,
            "emotion_scores": [],
            "modality_weights": {"voice_pct": 0.0, "face_pct": 0.0},
            "disagreement": {
                "modalities_agree": None,
                "js_divergence": None,
                "status": "Insufficient Data",
                "message": reason,
            },
            "smoothing_applied": False,
            "switch_stabilized": False,
            "warnings": warnings,
        }
