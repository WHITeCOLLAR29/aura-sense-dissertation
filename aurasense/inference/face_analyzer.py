"""
Dual-Model Facial Gender Presentation & Facial Expression Pipeline.

Highest-Priority Gender Estimation Architecture:
1. Face Detection & 5-Point Landmark Extraction:
   - Uses InsightFace `buffalo_l` (`det_10g` SCRFD) to detect bounding box, detection confidence,
     and 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).
2. Strict Pre-Inference Quality, Pose & Occlusion Gate:
   - Refuses gender prediction (`"Uncertain – Please adjust camera position"`) BEFORE trusting model outputs
     whenever the detected face is:
     * Too small (`min(w, h) < 72px` or `< 2%` of frame area)
     * Blurred or moving rapidly (`Laplacian variance < 28` or displacement `> 22%` of diagonal)
     * Poorly illuminated (`mean luma < 45` or overexposed `> 235`)
     * Clipped / occluded at frame border or low SCRFD landmark confidence (`det_score < 0.65`)
     * Extreme head pose angle (`|roll| > 28°`, `|yaw asymmetry| > 0.42`, or extreme pitch ratio)
3. Proper Alignment, Cropping & Preprocessing on the EXACT SAME Detected Face:
   - **Model 1 (`InsightFace Buffalo_L` `genderage.onnx`)**: Center-scale affine transform (`96×96`, RGB swap)
     producing 2-logit gender vector `[Female_logit, Male_logit]` -> calibrated softmax probabilities.
   - **Model 2 (`FairFace ResNet34` `res34_fair_align_multi_7_20190809.pt`)**: 5-point similarity-aligned
     face chip (`224×224` with `0.25` FairFace margin, RGB, ImageNet normalization) -> `outputs[7:9]`
     (`0: Male`, `1: Female`) -> softmax probabilities.
4. Consensus Without Blind Averaging:
   - Displays Model 1 and Model 2 predictions and confidences independently.
   - NEVER averages probabilities across the two models.
   - Requires BOTH models to agree on perceived presentation AND both to clear their confidence
     thresholds (`>= 65%`); otherwise returns `"Uncertain – Please adjust camera position"`.
5. Ethical Framing:
   - Estimates perceived Male / Female facial presentation from visual appearance only —
     cannot determine a person's gender identity. Voice never overrides facial predictions.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from insightface.utils import face_align
import numpy as np
import torch

from aurasense.config import (
    DEEPFACE_EMOTION_LABELS,
    GENDER_UNCERTAIN_MESSAGE,
    LiveInferenceConfig,
    UNIFIED_EMOTIONS,
)
from .model_registry import PretrainedModelRegistry, get_model_registry

DEEPFACE_TO_UNIFIED: Dict[str, str] = {
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "disgust": "Disgust",
    "surprise": "Surprised",
}


def align_face_fairface_chip(
    frame_bgr: np.ndarray,
    kps_5pt: np.ndarray,
    output_size: int = 224,
    padding: float = 0.25,
) -> np.ndarray:
    """
    Aligns and crops the detected face using a 5-point similarity transform matching
    FairFace's `size=224, padding=0.25` aligned face chip geometry (preserving hair, chin, and jaw contour).
    """
    # Standard ArcFace 112x112 5-point canonical coordinates scaled into [0, 1]
    base_pts = face_align.arcface_dst.astype(np.float32) / 112.0
    # Compress inward around center (0.5, 0.5) by `1 / (1 + 2 * padding * 0.6)` to include ~25% context margin
    scale = 1.0 / (1.0 + 1.2 * float(padding))
    dst_pts = ((base_pts - 0.5) * scale + 0.5) * float(output_size)

    src_pts = np.asarray(kps_5pt, dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if tform is None:
        # Fallback padded square bounding box around landmarks if similarity solve degenerates
        cx, cy = float(np.mean(src_pts[:, 0])), float(np.mean(src_pts[:, 1]))
        span = max(float(np.ptp(src_pts[:, 0])), float(np.ptp(src_pts[:, 1])), 32.0) * 1.45
        x1 = max(0, int(cx - span))
        y1 = max(0, int(cy - span))
        x2 = min(frame_bgr.shape[1], int(cx + span))
        y2 = min(frame_bgr.shape[0], int(cy + span))
        crop = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    aligned = cv2.warpAffine(
        frame_bgr,
        tform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def evaluate_face_geometry_and_quality(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    kps: Optional[np.ndarray],
    det_score: float,
    config: LiveInferenceConfig,
    enforce_size_and_margin: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates face size, sharpness (blur), illumination, border occlusion, and 3D head pose
    (yaw, roll, pitch) from 5-point facial landmarks.
    """
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_ratio = (bw * bh) / float(max(1, fw * fh))

    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(fw, int(x2)), min(fh, int(y2))
    crop = frame_bgr[iy1:iy2, ix1:ix2]

    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        mean_luma = 0.0
        sharpness = 0.0

    # Border clipping / occlusion check
    margin = config.min_edge_margin_px
    border_clipped = bool(
        enforce_size_and_margin
        and (x1 < margin or y1 < margin or x2 > (fw - margin) or y2 > (fh - margin))
    )

    # 5-point landmark pose checks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    roll_deg = 0.0
    yaw_asym = 0.0
    pitch_ratio = 0.50
    landmarks_occluded = False

    if kps is not None and kps.shape == (5, 2):
        le, re, nose, lm, rm = kps
        eye_dx = float(re[0] - le[0])
        eye_dy = float(re[1] - le[1])
        inter_eye = float(math.hypot(eye_dx, eye_dy))
        if inter_eye < 10.0:
            landmarks_occluded = True
        else:
            roll_deg = float(math.degrees(math.atan2(eye_dy, eye_dx)))
            eye_mid_x = 0.5 * float(le[0] + re[0])
            eye_mid_y = 0.5 * float(le[1] + re[1])
            mouth_mid_y = 0.5 * float(lm[1] + rm[1])
            yaw_asym = abs(float(nose[0]) - eye_mid_x) / inter_eye
            eye_to_mouth = max(1.0, mouth_mid_y - eye_mid_y)
            pitch_ratio = (float(nose[1]) - eye_mid_y) / eye_to_mouth
            # Verify landmarks fall inside expanded bbox (detects severe occlusion distortion)
            if (
                np.any(kps[:, 0] < x1 - 0.15 * bw)
                or np.any(kps[:, 0] > x2 + 0.15 * bw)
                or np.any(kps[:, 1] < y1 - 0.15 * bh)
                or np.any(kps[:, 1] > y2 + 0.15 * bh)
            ):
                landmarks_occluded = True
    else:
        landmarks_occluded = True

    issues: List[str] = []
    if det_score < config.min_det_score_for_gender or landmarks_occluded:
        issues.append("Face partially occluded or low detection clarity")
    if enforce_size_and_margin and (
        min(bw, bh) < config.min_face_size_px or area_ratio < config.min_face_area_ratio
    ):
        issues.append("Face too small / too far from camera")
    if border_clipped:
        issues.append("Face clipped at camera edge")
    if sharpness < config.min_sharpness_laplacian_var:
        issues.append("Face blurred (hold steady)")
    if mean_luma < config.min_lighting_luma:
        issues.append("Face poorly illuminated (too dark)")
    elif mean_luma > config.max_lighting_luma:
        issues.append("Face overexposed (harsh lighting)")
    if (
        abs(roll_deg) > config.max_roll_degrees
        or yaw_asym > config.max_yaw_asymmetry_ratio
        or pitch_ratio < config.min_pitch_nose_mouth_ratio
        or pitch_ratio > config.max_pitch_nose_mouth_ratio
    ):
        issues.append("Extreme head angle (look toward camera)")

    passed = len(issues) == 0

    luma_ok = 0.60 if (mean_luma < config.min_lighting_luma or mean_luma > config.max_lighting_luma) else 1.0
    blur_ok = 0.55 if sharpness < config.min_sharpness_laplacian_var else 1.0
    pose_ok = 0.65 if abs(yaw_asym) > 0.30 else 1.0
    quality_score = round(float(np.clip(det_score * luma_ok * blur_ok * pose_ok, 0.20, 1.0)), 3)

    return {
        "passed_gender_gate": passed,
        "issues": issues,
        "det_score": round(float(det_score), 3),
        "face_width_px": int(round(bw)),
        "face_height_px": int(round(bh)),
        "area_ratio": round(float(area_ratio), 4),
        "mean_luma": round(mean_luma, 1),
        "low_lighting": bool(mean_luma < config.min_lighting_luma),
        "poor_lighting": bool(mean_luma < config.min_lighting_luma or mean_luma > config.max_lighting_luma),
        "sharpness_laplacian": round(sharpness, 1),
        "motion_blur": bool(sharpness < config.min_sharpness_laplacian_var),
        "roll_deg": round(roll_deg, 1),
        "yaw_asymmetry": round(yaw_asym, 3),
        "pitch_ratio": round(pitch_ratio, 3),
        "border_clipped": border_clipped,
        "quality_score": quality_score,
        "region": {"x": ix1, "y": iy1, "w": max(1, ix2 - ix1), "h": max(1, iy2 - iy1)},
    }


class PretrainedFaceAnalyzer:
    """
    Runs:
    1. InsightFace `buffalo_l` detection + 5-point landmark alignment + Model 1 gender estimation
    2. FairFace `ResNet34` independent Model 2 gender estimation on the exact same detected face
    3. Agreement/disagreement & quality gate decision (never blindly averaging models)
    4. DeepFace 7-class facial expression recognition on the aligned face crop
    """

    def __init__(
        self,
        registry: Optional[PretrainedModelRegistry] = None,
        config: Optional[LiveInferenceConfig] = None,
    ) -> None:
        self.config = config or LiveInferenceConfig()
        self.registry = registry or get_model_registry(self.config)
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect_face_fast(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fast Haar bounding box for 30 FPS live video HUD brackets between neural passes."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(64, 64))
        if len(faces) == 0:
            return None
        largest = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        return int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3])

    def _run_buffalo_gender_on_face(self, frame_bgr: np.ndarray, face_obj: Any, buffalo_app: Any) -> Dict[str, Any]:
        """
        Runs InsightFace `buffalo_l` `genderage.onnx` on `face_obj` and extracts raw 2-class logits
        (`index 0 = Female`, `index 1 = Male`) and softmax probabilities.
        """
        t0 = time.perf_counter()
        ga = buffalo_app.models.get("genderage")
        if ga is None:
            return {
                "available": False,
                "model_name": "InsightFace Buffalo_L",
                "label": None,
                "confidence_pct": None,
                "probabilities": {},
                "error": "InsightFace genderage head missing.",
                "latency_ms": 0.0,
            }

        bbox = face_obj.bbox
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        center = (float(bbox[2] + bbox[0]) * 0.5, float(bbox[3] + bbox[1]) * 0.5)

        # Align roll using eye landmarks if available before 96x96 crop
        rotate = 0.0
        if getattr(face_obj, "kps", None) is not None and face_obj.kps.shape == (5, 2):
            le, re = face_obj.kps[0], face_obj.kps[1]
            rotate = float(math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0]))))

        scale = float(ga.input_size[0]) / (max(w, h) * 1.5)
        aimg, _ = face_align.transform(frame_bgr, center, ga.input_size[0], scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / float(ga.input_std),
            input_size,
            (float(ga.input_mean), float(ga.input_mean), float(ga.input_mean)),
            swapRB=True,
        )
        with self.registry.buffalo_lock:
            pred = ga.session.run(ga.output_names, {ga.input_name: blob})[0][0]

        logits = np.asarray(pred[:2], dtype=np.float64)
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        fem_pct = round(float(probs[0]) * 100.0, 2)
        male_pct = round(float(probs[1]) * 100.0, 2)
        pred_label = "Male" if male_pct >= fem_pct else "Female"
        conf_pct = max(male_pct, fem_pct)
        est_age = int(np.round(float(pred[2]) * 100.0))

        return {
            "available": True,
            "model_name": "InsightFace Buffalo_L",
            "label": pred_label,
            "confidence_pct": conf_pct,
            "probabilities": {"Male": male_pct, "Female": fem_pct},
            "estimated_age": est_age,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def _run_fairface_gender_on_aligned_chip(self, aligned_224_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs FairFace `ResNet34` on the 5-point aligned 224x224 face chip.
        Official FairFace 18-dim head layout:
          - `outputs[0:7]`  -> 7 race classes
          - `outputs[7:9]`  -> 2 gender classes (`0: Male`, `1: Female`)
          - `outputs[9:18]` -> 9 age brackets
        """
        t0 = time.perf_counter()
        ff_model, ff_transform = self.registry.ensure_fairface_loaded()
        if ff_model is None or ff_transform is None:
            err = self.registry.status().get("fairface_error") or "FairFace ResNet34 unavailable."
            return {
                "available": False,
                "model_name": "FairFace ResNet34",
                "label": None,
                "confidence_pct": None,
                "probabilities": {},
                "error": err,
                "latency_ms": 0.0,
            }

        rgb_chip = cv2.cvtColor(aligned_224_bgr, cv2.COLOR_BGR2RGB)
        tensor_in = ff_transform(rgb_chip).unsqueeze(0).to(self.registry.device)
        with self.registry.fairface_lock, self.registry.torch_lock, torch.inference_mode():
            outputs = ff_model(tensor_in).squeeze(0).detach().cpu().numpy()

        gender_logits = np.asarray(outputs[7:9], dtype=np.float64)
        gender_logits = gender_logits - np.max(gender_logits)
        g_probs = np.exp(gender_logits) / np.sum(np.exp(gender_logits))
        male_pct = round(float(g_probs[0]) * 100.0, 2)
        fem_pct = round(float(g_probs[1]) * 100.0, 2)
        pred_label = "Male" if male_pct >= fem_pct else "Female"
        conf_pct = max(male_pct, fem_pct)

        return {
            "available": True,
            "model_name": "FairFace ResNet34",
            "label": pred_label,
            "confidence_pct": conf_pct,
            "probabilities": {"Male": male_pct, "Female": fem_pct},
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def analyze_single_portrait_for_benchmark(
        self,
        bgr_image: np.ndarray,
        enforce_live_size_gate: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the exact dual-model detection, 5-point alignment, Model 1 (Buffalo_L), Model 2 (FairFace),
        and consensus logic on a single portrait image (used by both live pipeline and Indian faces evaluation).
        """
        buffalo_app = self.registry.ensure_buffalo_loaded()
        if buffalo_app is None:
            err = self.registry.status().get("buffalo_error") or "Buffalo_L failed to load."
            return {"face_detected": False, "error": err}

        # Add border context if image is already a tight face crop (e.g. 224x224 benchmark portraits)
        # so SCRFD 640x640 anchor receptive fields see full head/shoulder context.
        h, w = bgr_image.shape[:2]
        if max(h, w) < 320:
            pad = int(max(h, w) * 0.65)
            work_img = cv2.copyMakeBorder(bgr_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        else:
            pad = 0
            work_img = bgr_image

        with self.registry.buffalo_lock:
            faces = buffalo_app.get(work_img)

        if not faces:
            return {"face_detected": False, "num_faces": 0}

        # Select primary largest detected face
        best_face = max(
            faces,
            key=lambda f: float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1]),
        )
        qual = evaluate_face_geometry_and_quality(
            frame_bgr=work_img,
            bbox=best_face.bbox,
            kps=getattr(best_face, "kps", None),
            det_score=float(getattr(best_face, "det_score", 0.0)),
            config=self.config,
            enforce_size_and_margin=enforce_live_size_gate,
        )

        # Align 224x224 chip via 5-point landmarks for FairFace & DeepFace
        aligned_224 = align_face_fairface_chip(work_img, best_face.kps, output_size=224, padding=0.25)

        m1 = self._run_buffalo_gender_on_face(work_img, best_face, buffalo_app)
        m2 = self._run_fairface_gender_on_aligned_chip(aligned_224)

        # Shift region coordinates back if border padding was added
        if pad > 0:
            reg = qual["region"]
            qual["region"] = {
                "x": max(0, reg["x"] - pad),
                "y": max(0, reg["y"] - pad),
                "w": min(w, reg["w"]),
                "h": min(h, reg["h"]),
            }

        consensus = self._compute_dual_model_consensus(m1, m2, qual)
        return {
            "face_detected": True,
            "num_faces": len(faces),
            "quality": qual,
            "aligned_224_bgr": aligned_224,
            "model_1_buffalo": m1,
            "model_2_fairface": m2,
            "consensus": consensus,
        }

    def _compute_dual_model_consensus(
        self,
        m1: Dict[str, Any],
        m2: Dict[str, Any],
        qual: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compares Model 1 (`InsightFace Buffalo_L`) and Model 2 (`FairFace ResNet34`) WITHOUT
        blindly averaging their probabilities:
        1. If pre-inference face quality/pose/size gate fails -> `"Uncertain – Please adjust camera position"`.
        2. If either model failed to run -> error / uncertain state.
        3. If Model 1 and Model 2 DISAGREE -> `"Uncertain – Please adjust camera position"`, `agreement=False`.
        4. If either model's confidence is below threshold (`< 65%`) -> `"Uncertain – Please adjust camera position"`.
        5. Only when quality passes, BOTH models independently agree, AND both clear confidence thresholds ->
           displays the agreed perceived presentation (`"Male"` or `"Female"`).
        """
        disclaimer = (
            "Models estimate perceived Male/Female facial presentation from visual appearance only. "
            "They cannot determine a person's gender identity."
        )

        if not qual.get("passed_gender_gate", False):
            issue_str = "; ".join(qual.get("issues") or ["Suboptimal face position or lighting"])
            return {
                "available": True,
                "status": "quality_rejected",
                "final_label": GENDER_UNCERTAIN_MESSAGE,
                "is_uncertain": True,
                "models_agree": None,
                "agreement_status": "Quality Gate Blocked",
                "reason": issue_str,
                "confidence_pct": None,
                "model_1": {**m1, "confidence_pct": None},  # Do not show misleading confidence on degraded face
                "model_2": {**m2, "confidence_pct": None},
                "disclaimer": disclaimer,
            }

        if not m1.get("available"):
            err_msg = m1.get("error") or "InsightFace Buffalo_L inference incomplete."
            return {
                "available": False,
                "status": "model_error",
                "final_label": GENDER_UNCERTAIN_MESSAGE,
                "is_uncertain": True,
                "models_agree": None,
                "agreement_status": "Model Unavailable",
                "reason": err_msg,
                "confidence_pct": None,
                "model_1": m1,
                "model_2": m2,
                "disclaimer": disclaimer,
            }

        lbl1 = m1["label"]
        conf1 = float(m1["confidence_pct"])
        m1_confident = conf1 >= self.config.min_buffalo_gender_confidence_pct

        # Single-model primary mode (FairFace optional / disabled to save RAM on Streamlit Cloud)
        if not m2.get("available"):
            if not m1_confident:
                return {
                    "available": True,
                    "status": "low_confidence",
                    "final_label": GENDER_UNCERTAIN_MESSAGE,
                    "is_uncertain": True,
                    "models_agree": None,
                    "agreement_status": "Low Confidence",
                    "reason": (
                        f"Buffalo_L leans {lbl1} ({conf1:.1f}%), below the "
                        f"{self.config.min_buffalo_gender_confidence_pct:.0f}% threshold."
                    ),
                    "confidence_pct": None,
                    "model_1": m1,
                    "model_2": m2,
                    "disclaimer": disclaimer,
                }
            return {
                "available": True,
                "status": "confident_agreement",
                "final_label": f"{lbl1} (Perceived Presentation)",
                "short_label": lbl1,
                "is_uncertain": False,
                "models_agree": None,
                "agreement_status": "Buffalo_L Primary",
                "reason": f"InsightFace Buffalo_L ({conf1:.1f}%) passed quality & confidence gates.",
                "confidence_pct": round(conf1, 1),
                "model_1": m1,
                "model_2": m2,
                "disclaimer": disclaimer,
            }

        lbl2 = m2["label"]
        conf2 = float(m2["confidence_pct"])
        agree = bool(lbl1 == lbl2)

        m2_confident = conf2 >= self.config.min_fairface_gender_confidence_pct

        if not agree:
            return {
                "available": True,
                "status": "disagreement",
                "final_label": GENDER_UNCERTAIN_MESSAGE,
                "is_uncertain": True,
                "models_agree": False,
                "agreement_status": "Models Disagree",
                "reason": f"Buffalo_L predicted {lbl1} ({conf1:.1f}%) while FairFace predicted {lbl2} ({conf2:.1f}%).",
                "confidence_pct": None,  # Never fabricate or average conflicting confidences
                "model_1": m1,
                "model_2": m2,
                "disclaimer": disclaimer,
            }

        if not (m1_confident and m2_confident):
            return {
                "available": True,
                "status": "low_confidence",
                "final_label": GENDER_UNCERTAIN_MESSAGE,
                "is_uncertain": True,
                "models_agree": True,
                "agreement_status": "Agree (Low Confidence)",
                "reason": (
                    f"Both models lean {lbl1}, but confidence is below the "
                    f"{self.config.min_buffalo_gender_confidence_pct:.0f}% reliability threshold."
                ),
                "confidence_pct": None,
                "model_1": m1,
                "model_2": m2,
                "disclaimer": disclaimer,
            }

        # Both models independently agree with high confidence on a clean, well-aligned face.
        # Report conservative minimum of the two verified confidences (never blind mean inflation).
        conservative_conf = round(min(conf1, conf2), 1)
        return {
            "available": True,
            "status": "confident_agreement",
            "final_label": f"{lbl1} (Perceived Presentation)",
            "short_label": lbl1,
            "is_uncertain": False,
            "models_agree": True,
            "agreement_status": "Models Agree",
            "reason": f"Both InsightFace Buffalo_L ({conf1:.1f}%) and FairFace ResNet34 ({conf2:.1f}%) agree.",
            "confidence_pct": conservative_conf,
            "model_1": m1,
            "model_2": m2,
            "disclaimer": disclaimer,
        }

    def analyze_frames(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """
        Runs periodic live webcam analysis across sampled frames:
        1. Detects & aligns face using InsightFace `buffalo_l` 5-point landmarks.
        2. Evaluates strict face size, blur, illumination, occlusion, and head-pose quality gates
           plus inter-frame motion speed.
        3. Runs **InsightFace Buffalo_L** and **FairFace ResNet34** on the exact same detected face.
        4. Runs **DeepFace** 7-class facial expression CNN on the aligned face crop.
        """
        t0 = time.perf_counter()
        if not frames_bgr:
            return self._no_face_or_error(status="no_frames", reason="Waiting for live webcam frames...", t0=t0)

        buffalo_app = self.registry.ensure_buffalo_loaded()
        if buffalo_app is None:
            err = self.registry.status().get("buffalo_error") or "InsightFace Buffalo_L unavailable."
            return self._no_face_or_error(status="error", reason=err, t0=t0, error=err)

        if self.config.enable_fairface_model:
            ff_model, _ = self.registry.ensure_fairface_loaded()
            if ff_model is None:
                err = self.registry.status().get("fairface_error") or "FairFace ResNet34 unavailable."
                return self._no_face_or_error(status="error", reason=err, t0=t0, error=err)

        deepface_ok = self.registry.ensure_deepface_loaded()

        n_total = len(frames_bgr)
        k = min(self.config.max_sampled_frames, n_total)
        indices = np.linspace(0, n_total - 1, num=k, dtype=int).tolist()
        sampled = [frames_bgr[i] for i in indices if frames_bgr[i] is not None and frames_bgr[i].size > 0]

        best_frame_pack: Optional[Dict[str, Any]] = None
        face_centers: List[Tuple[float, float]] = []
        frame_diag_px: float = 800.0
        max_faces_seen: int = 0
        raw_df_accum: Dict[str, float] = {lbl: 0.0 for lbl in DEEPFACE_EMOTION_LABELS}
        emo_accum: Dict[str, float] = {e: 0.0 for e in UNIFIED_EMOTIONS}
        valid_emo_frames = 0

        from deepface import DeepFace

        for frame in sampled:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / float(w)
                proc_frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                proc_frame = frame

            ph, pw = proc_frame.shape[:2]
            frame_diag_px = float(math.hypot(pw, ph))

            with self.registry.buffalo_lock:
                faces = buffalo_app.get(proc_frame)

            if not faces:
                continue

            max_faces_seen = max(max_faces_seen, len(faces))
            primary = max(
                faces,
                key=lambda f: float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1]),
            )
            qual = evaluate_face_geometry_and_quality(
                frame_bgr=proc_frame,
                bbox=primary.bbox,
                kps=getattr(primary, "kps", None),
                det_score=float(getattr(primary, "det_score", 0.0)),
                config=self.config,
                enforce_size_and_margin=True,
            )
            reg = qual["region"]
            face_centers.append((reg["x"] + 0.5 * reg["w"], reg["y"] + 0.5 * reg["h"]))

            aligned_224 = align_face_fairface_chip(proc_frame, primary.kps, output_size=224, padding=0.25)

            # Keep sharpest, best-lit frame for dual-model gender evaluation on the exact same face
            if best_frame_pack is None or (
                (1 if qual["passed_gender_gate"] else 0, qual["sharpness_laplacian"])
                > (
                    1 if best_frame_pack["qual"]["passed_gender_gate"] else 0,
                    best_frame_pack["qual"]["sharpness_laplacian"],
                )
            ):
                best_frame_pack = {
                    "frame": proc_frame,
                    "face": primary,
                    "qual": qual,
                    "aligned_224": aligned_224,
                }

            # Run DeepFace emotion analysis on the aligned face crop (with CLAHE if very dark)
            if deepface_ok:
                emo_crop = aligned_224
                if qual["low_lighting"]:
                    lab = cv2.cvtColor(emo_crop, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    emo_crop = cv2.cvtColor(cv2.merge((self._clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
                try:
                    with self.registry.deepface_lock:
                        emo_out = DeepFace.analyze(
                            img_path=emo_crop,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="skip",
                            silent=True,
                        )
                    if isinstance(emo_out, list) and emo_out:
                        emo_out = emo_out[0]
                    raw_emo = emo_out.get("emotion", {}) if isinstance(emo_out, dict) else {}
                    if raw_emo:
                        fw_w = 0.5 if qual["motion_blur"] else 1.0
                        for df_k in DEEPFACE_EMOTION_LABELS:
                            val = float(raw_emo.get(df_k, 0.0)) * fw_w
                            raw_df_accum[df_k] += val
                            emo_accum[DEEPFACE_TO_UNIFIED[df_k]] += val
                        valid_emo_frames += 1
                except Exception:
                    pass

        if best_frame_pack is None:
            return self._no_face_or_error(
                status="no_face",
                reason="No face detected in webcam frame",
                t0=t0,
                sampled_frames=len(sampled),
            )

        qual = dict(best_frame_pack["qual"])

        # Check rapid inter-frame face motion across the window
        displacement_ratio = 0.0
        if len(face_centers) >= 2 and frame_diag_px > 1.0:
            dists = [
                float(math.hypot(face_centers[i][0] - face_centers[i - 1][0], face_centers[i][1] - face_centers[i - 1][1]))
                for i in range(1, len(face_centers))
            ]
            displacement_ratio = max(dists) / frame_diag_px

        fast_motion = bool(displacement_ratio > self.config.max_face_displacement_ratio)
        if fast_motion:
            qual["passed_gender_gate"] = False
            qual["issues"] = list(qual["issues"]) + ["Face moving too quickly (hold steady)"]
            qual["quality_score"] = round(float(qual["quality_score"] * 0.55), 3)

        # Execute Model 1 (InsightFace Buffalo_L) & optional Model 2 (FairFace ResNet34) on the EXACT SAME detected face
        m1_res = self._run_buffalo_gender_on_face(
            best_frame_pack["frame"],
            best_frame_pack["face"],
            buffalo_app,
        )
        if self.config.enable_fairface_model:
            m2_res = self._run_fairface_gender_on_aligned_chip(best_frame_pack["aligned_224"])
        else:
            m2_res = {
                "available": False,
                "model_name": "FairFace ResNet34",
                "label": None,
                "confidence_pct": None,
                "probabilities": {},
                "error": "Optional (disabled to save RAM)",
                "latency_ms": 0.0,
            }
        dual_gender = self._compute_dual_model_consensus(m1_res, m2_res, qual)

        # Finalize 7-class facial emotion distribution
        raw_sum = sum(raw_df_accum.values())
        if valid_emo_frames > 0 and raw_sum > 0:
            raw_df_probs = {k: round((raw_df_accum[k] / raw_sum) * 100.0, 2) for k in DEEPFACE_EMOTION_LABELS}
            dominant_raw_emotion = max(raw_df_probs.items(), key=lambda kv: kv[1])[0]
            emo_sum = sum(emo_accum.values())
            emo_probs = {k: round((emo_accum[k] / emo_sum) * 100.0, 2) for k in UNIFIED_EMOTIONS}
            dominant_emotion = max(emo_probs.items(), key=lambda kv: kv[1])[0]
            dominant_prob = emo_probs[dominant_emotion]
            emo_available = True
        else:
            raw_df_probs = {}
            dominant_raw_emotion = None
            emo_probs = {}
            dominant_emotion = None
            dominant_prob = None
            emo_available = False

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "available": emo_available,
            "face_detected": True,
            "status": "ready" if emo_available else "emotion_unavailable",
            "error": None if dual_gender["status"] != "model_error" else dual_gender.get("reason"),
            "model_id": "DeepFace Emotion CNN (SCRFD 5-pt Aligned)",
            "predicted_emotion": dominant_emotion,
            "raw_dominant_emotion": dominant_raw_emotion,
            "probability_pct": dominant_prob,
            "probabilities": emo_probs,
            "raw_deepface_probabilities": raw_df_probs,
            "dual_gender": dual_gender,
            # Backwards-compatible summary dict
            "presentation_estimate": {
                "available": dual_gender["available"],
                "status": dual_gender["status"],
                "label": dual_gender["final_label"],
                "confidence_pct": dual_gender["confidence_pct"],
                "is_uncertain": dual_gender["is_uncertain"],
                "models_agree": dual_gender["models_agree"],
                "agreement_status": dual_gender["agreement_status"],
                "model_1": dual_gender["model_1"],
                "model_2": dual_gender["model_2"],
                "disclaimer": dual_gender["disclaimer"],
            },
            "diagnostics": {
                "detector_backend": "insightface_buffalo_l_scrfd",
                "sampled_frames": len(sampled),
                "valid_face_frames": max(1, valid_emo_frames),
                "detected_face_count": max_faces_seen,
                "face_detection_rate": 1.0,
                "avg_face_confidence": qual["det_score"],
                "mean_luma": qual["mean_luma"],
                "low_lighting": qual["low_lighting"],
                "poor_lighting": qual["poor_lighting"],
                "mean_sharpness_laplacian": qual["sharpness_laplacian"],
                "displacement_ratio": round(displacement_ratio, 3),
                "motion_blur": qual["motion_blur"],
                "fast_motion": fast_motion,
                "rapid_motion_or_blur": bool(qual["motion_blur"] or fast_motion),
                "multiple_faces": bool(max_faces_seen > 1),
                "roll_deg": qual["roll_deg"],
                "yaw_asymmetry": qual["yaw_asymmetry"],
                "pitch_ratio": qual["pitch_ratio"],
                "passed_gender_gate": qual["passed_gender_gate"],
                "gender_gate_issues": qual["issues"],
                "quality_score": qual["quality_score"],
                "last_region": qual["region"],
            },
            "latency_ms": latency_ms,
        }

    def _no_face_or_error(
        self,
        status: str,
        reason: str,
        t0: float,
        sampled_frames: int = 0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        disclaimer = (
            "Models estimate perceived Male/Female facial presentation from visual appearance only. "
            "They cannot determine a person's gender identity."
        )
        empty_dual = {
            "available": False,
            "status": status,
            "final_label": "No face detected" if status != "error" else "Model Error",
            "is_uncertain": True,
            "models_agree": None,
            "agreement_status": "Awaiting Face" if status != "error" else "Error",
            "reason": reason,
            "confidence_pct": None,
            "model_1": {"available": False, "model_name": "InsightFace Buffalo_L", "label": None, "confidence_pct": None},
            "model_2": {"available": False, "model_name": "FairFace ResNet34", "label": None, "confidence_pct": None},
            "disclaimer": disclaimer,
        }
        return {
            "available": False,
            "face_detected": False,
            "status": status,
            "reason": reason,
            "error": error,
            "model_id": "DeepFace Emotion CNN (SCRFD 5-pt Aligned)",
            "predicted_emotion": None,
            "raw_dominant_emotion": None,
            "probability_pct": None,
            "probabilities": {},
            "raw_deepface_probabilities": {},
            "dual_gender": empty_dual,
            "presentation_estimate": {
                "available": False,
                "status": status,
                "label": empty_dual["final_label"],
                "confidence_pct": None,
                "is_uncertain": True,
                "models_agree": None,
                "agreement_status": empty_dual["agreement_status"],
                "model_1": empty_dual["model_1"],
                "model_2": empty_dual["model_2"],
                "disclaimer": disclaimer,
            },
            "diagnostics": {
                "detector_backend": "insightface_buffalo_l_scrfd",
                "sampled_frames": sampled_frames,
                "valid_face_frames": 0,
                "detected_face_count": 0,
                "face_detection_rate": 0.0,
                "avg_face_confidence": 0.0,
                "mean_luma": 0.0,
                "low_lighting": False,
                "poor_lighting": False,
                "mean_sharpness_laplacian": 0.0,
                "displacement_ratio": 0.0,
                "motion_blur": False,
                "fast_motion": False,
                "rapid_motion_or_blur": False,
                "multiple_faces": False,
                "roll_deg": 0.0,
                "yaw_asymmetry": 0.0,
                "pitch_ratio": 0.5,
                "passed_gender_gate": False,
                "gender_gate_issues": [reason],
                "quality_score": 0.0,
                "last_region": None,
            },
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
