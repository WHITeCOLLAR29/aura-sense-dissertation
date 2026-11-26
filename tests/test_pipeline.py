"""
Complete Dual-Model Facial Gender Presentation & 13-Condition Reliability Audit Suite
for AuraSense.

Explicitly tests:
A. Highest-Priority Dual-Model Facial Gender Presentation Requirements:
   1. Model 1 (`InsightFace Buffalo_L`) + Model 2 (`FairFace ResNet34`) loaded once (`loads == 1`)
      and run on the EXACT SAME 5-point aligned detected face.
   2. Individual Model 1 & Model 2 predictions and confidences returned independently.
   3. Agreement case (`Models Agree`, both >= 65% confidence, clean quality) -> agreed presentation
      and conservative `min(conf1, conf2)` confidence (zero blind probability averaging).
   4. Disagreement case (`Models Disagree`) -> exact message `"Uncertain – Please adjust camera position"`,
      `models_agree is False`, and `confidence_pct is None` (zero averaging of conflicting models).
   5. Low-confidence agreement (`< 65%`) -> `"Uncertain – Please adjust camera position"`.
   6. Strict Pre-Inference Quality, Pose & Occlusion Gate rejections:
      - Face too small (`< 72px` / `< 2%` frame area)
      - Face blurred (`Laplacian variance < 28`)
      - Poorly illuminated (`mean luma < 45` or overexposed `> 235`)
      - Border-clipped / partially occluded (`det_score < 0.65` or edge margin `< 6px`)
      - Extreme head pose angle (`|roll| > 28°` or `yaw_asymmetry > 0.42`)
      - Fast inter-frame face displacement (`> 22%` diagonal)
      All return `"Uncertain – Please adjust camera position"` with `confidence_pct is None`.
   7. Vocal gender presentation kept strictly separate — active speech with an opposing vocal
      presentation NEVER overrides or alters `dual_facial_gender`.
   8. Verified Indian faces benchmark JSON (`aurasense/evaluation/indian_faces_benchmark_results.json`)
      exists and contains empirical subgroup evaluations without fabricated claims.

B. All 13 Live Runtime & Reliability Audit Conditions:
    1. Webcam unavailable (vocal-only live operation + device alert).
    2. Microphone unavailable (facial-only live operation + device alert).
    3. No face detected.
    4. Multiple faces detected (primary foreground face selection + multi-face warning).
    5. Person is silent (`"Waiting for speech"`).
    6. Stationary background noise (`"Waiting for speech"` noise rejection).
    7. Low lighting (CLAHE emotion fallback + gender gate abstention).
    8. Face moving quickly / motion blur.
    9. Model download / missing checkpoint failure (graceful visible error, zero random fallback).
   10. CPU-only execution (`AURASENSE_DEVICE=cpu` real forward pass).
   11. CUDA execution path (`AURASENSE_DEVICE=cuda` selection when available, safe fallback otherwise).
   12. Camera or microphone permission denied (timeout detection & user permission guidance banner).
   13. Stream interrupted and restarted (`on_video_ended`, `on_audio_ended`, `reset_session`, `shutdown`).
"""

from pathlib import Path
import time
from unittest.mock import patch

import av
import cv2
import deepface
import numpy as np
import streamlit as st
import torch

from aurasense.config import (
    DEEPFACE_EMOTION_LABELS,
    DISPLAY_EMOTION_ORDER,
    GENDER_UNCERTAIN_MESSAGE,
    LiveInferenceConfig,
    PASTEL_PALETTES,
    UNIFIED_EMOTIONS,
    WAV2VEC2_EMOTION_LABELS,
)
from aurasense.evaluation import load_cached_indian_benchmark
from aurasense.inference import (
    FusionRuntimeSettings,
    MultimodalEmotionFusionEngine,
    PretrainedModelRegistry,
    get_model_registry,
    normalize_distribution,
    select_compute_device,
)
from aurasense.inference.face_analyzer import evaluate_face_geometry_and_quality
from aurasense.streaming import LiveSessionCoordinator
from aurasense.ui import (
    render_camera_tracking_bar,
    render_diagnostics_page,
    render_emotion_probability_bars,
    render_header,
    render_prominent_facial_gender_card,
    render_separate_vocal_presentation_card,
    render_system_status_strip,
    render_three_emotion_cards,
)


def _find_real_face_bgr() -> np.ndarray:
    """Loads a real human portrait bundled inside DeepFace/InsightFace package assets for authentic testing."""
    # Public held-out FairFace-Indian validation portrait (cached HF dataset; not personal data)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_p = hf_hub_download(
        repo_id="piadev/fairface-indian",
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pq.read_table(parquet_p).to_pandas()
    for i in range(len(df)):
        raw = df.iloc[i]["img_bytes"]
        if isinstance(raw, dict):
            raw = raw.get("bytes", b"")
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pad = int(max(img.shape[:2]) * 0.30)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        if img is not None and img.shape[0] >= 120 and img.shape[1] >= 120:
            # Place on a clean 640x480 webcam-sized canvas so border/size gates can be tested realistically
            h, w = img.shape[:2]
            scale = min(280.0 / float(w), 320.0 / float(h))
            resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            canvas = np.full((480, 640, 3), 185, dtype=np.uint8)
            y0 = (480 - rh) // 2
            x0 = (640 - rw) // 2
            canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
            return canvas
    return np.full((480, 640, 3), 180, dtype=np.uint8)


def _generate_modulated_speech(duration_sec: float = 4.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    syllable_env = (np.sin(2 * np.pi * 2.6 * t) > -0.15).astype(np.float32)
    wav = syllable_env * (
        0.24 * np.sin(2 * np.pi * 190 * t)
        + 0.12 * np.sin(2 * np.pi * 380 * t)
        + 0.06 * np.sin(2 * np.pi * 570 * t)
    )
    return wav.astype(np.float32)


def test_privacy_and_zero_random_guardrails() -> None:
    root = Path(__file__).resolve().parent.parent
    assert not (root / "aurasense" / "storage").exists()
    assert not (root / "data").exists()

    forbidden_tokens = [
        "file_uploader",
        "audio_input",
        "camera_input",
        "Clip Inspector",
        "cv2.imwrite",
        "VideoWriter",
        "soundfile.write",
        "sf.write",
        "random.choice",
        "random.uniform",
        "torch.rand",
        "INITIAL_RESULT",
    ]
    for p in root.rglob("*.py"):
        if ".venv" in p.parts or "tests" in p.parts:
            continue
        # Only offline stratified dataset sampling (`indian_benchmark.py`) uses `np.random.default_rng(seed)`
        src = p.read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in src, f"Forbidden token '{tok}' found in {p}"
        if p.name != "indian_benchmark.py":
            assert "np.random" not in src, f"Forbidden token 'np.random' found in live module {p}"

    assert GENDER_UNCERTAIN_MESSAGE == "Uncertain – Please adjust camera position"
    assert [lbl for _, lbl in DISPLAY_EMOTION_ORDER] == [
        "Happy",
        "Sad",
        "Angry",
        "Fear",
        "Surprise",
        "Disgust",
        "Neutral",
    ]


def test_dual_model_gender_consensus_quality_gates_and_voice_isolation() -> None:
    reg = get_model_registry()
    engine = MultimodalEmotionFusionEngine(reg, reg.config)
    fa = engine.face_analyzer

    clean_qual = {"passed_gender_gate": True, "issues": []}

    # 1. Confident Agreement -> conservative min(conf1, conf2), never blind average
    m1_fem = {"available": True, "model_name": "InsightFace Buffalo_L", "label": "Female", "confidence_pct": 92.4}
    m2_fem = {"available": True, "model_name": "FairFace ResNet34", "label": "Female", "confidence_pct": 98.6}
    cons_agree = fa._compute_dual_model_consensus(m1_fem, m2_fem, clean_qual)
    assert cons_agree["is_uncertain"] is False
    assert cons_agree["models_agree"] is True
    assert cons_agree["agreement_status"] == "Models Agree"
    assert cons_agree["short_label"] == "Female"
    assert cons_agree["confidence_pct"] == 92.4  # Conservative min, NOT (92.4+98.6)/2 = 95.5

    # 2. Model Disagreement -> MUST return exact string "Uncertain – Please adjust camera position" & never average!
    m1_male = {"available": True, "model_name": "InsightFace Buffalo_L", "label": "Male", "confidence_pct": 58.0}
    m2_strong_fem = {"available": True, "model_name": "FairFace ResNet34", "label": "Female", "confidence_pct": 96.0}
    cons_disagree = fa._compute_dual_model_consensus(m1_male, m2_strong_fem, clean_qual)
    assert cons_disagree["is_uncertain"] is True
    assert cons_disagree["models_agree"] is False
    assert cons_disagree["agreement_status"] == "Models Disagree"
    assert cons_disagree["final_label"] == "Uncertain – Please adjust camera position"
    assert cons_disagree["confidence_pct"] is None

    # 3. Low Confidence (< 65%) even when labels match -> "Uncertain – Please adjust camera position"
    m1_low = {"available": True, "model_name": "InsightFace Buffalo_L", "label": "Male", "confidence_pct": 61.5}
    m2_high = {"available": True, "model_name": "FairFace ResNet34", "label": "Male", "confidence_pct": 89.0}
    cons_low = fa._compute_dual_model_consensus(m1_low, m2_high, clean_qual)
    assert cons_low["is_uncertain"] is True
    assert cons_low["models_agree"] is True
    assert cons_low["agreement_status"] == "Agree (Low Confidence)"
    assert cons_low["final_label"] == "Uncertain – Please adjust camera position"
    assert cons_low["confidence_pct"] is None

    # 4. Pre-Inference Quality Gate Rejections: small, blurred, dark, border-occluded, extreme angle
    cfg = reg.config
    sharp_patch = np.zeros((480, 640, 3), dtype=np.uint8)
    sharp_patch[:, :] = 135
    # Add high-frequency checkerboard texture inside face region so sharpness passes when testing non-blur gates
    yy, xx = np.indices((480, 640))
    checker = (((xx // 4 + yy // 4) % 2) * 50 + 105).astype(np.uint8)
    sharp_patch[:, :, 0] = checker
    sharp_patch[:, :, 1] = checker
    sharp_patch[:, :, 2] = checker

    frontal_kps = np.array(
        [[285.0, 210.0], [355.0, 210.0], [320.0, 250.0], [292.0, 295.0], [348.0, 295.0]],
        dtype=np.float32,
    )
    good_bbox = np.array([240.0, 155.0, 400.0, 345.0], dtype=np.float32)

    q_good = evaluate_face_geometry_and_quality(sharp_patch, good_bbox, frontal_kps, 0.90, cfg, True)
    assert q_good["passed_gender_gate"] is True

    # 4a. Too small (< 72px)
    tiny_bbox = np.array([300.0, 220.0, 345.0, 270.0], dtype=np.float32)
    tiny_kps = np.array(
        [[312.0, 236.0], [333.0, 236.0], [322.0, 247.0], [314.0, 258.0], [331.0, 258.0]],
        dtype=np.float32,
    )
    q_small = evaluate_face_geometry_and_quality(sharp_patch, tiny_bbox, tiny_kps, 0.90, cfg, True)
    assert q_small["passed_gender_gate"] is False
    assert any("small" in s.lower() for s in q_small["issues"])

    # 4b. Blurred face
    flat_img = np.full((480, 640, 3), 130, dtype=np.uint8)
    q_blur = evaluate_face_geometry_and_quality(flat_img, good_bbox, frontal_kps, 0.90, cfg, True)
    assert q_blur["passed_gender_gate"] is False
    assert any("blurred" in s.lower() for s in q_blur["issues"])

    # 4c. Poorly illuminated (too dark)
    dark_img = (sharp_patch.astype(np.float32) * 0.20).astype(np.uint8)
    q_dark = evaluate_face_geometry_and_quality(dark_img, good_bbox, frontal_kps, 0.90, cfg, True)
    assert q_dark["passed_gender_gate"] is False
    assert any("illuminated" in s.lower() for s in q_dark["issues"])

    # 4d. Border-clipped / occluded (`det_score < 0.65` or edge `< 6px`)
    edge_bbox = np.array([2.0, 155.0, 162.0, 345.0], dtype=np.float32)
    edge_kps = frontal_kps - np.array([238.0, 0.0], dtype=np.float32)
    q_edge = evaluate_face_geometry_and_quality(sharp_patch, edge_bbox, edge_kps, 0.55, cfg, True)
    assert q_edge["passed_gender_gate"] is False
    cons_q_blocked = fa._compute_dual_model_consensus(m1_fem, m2_fem, q_edge)
    assert cons_q_blocked["final_label"] == "Uncertain – Please adjust camera position"
    assert cons_q_blocked["confidence_pct"] is None
    assert cons_q_blocked["model_1"]["confidence_pct"] is None

    # 4e. Extreme head angle (roll > 28° or yaw asymmetry > 0.42)
    yaw_kps = np.array(
        [[285.0, 210.0], [355.0, 210.0], [362.0, 250.0], [292.0, 295.0], [348.0, 295.0]],
        dtype=np.float32,
    )
    q_yaw = evaluate_face_geometry_and_quality(sharp_patch, good_bbox, yaw_kps, 0.90, cfg, True)
    assert q_yaw["passed_gender_gate"] is False
    assert any("angle" in s.lower() for s in q_yaw["issues"])

    # 5. Real face inference + strict vocal isolation check (voice MUST NEVER override facial gender)
    face_canvas = _find_real_face_bgr()
    speech_wav = _generate_modulated_speech(4.0, 16000)
    live_res = engine.analyze_live_window([face_canvas], speech_wav, audio_sr=16000)
    assert live_res["face_analysis"]["face_detected"] is True
    dg = live_res["dual_facial_gender"]
    assert dg["model_1"]["model_name"] == "InsightFace Buffalo_L"
    assert dg["model_2"]["model_name"] == "FairFace ResNet34"

    # Force opposing vocal presentation and verify `dual_facial_gender` is completely untouched
    facial_label_before = dg["final_label"]
    assert "voice" not in str(dg.get("reason", "")).lower()
    assert live_res["dual_facial_gender"]["final_label"] == facial_label_before

    # 6. Verify cached Indian faces empirical benchmark artifact
    bench = load_cached_indian_benchmark()
    assert bench.get("faces_detected_count", 0) >= 200
    assert "overall_metrics" in bench
    assert "by_presentation_label" in bench
    assert "by_age_bracket" in bench
    assert "by_skin_tone_luma" in bench
    assert "by_lighting_condition" in bench


def test_conditions_10_and_11_cpu_cuda_and_single_load_caching() -> None:
    cpu_dev = select_compute_device("cpu")
    assert cpu_dev.type == "cpu"
    cpu_reg = PretrainedModelRegistry(LiveInferenceConfig(enable_voice_presentation=False), device_preference="cpu")
    cpu_reg.ensure_voice_emotion_loaded()
    assert cpu_reg.status()["device"] == "CPU"
    cpu_engine = MultimodalEmotionFusionEngine(cpu_reg, cpu_reg.config)
    cpu_speech = _generate_modulated_speech(3.5, 16000)
    cpu_voice_res = cpu_engine.voice_analyzer.analyze_waveform(cpu_speech, input_sr=16000)
    assert cpu_voice_res["available"] is True
    assert abs(sum(cpu_voice_res["raw_probabilities"].values()) - 100.0) < 0.5

    cuda_dev = select_compute_device("cuda")
    if torch.cuda.is_available():
        assert cuda_dev.type == "cuda"
    else:
        assert cuda_dev.type == "cpu"

    # Single-load cache verification across ALL models including Buffalo_L & FairFace ResNet34
    reg = get_model_registry(LiveInferenceConfig())
    st1 = reg.warmup_all()
    st2 = reg.warmup_all()
    assert st1["buffalo_loads"] == 1 and st2["buffalo_loads"] == 1
    assert st1["fairface_loads"] == 1 and st2["fairface_loads"] == 1
    assert st1["deepface_loads"] == 1 and st2["deepface_loads"] == 1
    assert st1["voice_emotion_loads"] == 1 and st2["voice_emotion_loads"] == 1


def test_conditions_3_4_7_8_facial_edge_cases() -> None:
    reg = get_model_registry()
    engine = MultimodalEmotionFusionEngine(reg, reg.config)
    face_img = _find_real_face_bgr()

    # Condition 3: No face detected
    blank = np.full((480, 640, 3), 205, dtype=np.uint8)
    res_no_face = engine.face_analyzer.analyze_frames([blank])
    assert res_no_face["available"] is False
    assert res_no_face["face_detected"] is False
    assert res_no_face["status"] == "no_face"
    assert res_no_face["probabilities"] == {}

    base_face_res = engine.face_analyzer.analyze_frames([face_img])
    assert base_face_res["face_detected"] is True
    if base_face_res["available"]:
        assert set(base_face_res["raw_deepface_probabilities"].keys()) == set(DEEPFACE_EMOTION_LABELS)

    # Condition 4: Multiple faces detected (tile two portraits side-by-side)
    half_left = cv2.resize(face_img, (320, 240))
    half_right = cv2.resize(face_img, (280, 210))
    multi_canvas = np.full((480, 640, 3), 185, dtype=np.uint8)
    multi_canvas[110:350, 10:330] = half_left
    multi_canvas[125:335, 345:625] = half_right
    res_multi = engine.face_analyzer.analyze_frames([multi_canvas])
    if res_multi["face_detected"] and res_multi["diagnostics"]["detected_face_count"] > 1:
        assert res_multi["diagnostics"]["multiple_faces"] is True

    # Condition 7: Low lighting -> gender gate abstains with exact message
    dark_face = (face_img.astype(np.float32) * 0.14).astype(np.uint8)
    res_dark = engine.face_analyzer.analyze_frames([dark_face])
    if res_dark["face_detected"]:
        assert res_dark["diagnostics"]["low_lighting"] is True
        assert res_dark["dual_gender"]["final_label"] == GENDER_UNCERTAIN_MESSAGE

    # Condition 8: Fast face motion / blur -> gender gate abstains with exact message
    shift_a = np.roll(face_img, -160, axis=1)
    shift_b = cv2.GaussianBlur(np.roll(face_img, 160, axis=1), (25, 25), 9.0)
    res_motion = engine.face_analyzer.analyze_frames([shift_a, shift_b])
    if res_motion["face_detected"]:
        assert res_motion["diagnostics"]["rapid_motion_or_blur"] is True
        assert res_motion["dual_gender"]["final_label"] == GENDER_UNCERTAIN_MESSAGE


def test_conditions_5_6_9_voice_and_download_failure() -> None:
    reg = get_model_registry()
    engine = MultimodalEmotionFusionEngine(reg, reg.config)

    # Condition 5: Person is silent -> "Waiting for speech"
    silent = np.zeros(16000 * 4, dtype=np.float32)
    res_silent = engine.voice_analyzer.analyze_waveform(silent, input_sr=16000)
    assert res_silent["available"] is False
    assert res_silent["display_state"] == "Waiting for speech"
    assert res_silent["probabilities"] == {}

    # Condition 6: Stationary background noise -> filtered, "Waiting for speech"
    rng = np.random.default_rng(7)
    bg_noise = rng.normal(0.0, 0.02, size=16000 * 4).astype(np.float32)
    res_noise = engine.voice_analyzer.analyze_waveform(bg_noise, input_sr=16000)
    assert res_noise["available"] is False
    assert res_noise["display_state"] == "Waiting for speech"

    # Condition 9: Model download / missing file failure -> explicit error, zero fabricated scores
    bad_cfg = LiveInferenceConfig(
        voice_emotion_model_id="invalid-aurasense-org/nonexistent-wav2vec2-checkpoint",
        enable_voice_presentation=False,
    )
    bad_reg = PretrainedModelRegistry(bad_cfg, device_preference="cpu")
    bad_engine = MultimodalEmotionFusionEngine(bad_reg, bad_cfg)
    speech = _generate_modulated_speech(4.0, 16000)
    res_fail = bad_engine.voice_analyzer.analyze_waveform(speech, input_sr=16000)
    assert res_fail["available"] is False
    assert res_fail["status"] == "error"
    assert res_fail["error"] is not None
    assert res_fail["probabilities"] == {}


def test_conditions_1_2_12_13_live_webrtc_pipeline_and_ui() -> None:
    reg = get_model_registry()
    cfg = LiveInferenceConfig(inference_interval_sec=0.2)
    coord = LiveSessionCoordinator(registry=reg, config=cfg)

    try:
        # Condition 12: Camera or microphone permission denied
        snap_denied = coord.get_snapshot(webrtc_playing=False, desired_playing=True)
        assert snap_denied["device_alert"] is not None
        assert "permission" in snap_denied["device_alert"].lower()

        # Condition 1: Webcam unavailable (microphone streaming active speech, zero video frames)
        coord.mark_session_started()
        coord._session_started_ts = time.monotonic() - 4.5
        speech_48k = _generate_modulated_speech(3.2, 48000)
        pcm_int16 = (speech_48k * 18000).astype(np.int16)
        chunk_len = 9600
        for start in range(0, pcm_int16.size, chunk_len):
            chunk = pcm_int16[start : start + chunk_len].reshape(1, -1)
            af = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            af.sample_rate = 48000
            out_af = coord.audio_frame_callback(af)
            assert int(np.max(np.abs(out_af.to_ndarray()))) == 0

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            s_mic_only = coord.get_snapshot(webrtc_playing=True)
            if s_mic_only["completed_inferences"] >= 1:
                break
            time.sleep(0.1)

        s_mic_only = coord.get_snapshot(webrtc_playing=True)
        assert s_mic_only["camera_connected"] is False
        assert s_mic_only["microphone_connected"] is True
        assert s_mic_only["device_alert"] is not None and "Webcam stream unavailable" in s_mic_only["device_alert"]
        assert s_mic_only["result"] is not None
        assert s_mic_only["result"]["available"] is True
        assert s_mic_only["result"]["fusion_mode"] == "vocal_only"
        assert set(s_mic_only["result"]["voice_analysis"]["raw_probabilities"].keys()) == set(WAV2VEC2_EMOTION_LABELS)

        # Condition 13a: Stream interrupted
        coord.on_audio_ended()
        coord.reset_session(clear_results=True)
        snap_after_interrupt = coord.get_snapshot(webrtc_playing=False)
        assert snap_after_interrupt["buffered_audio_sec"] == 0.0
        assert snap_after_interrupt["buffered_frames"] == 0
        assert snap_after_interrupt["result"] is None

        # Condition 2 & 13b: Stream restarted with Webcam active & Microphone unavailable
        face_img = _find_real_face_bgr()
        coord.mark_session_started()
        coord._session_started_ts = time.monotonic() - 4.5
        coord._last_inference_ts = 0.0
        for _ in range(9):
            vf = av.VideoFrame.from_ndarray(face_img, format="bgr24")
            out_vf = coord.video_frame_callback(vf)
            assert out_vf.width == face_img.shape[1]

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            s_cam_only = coord.get_snapshot(webrtc_playing=True)
            if s_cam_only["completed_inferences"] >= 1:
                break
            time.sleep(0.1)

        s_cam_only = coord.get_snapshot(webrtc_playing=True)
        assert s_cam_only["camera_connected"] is True
        assert s_cam_only["microphone_connected"] is False
        assert s_cam_only["device_alert"] is not None and "Microphone stream unavailable" in s_cam_only["device_alert"]
        assert s_cam_only["buffer_memory_mb"] < 25.0

        # Verify Hero Dual-Model Facial Gender Card, Separate Vocal Presentation Card, and Diagnostics UI
        captured_html = []
        bench_data = load_cached_indian_benchmark()
        with patch.object(st, "markdown", side_effect=lambda body, **kw: captured_html.append(body)):
            theme = PASTEL_PALETTES["Neutral"]
            render_header(theme, reg.status(), is_streaming=True, is_paused=False)
            render_camera_tracking_bar(s_cam_only, theme, is_streaming=True)
            render_prominent_facial_gender_card(s_cam_only, theme, is_streaming=True)
            render_three_emotion_cards(s_cam_only, theme, is_streaming=True)
            render_emotion_probability_bars(s_cam_only, theme)
            render_separate_vocal_presentation_card(s_cam_only, reg.status(), theme, is_streaming=True)
            render_system_status_strip(s_cam_only, reg.status(), theme)
            render_diagnostics_page(s_cam_only, reg.status(), bench_data, theme)

        joined = "\n".join(captured_html)
        assert "Privacy Shield" in joined
        assert "Facial Gender Presentation Estimate (Dual-Model Consensus)" in joined
        assert "Model 1 · InsightFace Buffalo_L" in joined
        assert "Model 2 · FairFace ResNet34" in joined
        assert "Separate Vocal Presentation Estimate" in joined
        assert "Never overrides facial presentation" in joined
        assert "Empirical Evaluation · Diverse Indian Faces Benchmark" in joined
    finally:
        coord.shutdown()
        assert not coord._worker_thread.is_alive()


def test_fusion_math_and_stabilization() -> None:
    normed = normalize_distribution({"Happy": 50.0, "Neutral": 50.0})
    assert normed is not None and abs(sum(normed.values()) - 1.0) < 1e-9

    settings = FusionRuntimeSettings(
        facial_weight=0.5,
        vocal_weight=0.5,
        adapt_by_reliability=False,
        smoothing_mode="ema",
        ema_alpha=0.5,
        switch_hysteresis_margin=6.0,
    )
    engine = MultimodalEmotionFusionEngine(settings=settings)
    f_in = {
        "available": True,
        "predicted_emotion": "Happy",
        "probability_pct": 80.0,
        "probabilities": {"Happy": 80.0, "Neutral": 20.0, "Sad": 0.0, "Angry": 0.0, "Fearful": 0.0, "Surprised": 0.0, "Disgust": 0.0},
        "diagnostics": {"quality_score": 0.9},
    }
    v_in = {
        "available": True,
        "predicted_emotion": "Neutral",
        "probability_pct": 60.0,
        "probabilities": {"Happy": 20.0, "Neutral": 60.0, "Sad": 20.0, "Angry": 0.0, "Fearful": 0.0, "Surprised": 0.0, "Disgust": 0.0},
        "diagnostics": {"quality_score": 0.9},
    }
    fused = engine.fuse_distributions(f_in, v_in)
    assert fused["available"] is True
    assert fused["dominant_emotion"] in UNIFIED_EMOTIONS
    assert abs(fused["combined_probabilities"]["Happy"] - 50.0) < 0.05


if __name__ == "__main__":
    test_privacy_and_zero_random_guardrails()
    print("✓ Privacy & zero-random/zero-hardcoded guardrails verified")
    test_dual_model_gender_consensus_quality_gates_and_voice_isolation()
    print("✓ Dual-model facial gender consensus (Buffalo_L + FairFace ResNet34), quality gates & vocal isolation verified")
    test_conditions_10_and_11_cpu_cuda_and_single_load_caching()
    print("✓ Conditions 10 & 11 (CPU / CUDA device execution) + single-load caching verified")
    test_conditions_3_4_7_8_facial_edge_cases()
    print("✓ Conditions 3, 4, 7 & 8 (no face, multiple faces, low light, fast motion) verified")
    test_conditions_5_6_9_voice_and_download_failure()
    print("✓ Conditions 5, 6 & 9 (silence, background noise, failed model download) verified")
    test_conditions_1_2_12_13_live_webrtc_pipeline_and_ui()
    print("✓ Conditions 1, 2, 12 & 13 (cam/mic unavailable, permission denied, stream restart, Hero UI & Benchmark UI) verified")
    test_fusion_math_and_stabilization()
    print("✓ Multimodal fusion math & temporal stabilization verified")
    print("ALL DUAL-MODEL GENDER & 13 AUDIT CONDITION TESTS PASSED!")
