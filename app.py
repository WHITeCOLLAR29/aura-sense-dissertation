"""
AuraSense — Real-Time Dual-Model Facial Gender Presentation & Multimodal Emotion Dashboard.

Highest-Priority Feature:
- Dual-Model Facial Gender Presentation Estimation running on the exact same 5-point aligned face:
  * Model 1: InsightFace `buffalo_l` (`SCRFD 10G` + `genderage.onnx`)
  * Model 2: FairFace `ResNet34` (`res34_fair_align_multi_7_20190809.pt`)
  * Independent predictions & confidences + Agreement/Disagreement indicator.
  * Never blindly averages models; displays `"Uncertain – Please adjust camera position"` on
    disagreement, low confidence, or suboptimal face size, blur, illumination, occlusion, or angle.
  * Vocal gender presentation is kept strictly separate and never overrides facial predictions.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, get_cloudflare_ice_servers, webrtc_streamer

from aurasense.config import PASTEL_PALETTES
from aurasense.evaluation import load_cached_indian_benchmark
from aurasense.inference import PretrainedModelRegistry, analyze_uploaded_clip, get_cached_model_registry
from aurasense.streaming import LiveSessionCoordinator
from aurasense.ui import (
    inject_base_css,
    inject_dynamic_palette_tokens,
    render_camera_tracking_bar,
    render_clip_inspector_results,
    render_diagnostics_page,
    render_emotion_probability_bars,
    render_header,
    render_html,
    render_prominent_facial_gender_card,
    render_separate_vocal_presentation_card,
    render_system_status_strip,
    render_three_emotion_cards,
)

st.set_page_config(
    page_title="AuraSense — Real-Time Multimodal Emotion & Presentation Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cloudflare_ice(key_id: str, api_token: str):
    return get_cloudflare_ice_servers(turn_key_id=key_id, turn_key_api_token=api_token)


def _build_rtc_configuration() -> tuple[RTCConfiguration, str]:
    """Builds ICE configuration using Cloudflare TURN (from `st.secrets`) + STUN fallback."""
    try:
        cf_id = st.secrets.get("CLOUDFLARE_TURN_KEY_ID")
        cf_tok = st.secrets.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
        if cf_id and cf_tok:
            ice_servers = _fetch_cloudflare_ice(str(cf_id), str(cf_tok))
            return RTCConfiguration({"iceServers": ice_servers}), "Cloudflare TURN + STUN"
    except Exception as exc:
        st.session_state["webrtc_turn_error"] = str(exc)

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    mode_label = "STUN Only (No TURN configured)"
    try:
        turn_url = st.secrets.get("TURN_SERVER_URL")
        turn_user = st.secrets.get("TURN_USERNAME")
        turn_cred = st.secrets.get("TURN_CREDENTIAL")
        if turn_url:
            entry = {"urls": [str(turn_url)]}
            if turn_user:
                entry["username"] = str(turn_user)
            if turn_cred:
                entry["credential"] = str(turn_cred)
            ice_servers.append(entry)
            mode_label = "Custom TURN + STUN"
    except Exception:
        pass
    return RTCConfiguration({"iceServers": ice_servers}), mode_label


def get_session_coordinator(registry: PretrainedModelRegistry) -> LiveSessionCoordinator:
    if "live_coordinator" not in st.session_state:
        st.session_state.live_coordinator = LiveSessionCoordinator(
            registry=registry,
            config=registry.config,
        )
    return st.session_state.live_coordinator


def main() -> None:
    inject_base_css()

    registry = get_cached_model_registry()
    coordinator = get_session_coordinator(registry)
    f_settings = coordinator.fusion_settings

    if "desired_playing" not in st.session_state:
        st.session_state.desired_playing = None

    is_currently_playing = bool(st.session_state.get("webrtc_playing", False))
    initial_snap = coordinator.get_snapshot(
        webrtc_playing=is_currently_playing,
        desired_playing=st.session_state.desired_playing,
    )
    init_theme_key = (initial_snap.get("result") or {}).get("theme_key", "Neutral")
    init_theme = PASTEL_PALETTES.get(init_theme_key, PASTEL_PALETTES["Neutral"])

    # 1. Header + Privacy Shield
    render_header(
        theme=init_theme,
        model_status=registry.status(),
        is_streaming=is_currently_playing,
        is_paused=coordinator.is_paused,
    )

    nav_col, _ = st.columns([6.5, 5.5])
    with nav_col:
        active_view = st.radio(
            label="AuraSense View",
            options=["📷 Live Analysis", "🎞️ Clip Inspector", "🩺 Diagnostics & Indian Benchmark"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if active_view == "🎞️ Clip Inspector":
        inject_dynamic_palette_tokens(init_theme)
        st.markdown("### 🎞️ Clip Inspector")
        st.caption("Upload an MP4, MOV, or WEBM clip to analyze facial presentation, facial emotion, vocal emotion, and multimodal fusion across sampled segments.")
        c1, c2, c3 = st.columns([2.2, 1.0, 1.0])
        with c1:
            uploaded_clip = st.file_uploader(
                "Upload Video Clip (MP4, MOV, WEBM)",
                type=["mp4", "mov", "webm"],
                accept_multiple_files=False,
            )
        with c2:
            clip_interval = st.slider("Sample Interval (sec)", min_value=1.0, max_value=3.0, value=1.5, step=0.5)
        with c3:
            clip_max_dur = st.slider("Max Duration (sec)", min_value=10, max_value=90, value=45, step=5)

        if uploaded_clip is not None:
            if st.button("🔍 Analyze Uploaded Clip", type="primary"):
                with st.spinner("Analyzing sampled video frames and 16 kHz audio segments..."):
                    clip_res = analyze_uploaded_clip(
                        file_bytes=uploaded_clip.getvalue(),
                        filename=uploaded_clip.name,
                        registry=registry,
                        config=registry.config,
                        sample_interval_sec=float(clip_interval),
                        max_duration_sec=float(clip_max_dur),
                    )
                render_clip_inspector_results(clip_res, init_theme)
        return

    if active_view == "🩺 Diagnostics & Indian Benchmark":
        benchmark_data = load_cached_indian_benchmark()

        @st.fragment(run_every=1.4)
        def diagnostics_fragment() -> None:
            snap = coordinator.get_snapshot(
                webrtc_playing=bool(st.session_state.get("webrtc_playing", False)),
                desired_playing=st.session_state.desired_playing,
            )
            cur_theme_key = (snap.get("result") or {}).get("theme_key", "Neutral")
            cur_theme = PASTEL_PALETTES.get(cur_theme_key, PASTEL_PALETTES["Neutral"])
            inject_dynamic_palette_tokens(cur_theme)
            render_diagnostics_page(
                snapshot=snap,
                model_status=registry.status(),
                benchmark_data=benchmark_data,
                theme=cur_theme,
            )

        diagnostics_fragment()
        return

    # Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns(
        [1.25, 1.15, 1.25, 1.2, 1.45], gap="small"
    )

    with ctrl_col1:
        if st.button(
            "▶️ Start Analysis",
            type="primary",
            disabled=is_currently_playing and not coordinator.is_paused,
            use_container_width=True,
        ):
            coordinator.set_paused(False)
            coordinator.mark_session_started()
            st.session_state.desired_playing = True
            st.rerun()

    with ctrl_col2:
        pause_label = "▶️ Resume" if coordinator.is_paused else "⏸️ Pause"
        if st.button(
            pause_label,
            disabled=not is_currently_playing,
            use_container_width=True,
        ):
            coordinator.set_paused(not coordinator.is_paused)
            st.rerun()

    with ctrl_col3:
        if st.button(
            "⏹️ Stop Analysis",
            disabled=not is_currently_playing,
            use_container_width=True,
        ):
            coordinator.set_paused(False)
            coordinator.reset_session(clear_results=False)
            st.session_state.desired_playing = False
            st.rerun()

    with ctrl_col4:
        if st.button(
            "🔄 Reset Session",
            use_container_width=True,
        ):
            coordinator.set_paused(False)
            coordinator.reset_session(clear_results=True)
            st.rerun()

    with ctrl_col5:
        with st.popover("⚖️ Fusion & Cadence", use_container_width=True):
            st.markdown("**Emotion Fusion & Sampling Settings**")
            coordinator.inference_interval_sec = st.slider(
                "Inference Interval (seconds)",
                min_value=1.0,
                max_value=4.5,
                value=float(coordinator.inference_interval_sec),
                step=0.2,
            )
            facial_w = st.slider(
                "Base Facial Emotion Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(f_settings.facial_weight),
                step=0.05,
            )
            f_settings.facial_weight = facial_w
            f_settings.vocal_weight = round(1.0 - facial_w, 2)
            st.caption(
                f"Emotion Formula: **{f_settings.facial_weight:.2f} × Facial + {f_settings.vocal_weight:.2f} × Vocal**"
            )
            f_settings.adapt_by_reliability = st.toggle(
                "Modulate Emotion Weights by Signal Quality",
                value=f_settings.adapt_by_reliability,
            )
            thresh = st.slider(
                "Minimum Emotion Score Threshold (%)",
                min_value=15.0,
                max_value=55.0,
                value=float(f_settings.min_modality_score_threshold),
                step=1.0,
            )
            f_settings.min_modality_score_threshold = thresh
            f_settings.min_combined_score_threshold = thresh

            smooth_choice = st.selectbox(
                "Emotion Temporal Smoothing",
                options=["ema", "rolling", "none"],
                index=["ema", "rolling", "none"].index(f_settings.smoothing_mode),
                format_func=lambda m: {
                    "ema": "Exponential Moving Average (EMA)",
                    "rolling": "Rolling Window Average",
                    "none": "None (Single Window)",
                }[m],
            )
            f_settings.smoothing_mode = smooth_choice

    left_stage, right_stage = st.columns([6.0, 6.0], gap="large")

    with left_stage:
        render_html(
            """
            <div class="aura-surface-card" style="padding:0.95rem 1.3rem 0.65rem 1.3rem; margin-bottom:0.55rem;">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:0.5rem;">
                    <span style="font-size:0.88rem; font-weight:800; color:#2B2836;">
                        📷 Live Camera &amp; Microphone Preview
                    </span>
                    <span class="aura-pill-tag">5-pt Aligned SCRFD + WebRTC</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rtc_cfg, ice_mode_label = _build_rtc_configuration()
        st.session_state["webrtc_ice_mode"] = ice_mode_label
        webrtc_ctx = webrtc_streamer(
            key="aurasense-live-session",
            mode=WebRtcMode.SENDRECV,
            desired_playing_state=st.session_state.desired_playing,
            rtc_configuration=rtc_cfg,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 24},
                },
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                },
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
                "style": {"width": "100%", "borderRadius": "22px"},
            },
            translations={
                "start": "Start Analysis",
                "stop": "Stop Analysis",
                "select_device": "Select Camera / Mic",
                "device_ask_permission": "Allow camera & microphone permission in browser",
                "device_not_available": "Camera or microphone hardware unavailable",
                "device_access_denied": "Camera or microphone permission denied",
            },
            video_frame_callback=coordinator.video_frame_callback,
            audio_frame_callback=coordinator.audio_frame_callback,
            on_video_ended=coordinator.on_video_ended,
            on_audio_ended=coordinator.on_audio_ended,
            async_processing=True,
        )

        playing_now = bool(webrtc_ctx.state.playing)
        if st.session_state.get("webrtc_playing") and not playing_now:
            coordinator.set_paused(False)
            coordinator.reset_session(clear_results=False)
            st.session_state.desired_playing = None
        elif playing_now:
            coordinator.mark_session_started()
            st.session_state.desired_playing = True
        st.session_state["webrtc_playing"] = playing_now

        cam_opt1, cam_opt2 = st.columns(2)
        with cam_opt1:
            coordinator.show_face_overlay = st.toggle(
                "Face Detection Brackets",
                value=coordinator.show_face_overlay,
            )
        with cam_opt2:
            coordinator.mirror_video = st.toggle(
                "Mirror Webcam View",
                value=coordinator.mirror_video,
            )

        @st.fragment(run_every=0.9)
        def live_camera_status_fragment() -> None:
            snap = coordinator.get_snapshot(
                webrtc_playing=playing_now,
                desired_playing=st.session_state.desired_playing,
            )
            cur_theme_key = (snap.get("result") or {}).get("theme_key", "Neutral")
            cur_theme = PASTEL_PALETTES.get(cur_theme_key, PASTEL_PALETTES["Neutral"])
            render_camera_tracking_bar(
                snapshot=snap,
                theme=cur_theme,
                is_streaming=bool(playing_now or snap.get("camera_connected") or snap.get("microphone_connected")),
            )

        live_camera_status_fragment()

    with right_stage:
        @st.fragment(run_every=1.0)
        def live_analysis_fragment() -> None:
            snap = coordinator.get_snapshot(
                webrtc_playing=playing_now,
                desired_playing=st.session_state.desired_playing,
            )
            cur_theme_key = (snap.get("result") or {}).get("theme_key", "Neutral")
            cur_theme = PASTEL_PALETTES.get(cur_theme_key, PASTEL_PALETTES["Neutral"])

            inject_dynamic_palette_tokens(cur_theme)

            live_active = bool(playing_now or snap.get("camera_connected") or snap.get("microphone_connected"))
            # TOP PROMINENT HERO: Dual-Model Facial Gender Presentation (Buffalo_L + FairFace ResNet34)
            render_prominent_facial_gender_card(
                snapshot=snap,
                theme=cur_theme,
                is_streaming=live_active,
            )

            # Live Emotion Analysis (Facial, Vocal, Combined)
            render_three_emotion_cards(
                snapshot=snap,
                theme=cur_theme,
                is_streaming=live_active,
            )

            # 7-Emotion Score Distribution Bars
            render_emotion_probability_bars(
                snapshot=snap,
                theme=cur_theme,
            )

            # Separate Vocal Presentation Card (never overrides facial gender)
            render_separate_vocal_presentation_card(
                snapshot=snap,
                model_status=registry.status(),
                theme=cur_theme,
                is_streaming=live_active,
            )

        live_analysis_fragment()

    @st.fragment(run_every=1.1)
    def live_system_status_fragment() -> None:
        snap = coordinator.get_snapshot(
            webrtc_playing=playing_now,
            desired_playing=st.session_state.desired_playing,
        )
        cur_theme_key = (snap.get("result") or {}).get("theme_key", "Neutral")
        cur_theme = PASTEL_PALETTES.get(cur_theme_key, PASTEL_PALETTES["Neutral"])
        render_system_status_strip(
            snapshot=snap,
            model_status=registry.status(),
            theme=cur_theme,
        )

    live_system_status_fragment()


if __name__ == "__main__":
    main()
