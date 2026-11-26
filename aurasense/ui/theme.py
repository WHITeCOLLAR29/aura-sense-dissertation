"""
Material Design 3 Pastel Theme, Prominent Dual-Model Facial Gender Presentation Hero Card,
Live Multimodal Emotion Cards, Indian Faces Benchmark Explorer, and Diagnostics UI.

Key Design Guarantees:
- Facial Gender Presentation Estimate is the highest-priority, top-most right-stage hero card:
  * Shows Model 1 (`InsightFace Buffalo_L`) prediction & confidence independently.
  * Shows Model 2 (`FairFace ResNet34`) prediction & confidence independently.
  * Shows clear Agreement / Disagreement indicator (`Models Agree`, `Models Disagree`, `Quality Gate Blocked`).
  * Never averages the two models; displays `"Uncertain – Please adjust camera position"` whenever
    models disagree, confidence is insufficient, or face quality/pose/size fails.
  * Shows confidence ONLY when properly available (omitted on quality-blocked/disagreed consensus).
- Keeps Vocal Gender Presentation strictly separate in its own card and NEVER overrides facial predictions.
- Clearly states that models estimate perceived Male/Female presentation from facial appearance and
  cannot determine a person's gender identity.
"""

import html
import re
from typing import Any, Dict, List

import streamlit as st

from aurasense.config import DISPLAY_EMOTION_ORDER, GENDER_UNCERTAIN_MESSAGE, PASTEL_PALETTES


def render_html(body: str, unsafe_allow_html: bool = True) -> None:
    """
    Renders an HTML/CSS snippet via `st.markdown(..., unsafe_allow_html=True)`.
    Strips per-line indentation and blank lines first: Markdown treats lines indented by
    4+ spaces (or HTML blocks interrupted by blank lines) as code blocks, which caused raw
    HTML/CSS source to be displayed as text instead of rendered.
    """
    cleaned = "\n".join(line.strip() for line in str(body).splitlines() if line.strip())
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.S)
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_base_css() -> None:
    """Injects static Material Design 3 typography, layout, card geometry, and smooth CSS transitions."""
    render_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            --aura-bg: #F6F4FA;
            --aura-card: #ECE8F6;
            --aura-accent: #7C6FA6;
            --aura-pill: #E0D9F2;
            --aura-ink: #2B2836;
            --aura-muted: #5E5A6E;
            --aura-subtle: #8C879B;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
            color: var(--aura-ink);
        }

        .stApp {
            background: radial-gradient(circle at 10% 6%, var(--aura-pill) 0%, var(--aura-bg) 45%, #FAF8F5 100%) !important;
            transition: background 0.65s ease;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0.35rem;
        }

        .block-container {
            max-width: 1280px !important;
            padding-top: 1.05rem !important;
            padding-bottom: 2.5rem !important;
        }

        .aura-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1rem 1.4rem;
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 32px;
            border: 1px solid rgba(43, 40, 54, 0.06);
            box-shadow: 0 10px 30px rgba(43, 40, 54, 0.045);
            margin-bottom: 0.8rem;
        }

        .aura-brand {
            display: flex;
            align-items: center;
            gap: 0.95rem;
        }

        .aura-logo-badge {
            width: 52px;
            height: 52px;
            border-radius: 20px;
            background: var(--aura-pill);
            color: var(--aura-accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: inset 0 0 0 1px rgba(43, 40, 54, 0.05);
            transition: background 0.45s ease, color 0.45s ease;
        }

        .aura-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #2B2836;
            margin: 0;
            line-height: 1.12;
        }

        .aura-subtitle {
            font-size: 0.82rem;
            color: #5E5A6E;
            margin: 0.18rem 0 0 0;
            font-weight: 600;
        }

        .aura-privacy-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.44rem 0.92rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            border: 1px solid rgba(43, 40, 54, 0.08);
        }

        .aura-surface-card {
            background: rgba(255, 255, 255, 0.89);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 30px;
            padding: 1.2rem 1.38rem;
            border: 1px solid rgba(43, 40, 54, 0.06);
            box-shadow: 0 8px 26px rgba(43, 40, 54, 0.038);
            margin-bottom: 0.88rem;
        }

        .aura-hero-gender-card {
            background: linear-gradient(140deg, var(--aura-card) 0%, rgba(255, 255, 255, 0.96) 95%);
            border-radius: 32px;
            padding: 1.3rem 1.45rem;
            border: 2px solid var(--aura-pill);
            box-shadow: 0 12px 32px rgba(43, 40, 54, 0.06);
            margin-bottom: 0.9rem;
            transition: background 0.45s ease, border-color 0.45s ease;
        }

        .aura-section-label {
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #68637A;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .aura-pill-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.26rem 0.75rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            background: var(--aura-pill);
            color: var(--aura-accent);
        }

        .aura-mono {
            font-family: 'JetBrains Mono', monospace;
        }

        .aura-error-banner {
            background: #FDF4F5;
            border: 1px solid #FADCE0;
            color: #9F2B3A;
            border-radius: 22px;
            padding: 0.82rem 1.12rem;
            font-size: 0.78rem;
            margin-bottom: 0.8rem;
        }

        .aura-warn-banner {
            background: #FFFBF0;
            border: 1px solid #FFE69E;
            color: #8A5D14;
            border-radius: 22px;
            padding: 0.78rem 1.1rem;
            font-size: 0.76rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stRadio"] > div {
            background: rgba(255, 255, 255, 0.84);
            backdrop-filter: blur(12px);
            padding: 0.32rem;
            border-radius: 999px;
            border: 1px solid rgba(43, 40, 54, 0.06);
            display: inline-flex;
            gap: 0.35rem;
            margin-bottom: 0.72rem;
        }

        div[data-testid="stRadio"] label {
            border-radius: 999px !important;
            padding: 0.38rem 0.98rem !important;
            font-size: 0.81rem !important;
            font-weight: 700 !important;
        }

        iframe[title*="streamlit_webrtc"] {
            border-radius: 26px !important;
            border: 3px solid var(--aura-pill) !important;
            background: #EFECE6 !important;
            box-shadow: 0 10px 28px rgba(43, 40, 54, 0.055);
            transition: border-color 0.45s ease;
        }

        .stButton > button {
            border-radius: 999px !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            padding: 0.5rem 1.1rem !important;
            border: 1px solid rgba(43, 40, 54, 0.09) !important;
            box-shadow: 0 3px 10px rgba(43, 40, 54, 0.035) !important;
        }

        .stButton > button[kind="primary"] {
            background: var(--aura-accent) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
        }

        *, *::before, *::after { box-sizing: border-box; }
        html, body, .stApp { overflow-x: hidden !important; max-width: 100vw; }
        .block-container { width: 100% !important; max-width: min(1280px, 100%) !important;
            padding-left: clamp(0.75rem, 2.5vw, 2rem) !important; padding-right: clamp(0.75rem, 2.5vw, 2rem) !important; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.75rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="column"] { min-width: min(260px, 100%) !important; }
        .aura-header, .aura-surface-card, .aura-hero-gender-card, .aura-error-banner, .aura-warn-banner {
            max-width: 100%; min-width: 0; overflow-wrap: anywhere; }
        .aura-surface-card div, .aura-hero-gender-card div { min-width: 0; }
        .aura-surface-card table { max-width: 100%; }
        iframe[title*="streamlit_webrtc"] { max-width: 100% !important; }
        video { max-width: 100% !important; height: auto !important; }
        @media (max-width: 900px) {
            .aura-grid-2, .aura-grid-3 { grid-template-columns: minmax(0, 1fr) !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_dynamic_palette_tokens(theme: Dict[str, str]) -> None:
    render_html(
        f"""
        <style>
        :root {{
            --aura-bg: {theme['bg']};
            --aura-card: {theme['card']};
            --aura-accent: {theme['accent']};
            --aura-pill: {theme['pill']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    theme: Dict[str, str],
    model_status: Dict[str, Any],
    is_streaming: bool,
    is_paused: bool,
) -> None:
    """1. Header: AuraSense + Subtitle + Visible Camera/Microphone In-Memory Privacy Indicator."""
    device_str = html.escape(str(model_status.get("device", "CPU")))
    crit_errors = model_status.get("critical_errors") or []

    if crit_errors:
        dot_color = "#EF4444"
        badge_text = f"Model Error · {device_str}"
    elif is_streaming and is_paused:
        dot_color = "#F59E0B"
        badge_text = f"Session Paused · {device_str}"
    elif is_streaming:
        dot_color = "#10B981"
        badge_text = f"Live Session Active · {device_str}"
    else:
        dot_color = theme["accent"]
        badge_text = f"Ready · {device_str}"

    if is_streaming and not is_paused:
        priv_bg, priv_fg, priv_dot = "#EBF7EE", "#1B5E2E", "#16A34A"
        priv_text = "🔒 Privacy Shield: Live Cam/Mic Active · Ephemeral RAM Only · Nothing Saved"
    else:
        priv_bg, priv_fg, priv_dot = "#F4F2F8", "#4A4658", "#7C6FA6"
        priv_text = "🔒 Privacy Shield: Cam/Mic Inactive · Zero Recording · Zero Stored Data"

    render_html(
        f"""
        <div class="aura-header">
            <div class="aura-brand">
                <div class="aura-logo-badge">{theme['emoji']}</div>
                <div>
                    <h1 class="aura-title">AuraSense</h1>
                    <p class="aura-subtitle">Real-Time Multimodal Emotion Analysis</p>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
                <div class="aura-privacy-pill" style="background:{priv_bg}; color:{priv_fg};">
                    <span style="width:8px; height:8px; border-radius:50%; background:{priv_dot};"></span>
                    <span>{priv_text}</span>
                </div>
                <div style="display:inline-flex; align-items:center; gap:0.48rem; padding:0.44rem 0.92rem;
                            border-radius:999px; background:#FAF8F5; border:1px solid rgba(43,40,54,0.07);
                            font-size:0.75rem; font-weight:700; color:#2B2836;">
                    <span style="width:9px; height:9px; border-radius:50%; background:{dot_color};
                                 box-shadow:0 0 0 3px {dot_color}28;"></span>
                    <span>{badge_text}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for err in crit_errors:
        render_html(
            f"""
            <div class="aura-error-banner">
                <strong>⚠️ Pretrained Model Load Error:</strong> {html.escape(str(err))}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prominent_facial_gender_card(
    snapshot: Dict[str, Any],
    theme: Dict[str, str],
    is_streaming: bool,
) -> None:
    """
    HIGHEST-PRIORITY HERO CARD: Dual-Model Facial Gender Presentation Estimate.
    - Runs InsightFace Buffalo_L (Model 1) and FairFace ResNet34 (Model 2) on the same detected face.
    - Displays Model 1 prediction & confidence and Model 2 prediction & confidence independently.
    - Displays Agreement / Disagreement indicator (`Models Agree`, `Models Disagree`, `Quality Gate Blocked`).
    - Displays `"Uncertain – Please adjust camera position"` whenever models disagree, confidence < 65%,
      or face is small, blurred, poorly lit, occluded, or at an extreme angle.
    - Never allows voice predictions to override facial predictions.
    """
    result = snapshot.get("result") or {}
    dual = result.get("dual_facial_gender") or {}
    face_diag = (result.get("face_analysis") or {}).get("diagnostics") or {}

    if not is_streaming:
        headline = "Standby — Start Live Camera"
        sub_reason = "InsightFace Buffalo_L + FairFace ResNet34 ready for 5-point aligned face verification."
        agree_badge_txt = "Awaiting Stream"
        agree_bg, agree_fg = "#F4F2F8", "#5E5A6E"
        m1_val, m1_conf = "Standby", "—"
        m2_val, m2_conf = "Standby", "—"
        conf_pill_html = ""
    elif not dual or dual.get("status") in {"no_face", "no_frames"}:
        headline = "No Face Detected"
        sub_reason = "Position your face clearly inside the webcam frame."
        agree_badge_txt = "Awaiting Face"
        agree_bg, agree_fg = "#FFF7E6", "#9A5B13"
        m1_val, m1_conf = "No face", "—"
        m2_val, m2_conf = "No face", "—"
        conf_pill_html = ""
    else:
        status = str(dual.get("status", ""))
        headline = html.escape(str(dual.get("final_label") or GENDER_UNCERTAIN_MESSAGE))
        sub_reason = html.escape(str(dual.get("reason") or ""))
        agree_badge_txt = html.escape(str(dual.get("agreement_status") or "Uncertain"))

        if status == "confident_agreement":
            agree_bg, agree_fg = "#EBF7EE", "#1B6533"
        elif status == "disagreement":
            agree_bg, agree_fg = "#FDF4F5", "#A62639"
        else:
            agree_bg, agree_fg = "#FFF7E6", "#9A5B13"

        m1 = dual.get("model_1") or {}
        m2 = dual.get("model_2") or {}

        if status == "quality_rejected":
            m1_val = "Blocked by Quality Gate"
            m1_conf = "Adjust position / light"
            m2_val = "Blocked by Quality Gate"
            m2_conf = "Adjust position / light"
        else:
            m1_val = html.escape(str(m1.get("label") or "Unavailable"))
            m1_conf = (
                f"{float(m1['confidence_pct']):.1f}% score"
                if m1.get("confidence_pct") is not None
                else "—"
            )
            m2_val = html.escape(str(m2.get("label") or "Unavailable"))
            m2_conf = (
                f"{float(m2['confidence_pct']):.1f}% score"
                if m2.get("confidence_pct") is not None
                else "—"
            )

        # Show overall confidence pill ONLY when properly available (both models agree & pass quality)
        cons_conf = dual.get("confidence_pct")
        if (not dual.get("is_uncertain", True)) and cons_conf is not None:
            conf_pill_html = (
                f'<span class="aura-mono" style="display:inline-block; margin-top:0.25rem; padding:0.2rem 0.68rem; '
                f'border-radius:999px; background:rgba(255,255,255,0.88); color:{theme["accent"]}; '
                f'font-size:0.75rem; font-weight:800;">Verified Min Score: {float(cons_conf):.1f}%</span>'
            )
        else:
            conf_pill_html = ""

    pose_meta = ""
    if is_streaming and face_diag.get("detected_face_count", 0) > 0:
        pose_meta = (
            f"5-pt Pose: yaw={face_diag.get('yaw_asymmetry', 0.0):.2f} · "
            f"roll={face_diag.get('roll_deg', 0.0):.0f}° · "
            f"sharpness={face_diag.get('mean_sharpness_laplacian', 0.0):.0f} · "
            f"luma={face_diag.get('mean_luma', 0.0):.0f}"
        )

    render_html(
        f"""
        <div class="aura-hero-gender-card">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;
                        padding-bottom:0.65rem; margin-bottom:0.75rem; border-bottom:1px solid rgba(43,40,54,0.07);">
                <span style="font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:0.07em; color:{theme['accent']};">
                    ★ Facial Gender Presentation Estimate (Dual-Model Consensus)
                </span>
                <span style="padding:0.28rem 0.78rem; border-radius:999px; background:{agree_bg}; color:{agree_fg};
                             font-size:0.71rem; font-weight:800;">
                    {agree_badge_txt}
                </span>
            </div>

            <div>
                <div style="font-size:1.42rem; font-weight:800; color:#2B2836; line-height:1.22;">
                    {headline}
                </div>
                {conf_pill_html}
                <div style="font-size:0.76rem; color:#5E5A6E; font-weight:600; margin-top:0.28rem;">
                    {sub_reason}
                </div>
            </div>

            <!-- Side-by-Side Independent Model 1 & Model 2 Outputs on the Same Face -->
            <div class="aura-grid-2" style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.72rem; margin-top:0.9rem;">
                <div style="background:rgba(255,255,255,0.88); border-radius:22px; padding:0.88rem 1rem; border:1px solid rgba(43,40,54,0.06);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:0.4rem;">
                        <span style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#68637A;">
                            Model 1 · InsightFace Buffalo_L
                        </span>
                        <span class="aura-mono" style="font-size:0.69rem; font-weight:700; color:{theme['accent']};">{m1_conf}</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:800; color:#2B2836; margin-top:0.24rem;">
                        {m1_val}
                    </div>
                    <div style="font-size:0.62rem; color:#8C879B; margin-top:0.12rem;">SCRFD + genderage.onnx (96×96 aligned)</div>
                </div>

                <div style="background:rgba(255,255,255,0.88); border-radius:22px; padding:0.88rem 1rem; border:1px solid rgba(43,40,54,0.06);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:0.4rem;">
                        <span style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#68637A;">
                            Model 2 · FairFace ResNet34
                        </span>
                        <span class="aura-mono" style="font-size:0.69rem; font-weight:700; color:{theme['accent']};">{m2_conf}</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:800; color:#2B2836; margin-top:0.24rem;">
                        {m2_val}
                    </div>
                    <div style="font-size:0.62rem; color:#8C879B; margin-top:0.12rem;">5-pt Landmark Aligned Chip (224×224)</div>
                </div>
            </div>

            <div style="margin-top:0.75rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;
                        font-size:0.65rem; color:#68637A;">
                <span><strong>Note:</strong> Estimates perceived Male/Female presentation from facial appearance — never gender identity. Voice never overrides face.</span>
                <span class="aura-mono" style="color:#8C879B;">{html.escape(pose_meta)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_separate_vocal_presentation_card(
    snapshot: Dict[str, Any],
    model_status: Dict[str, Any],
    theme: Dict[str, str],
    is_streaming: bool,
) -> None:
    """Renders Vocal Gender Presentation in its own clearly isolated card (never merged with facial)."""
    result = snapshot.get("result") or {}
    voice_pres = result.get("voice_presentation") or {}
    vp_model_ready = bool(model_status.get("voice_presentation_ready"))

    if not vp_model_ready:
        v_lbl = "Not Available"
        v_meta = "No verified vocal presentation model loaded"
    elif not is_streaming:
        v_lbl = "Standby"
        v_meta = "Awaiting live microphone speech window"
    elif voice_pres.get("available"):
        v_lbl = html.escape(str(voice_pres.get("label") or "Uncertain"))
        v_conf = voice_pres.get("confidence_pct")
        if (not voice_pres.get("is_uncertain")) and v_conf is not None:
            v_meta = f"{float(v_conf):.1f}% acoustic score"
        else:
            v_meta = "Uncertain acoustic margin"
    else:
        raw_v_lbl = str(voice_pres.get("label") or "")
        if raw_v_lbl.lower() == "not available":
            v_lbl = "Not Available"
            v_meta = "Model unavailable"
        else:
            v_lbl = "Waiting for speech"
            v_meta = "Speak naturally (3–5s) · Kept strictly separate from facial estimate"

    render_html(
        f"""
        <div class="aura-surface-card" style="padding:0.95rem 1.25rem; margin-bottom:0;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.6rem;">
                <div>
                    <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; color:#68637A;">
                        🎙️ Separate Vocal Presentation Estimate (Independent of Face)
                    </div>
                    <div style="font-size:0.95rem; font-weight:800; color:#2B2836; margin-top:0.18rem;">
                        {v_lbl}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.72rem; font-weight:700; color:{theme['accent']};">{v_meta}</div>
                    <div style="font-size:0.62rem; color:#8C879B;">Never overrides facial presentation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_camera_tracking_bar(
    snapshot: Dict[str, Any],
    theme: Dict[str, str],
    is_streaming: bool,
) -> None:
    """2. Live Camera Telemetry: Face detection indicator, Face tracking status, device alerts, and 16 kHz Audio Spectrum."""
    device_alert = snapshot.get("device_alert")
    if device_alert:
        render_html(
            f"""
            <div class="aura-warn-banner">
                <strong>📷 Hardware / Permission Notice:</strong> {html.escape(str(device_alert))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    face_detected = bool(snapshot.get("face_detected"))
    tracking_status = html.escape(str(snapshot.get("face_tracking_status", "Camera Standby")))
    audio_bars: List[int] = snapshot.get("audio_bars") or [12] * 24
    buffered_sec = float(snapshot.get("buffered_audio_sec", 0.0))
    target_sec = float(snapshot.get("rolling_window_target_sec", 4.0))

    if not is_streaming:
        det_bg, det_fg, det_dot, det_text = "#F3F1F8", "#68637A", "#9CA3AF", "Face Detection: Standby"
    elif face_detected:
        det_bg, det_fg, det_dot, det_text = "#EBF7EE", "#1E6B38", "#22C55E", "Face Detected (5-pt SCRFD)"
    else:
        det_bg, det_fg, det_dot, det_text = "#FFF6E8", "#A66315", "#F59E0B", "No Face in Frame"

    bars_html = "".join(
        f'<div style="width:5px; height:{int(h)}px; border-radius:999px; '
        f'background:{theme["accent"]}; opacity:{0.88 if is_streaming else 0.32}; '
        f'transition:height 0.14s ease;"></div>'
        for h in audio_bars
    )

    f_diag = ((snapshot.get("result") or {}).get("face_analysis") or {}).get("diagnostics") or {}
    gate_issues = f_diag.get("gender_gate_issues") or []
    env_html = ""
    if is_streaming and face_detected and gate_issues:
        env_html = (
            f'<div style="margin-top:0.65rem; padding:0.45rem 0.78rem; border-radius:14px; '
            f'background:#FFFBF0; border:1px solid #FFE69E; font-size:0.71rem; color:#8A5D14; font-weight:600;">'
            f'⚠️ Camera Guidance: {html.escape("; ".join(gate_issues))}</div>'
        )

    render_html(
        f"""
        <div class="aura-surface-card" style="padding:1rem 1.25rem; margin-top:0.35rem;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.75rem; margin-bottom:0.8rem;">
                <div style="display:inline-flex; align-items:center; gap:0.48rem; padding:0.38rem 0.85rem;
                            border-radius:999px; background:{det_bg}; color:{det_fg}; font-size:0.75rem; font-weight:700;">
                    <span style="width:8px; height:8px; border-radius:50%; background:{det_dot};"></span>
                    <span>{det_text}</span>
                </div>

                <div style="display:inline-flex; align-items:center; gap:0.45rem; padding:0.38rem 0.88rem;
                            border-radius:999px; background:#FAF8F5; border:1px solid rgba(43,40,54,0.07);
                            font-size:0.74rem; font-weight:700; color:#4A4658;">
                    <span>🎯 Tracking:</span>
                    <span style="color:{theme['accent']};">{tracking_status}</span>
                </div>
            </div>

            <div style="display:flex; align-items:center; justify-content:space-between; gap:0.8rem; flex-wrap:wrap;
                        padding-top:0.7rem; border-top:1px solid rgba(43,40,54,0.06);">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <div style="width:34px; height:34px; border-radius:12px; background:{theme['pill']}; color:{theme['accent']};
                                display:flex; align-items:center; justify-content:center; font-size:0.95rem;">
                        🎙️
                    </div>
                    <div style="display:flex; align-items:flex-end; gap:4px; height:36px;">
                        {bars_html}
                    </div>
                </div>
                <div class="aura-mono" style="font-size:0.71rem; color:#68637A; font-weight:600;">
                    Speech Buffer: {buffered_sec:.1f}s / {target_sec:.1f}s (16 kHz Mono)
                </div>
            </div>
            {env_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_three_emotion_cards(
    snapshot: Dict[str, Any],
    theme: Dict[str, str],
    is_streaming: bool,
) -> None:
    """3. Live Emotion Analysis — Three separate cards (Facial, Vocal, Combined)."""
    result = snapshot.get("result")
    face_a = (result or {}).get("face_analysis") or {}
    voice_a = (result or {}).get("voice_analysis") or {}
    weights = (result or {}).get("modality_weights") or {"voice_pct": 0.0, "face_pct": 0.0}
    fusion_mode = (result or {}).get("fusion_mode", "")
    contrib_modalities = (result or {}).get("contributing_modalities") or []

    if face_a.get("available"):
        f_emo = str(face_a["predicted_emotion"])
        f_pal = PASTEL_PALETTES.get(f_emo, PASTEL_PALETTES["Neutral"])
        f_title = html.escape(f_emo)
        f_used = "facial" in contrib_modalities
        f_sub = (
            f"Score: {float(face_a['probability_pct']):.1f}%"
            if f_used
            else f"Below threshold ({float(face_a['probability_pct']):.1f}%)"
        )
        f_emoji = f_pal["emoji"]
        f_bg = f_pal["card"] if f_used else "#F5F3F9"
        f_accent = f_pal["accent"] if f_used else "#8C879B"
    else:
        f_title = "Standby" if not is_streaming else (
            "Model Error" if face_a.get("error") else "No Face Detected"
        )
        f_sub = "Start camera" if not is_streaming else "Center face"
        f_emoji = "📷"
        f_bg = "#F5F3F9"
        f_accent = "#68637A"

    if voice_a.get("available"):
        v_uni = str(voice_a["predicted_emotion"])
        v_raw = str(voice_a.get("raw_model_label") or v_uni).capitalize()
        v_pal = PASTEL_PALETTES.get("Calm" if v_raw.lower() == "calm" else v_uni, PASTEL_PALETTES["Neutral"])
        v_title = html.escape(v_raw)
        v_prob = float(voice_a.get("raw_probability_pct") or voice_a["probability_pct"])
        v_used = "vocal" in contrib_modalities
        v_sub = f"Score: {v_prob:.1f}%" if v_used else f"Below threshold ({v_prob:.1f}%)"
        v_emoji = v_pal["emoji"]
        v_bg = v_pal["card"] if v_used else "#F5F3F9"
        v_accent = v_pal["accent"] if v_used else "#8C879B"
    else:
        v_title = "Standby" if not is_streaming else html.escape(
            str(voice_a.get("display_state") or "Waiting for speech")
        )
        v_sub = "Start mic" if not is_streaming else "Speak 3–5s"
        v_emoji = "🎙️"
        v_bg = "#F5F3F9"
        v_accent = "#68637A"

    if result and result.get("available") and result.get("dominant_emotion"):
        c_emo = str(result["dominant_emotion"])
        c_theme_key = str(result.get("theme_key") or c_emo)
        c_pal = PASTEL_PALETTES.get(c_theme_key, PASTEL_PALETTES["Neutral"])
        c_title = html.escape(c_emo)
        c_sub = f"Fused Score: {float(result['emotion_confidence']):.1f}%"
        c_emoji = c_pal["emoji"]
        c_bg = c_pal["card"]
        c_accent = c_pal["accent"]
        if fusion_mode == "facial_only":
            c_footer = "⚠️ Unimodal: Facial Only"
        elif fusion_mode == "vocal_only":
            c_footer = "⚠️ Unimodal: Vocal Only"
        else:
            c_footer = f"Face {float(weights.get('face_pct', 0.0)):.0f}% · Voice {float(weights.get('voice_pct', 0.0)):.0f}%"
    elif result is not None and not result.get("available"):
        c_title = "Insufficient data"
        c_sub = "No reliable signal"
        c_emoji = "⏳"
        c_bg = "#F5F3F9"
        c_accent = "#7C778B"
        c_footer = "Waiting for face/speech"
    else:
        c_title = "Standby" if not is_streaming else "Buffering..."
        c_sub = "Expressed Emotion"
        c_emoji = "✨"
        c_bg = theme["card"]
        c_accent = theme["accent"]
        c_footer = "Multimodal Fusion"

    render_html(
        f"""
        <div class="aura-surface-card" style="padding:1.1rem 1.3rem;">
            <div class="aura-section-label">
                <span>Live Emotion Analysis · Estimated Expressed Emotion</span>
                <span class="aura-mono" style="font-size:0.67rem; color:#8C879B;">Uncalibrated model scores</span>
            </div>

            <div class="aura-grid-3" style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:0.68rem;">
                <div style="background:{f_bg}; border-radius:22px; padding:0.92rem; border:1px solid rgba(43,40,54,0.06);">
                    <div style="display:flex; justify-content:space-between; font-size:0.63rem; font-weight:800; text-transform:uppercase; color:#5E5A6E;">
                        <span>Facial Emotion</span><span>{f_emoji}</span>
                    </div>
                    <div style="font-size:1.06rem; font-weight:800; color:#2B2836; margin-top:0.35rem;">{f_title}</div>
                    <div style="font-size:0.7rem; font-weight:700; color:{f_accent}; margin-top:0.12rem;">{f_sub}</div>
                </div>

                <div style="background:{v_bg}; border-radius:22px; padding:0.92rem; border:1px solid rgba(43,40,54,0.06);">
                    <div style="display:flex; justify-content:space-between; font-size:0.63rem; font-weight:800; text-transform:uppercase; color:#5E5A6E;">
                        <span>Vocal Emotion</span><span>{v_emoji}</span>
                    </div>
                    <div style="font-size:1.06rem; font-weight:800; color:#2B2836; margin-top:0.35rem;">{v_title}</div>
                    <div style="font-size:0.7rem; font-weight:700; color:{v_accent}; margin-top:0.12rem;">{v_sub}</div>
                </div>

                <div style="background:{c_bg}; border-radius:22px; padding:0.92rem; border:2px solid {c_accent}55;">
                    <div style="display:flex; justify-content:space-between; font-size:0.63rem; font-weight:800; text-transform:uppercase; color:#5E5A6E;">
                        <span>Combined Emotion</span><span>{c_emoji}</span>
                    </div>
                    <div style="font-size:1.06rem; font-weight:800; color:#2B2836; margin-top:0.35rem;">{c_title}</div>
                    <div style="font-size:0.7rem; font-weight:700; color:{c_accent}; margin-top:0.12rem;">{c_sub}</div>
                    <div style="font-size:0.6rem; color:#4A4658; font-weight:700; margin-top:0.2rem;">{c_footer}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_emotion_probability_bars(snapshot: Dict[str, Any], theme: Dict[str, str]) -> None:
    """4. Emotion Probability: Horizontal bars for Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral."""
    result = snapshot.get("result") or {}
    has_data = bool(result.get("available") and result.get("combined_probabilities"))
    probs: Dict[str, float] = result.get("combined_probabilities") or {}
    face_norm: Dict[str, float] = result.get("normalized_facial_probabilities") or {}
    voice_norm: Dict[str, float] = result.get("normalized_vocal_probabilities") or {}
    dom_emo = result.get("dominant_emotion")

    rows_html: List[str] = []
    for internal_key, display_label in DISPLAY_EMOTION_ORDER:
        val = float(probs.get(internal_key, 0.0)) if has_data else 0.0
        is_dominant = bool(has_data and internal_key == dom_emo)
        pal = PASTEL_PALETTES.get(internal_key, PASTEL_PALETTES["Neutral"])
        bar_color = pal["accent"] if is_dominant else "#B6AFC7"
        val_txt = f"{val:.1f}%" if has_data else "—"

        sub_parts = []
        if has_data and internal_key in face_norm:
            sub_parts.append(f"F:{face_norm[internal_key]:.0f}%")
        if has_data and internal_key in voice_norm:
            sub_parts.append(f"V:{voice_norm[internal_key]:.0f}%")
        sub_txt = f" ({' · '.join(sub_parts)})" if sub_parts else ""

        rows_html.append(
            f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                            font-size:0.74rem; font-weight:{'800' if is_dominant else '600'}; margin-bottom:0.18rem;">
                    <span style="color:#2B2836; display:inline-flex; align-items:center; gap:0.38rem;">
                        <span>{pal['emoji']}</span>
                        <span>{html.escape(display_label)}</span>
                    </span>
                    <span class="aura-mono" style="color:{pal['accent'] if is_dominant else '#68637A'}; font-size:0.71rem;">
                        {val_txt}<span style="color:#9A94A8; font-weight:500;">{sub_txt}</span>
                    </span>
                </div>
                <div style="width:100%; height:8px; border-radius:999px; background:#F1EEF6; overflow:hidden;">
                    <div style="width:{min(100.0, max(0.0, val))}%; height:100%; border-radius:999px;
                                background:{bar_color}; transition:width 0.38s ease;"></div>
                </div>
            </div>
            """
        )

    render_html(
        f"""
        <div class="aura-surface-card" style="padding:1.1rem 1.3rem;">
            <div class="aura-section-label">
                <span>Emotion Probability Distribution</span>
                <span style="font-size:0.66rem; font-weight:700; color:{theme['accent']};">
                    {"Smoothed" if result.get("smoothing_applied") else "Normalized Scores"}
                </span>
            </div>
            {"".join(rows_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status_strip(
    snapshot: Dict[str, Any],
    model_status: Dict[str, Any],
    theme: Dict[str, str],
) -> None:
    """6. System Status strip."""
    last_error = snapshot.get("last_error")
    if last_error:
        render_html(
            f"""
            <div class="aura-error-banner">
                <strong>⚠️ Live Model Inference Error:</strong> {html.escape(str(last_error))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    cam_ok = bool(snapshot.get("camera_connected"))
    mic_ok = bool(snapshot.get("microphone_connected"))
    face_ok = bool(snapshot.get("face_detected"))
    voice_ok = bool(snapshot.get("voice_detected"))
    models_ok = bool(model_status.get("all_core_ready"))
    inf_status = html.escape(str(snapshot.get("inference_status", "Standby")))

    items = [
        ("Camera Connected", "Connected" if cam_ok else "Disconnected", cam_ok),
        ("Microphone Connected", "Connected" if mic_ok else "Disconnected", mic_ok),
        ("Face Detected", "Detected" if face_ok else "No Face", face_ok),
        ("Voice Detected", "Speech Active" if voice_ok else "Waiting for Speech", voice_ok),
        ("Models Loaded", f"Ready ({model_status.get('device', 'CPU')})" if models_ok else "Error", models_ok),
        ("Inference Status", inf_status, inf_status not in {"Standby", "Error", "Paused"}),
    ]

    pills_html = []
    for title, value, active in items:
        dot_color = "#10B981" if active else ("#EF4444" if value == "Error" else "#9CA3AF")
        bg_color = "#F3FAF6" if active else "#FAF8F5"
        pills_html.append(
            f"""
            <div style="background:{bg_color}; border-radius:20px; padding:0.72rem 0.92rem;
                        border:1px solid rgba(43,40,54,0.06); display:flex; flex-direction:column; gap:0.2rem;">
                <div style="font-size:0.63rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; color:#7C778B;">
                    {html.escape(title)}
                </div>
                <div style="display:flex; align-items:center; gap:0.45rem; font-size:0.8rem; font-weight:800; color:#2B2836;">
                    <span style="width:8px; height:8px; border-radius:50%; background:{dot_color}; flex-shrink:0;"></span>
                    <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{value}</span>
                </div>
            </div>
            """
        )

    render_html(
        f"""
        <div class="aura-surface-card" style="padding:1.1rem 1.35rem; margin-top:0.85rem;">
            <div class="aura-section-label" style="margin-bottom:0.65rem;">
                <span>System Status</span>
                <span class="aura-mono" style="font-size:0.68rem; color:{theme['accent']};">
                    Buffalo_L + FairFace ResNet34 · RAM Buffer: {float(snapshot.get('buffer_memory_mb', 0.0)):.2f} MB
                </span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(min(160px, 100%), 1fr)); gap:0.65rem;">
                {"".join(pills_html)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnostics_page(
    snapshot: Dict[str, Any],
    model_status: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    theme: Dict[str, str],
) -> None:
    """
    Model Diagnostics & Empirical Indian Faces Evaluation Page:
    Displays loaded model names, cache load counts, latency, and actual measured benchmark results
    on diverse Indian faces (`piadev/fairface-indian`) alongside documented limitations.
    """
    device_name = html.escape(str(model_status.get("device", "CPU")))
    model_rows = [
        (
            "Facial Gender Model 1 + 5-pt Detector",
            str(model_status.get("buffalo_model")),
            "Loaded & Cached" if model_status.get("buffalo_ready") else "Load Failed",
            model_status.get("buffalo_ready"),
            f"{float(model_status.get('buffalo_load_ms', 0.0)):.0f} ms (loads={model_status.get('buffalo_loads', 0)})",
        ),
        (
            "Facial Gender Model 2 (Independent)",
            str(model_status.get("fairface_model")),
            "Loaded & Cached" if model_status.get("fairface_ready") else "Load Failed",
            model_status.get("fairface_ready"),
            f"{float(model_status.get('fairface_load_ms', 0.0)):.0f} ms (loads={model_status.get('fairface_loads', 0)})",
        ),
        (
            "Facial Emotion Recognition",
            "DeepFace Facial Expression CNN (5-pt Aligned Crop)",
            "Loaded & Cached" if model_status.get("deepface_ready") else "Load Failed",
            model_status.get("deepface_ready"),
            f"{float(model_status.get('deepface_load_ms', 0.0)):.0f} ms (loads={model_status.get('deepface_loads', 0)})",
        ),
        (
            "Voice Emotion Recognition",
            str(model_status.get("voice_emotion_model")),
            "Loaded & Cached" if model_status.get("voice_emotion_ready") else "Load Failed",
            model_status.get("voice_emotion_ready"),
            f"{float(model_status.get('voice_emotion_load_ms', 0.0)):.0f} ms (loads={model_status.get('voice_emotion_loads', 0)})",
        ),
        (
            "Vocal Presentation (Kept Separate)",
            str(model_status.get("voice_presentation_model")),
            "Loaded & Cached" if model_status.get("voice_presentation_ready") else "Not Available",
            model_status.get("voice_presentation_ready"),
            f"{float(model_status.get('voice_presentation_load_ms', 0.0)):.0f} ms (loads={model_status.get('voice_presentation_loads', 0)})",
        ),
    ]

    tr_html = []
    for task, name, status_txt, ok, init_lat in model_rows:
        pill_bg = "#EBF7EE" if ok else "#FFF4E5"
        pill_fg = "#1E6B38" if ok else "#B45309"
        tr_html.append(
            f"""
            <tr style="border-bottom:1px solid rgba(43,40,54,0.06);">
                <td style="padding:0.7rem 0.6rem; font-weight:700; color:#2B2836;">{html.escape(task)}</td>
                <td class="aura-mono" style="padding:0.7rem 0.6rem; font-size:0.74rem; color:{theme['accent']};">{html.escape(name)}</td>
                <td style="padding:0.7rem 0.6rem;">
                    <span style="padding:0.22rem 0.65rem; border-radius:999px; background:{pill_bg}; color:{pill_fg}; font-size:0.7rem; font-weight:700;">
                        {html.escape(status_txt)}
                    </span>
                </td>
                <td class="aura-mono" style="padding:0.7rem 0.6rem; font-size:0.73rem; color:#5E5A6E;">{html.escape(init_lat)}</td>
            </tr>
            """
        )

    render_html(
        f"""
        <div class="aura-surface-card">
            <div class="aura-section-label">
                <span>Pretrained Model Registry &amp; Live Runtime Diagnostics</span>
                <span class="aura-pill-tag">Device: {device_name} · Window Latency: {float(snapshot.get('total_latency_ms', 0.0)):.0f} ms</span>
            </div>
            <div style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.79rem;">
                    <thead>
                        <tr style="border-bottom:2px solid rgba(43,40,54,0.08); font-size:0.68rem; text-transform:uppercase; color:#68637A;">
                            <th style="padding:0.55rem;">Pipeline Component</th>
                            <th style="padding:0.55rem;">Pretrained Checkpoint</th>
                            <th style="padding:0.55rem;">Status</th>
                            <th style="padding:0.55rem;">Startup Load Time</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(tr_html)}</tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if benchmark_data and benchmark_data.get("overall_metrics"):
        ov = benchmark_data["overall_metrics"]
        subgroup_rows = []
        for group_title, group_dict in [
            ("Presentation Label", benchmark_data.get("by_presentation_label", {})),
            ("Age Bracket", benchmark_data.get("by_age_bracket", {})),
            ("Skin-Tone Lightness (CIE L*)", benchmark_data.get("by_skin_tone_luma", {})),
            ("Lighting Condition", benchmark_data.get("by_lighting_condition", {})),
        ]:
            for slice_name, metrics in group_dict.items():
                if not metrics.get("n"):
                    continue
                sel_acc = (
                    f"{metrics['consensus_selective_accuracy_pct']:.2f}%"
                    if metrics.get("consensus_selective_accuracy_pct") is not None
                    else "—"
                )
                agree65_acc = (
                    f"{metrics['agreement_65_accuracy_pct']:.2f}%"
                    if metrics.get("agreement_65_accuracy_pct") is not None
                    else "—"
                )
                subgroup_rows.append(
                    f"""
                    <tr style="border-bottom:1px solid rgba(43,40,54,0.06);">
                        <td style="padding:0.58rem; font-size:0.72rem; color:#68637A; font-weight:700;">{html.escape(group_title)}</td>
                        <td style="padding:0.58rem; font-weight:700; color:#2B2836;">{html.escape(slice_name)} (n={metrics['n']})</td>
                        <td class="aura-mono" style="padding:0.58rem;">{metrics['buffalo_l_accuracy_pct']:.2f}%</td>
                        <td class="aura-mono" style="padding:0.58rem;">{metrics['fairface_resnet34_accuracy_pct']:.2f}%</td>
                        <td class="aura-mono" style="padding:0.58rem;">{metrics['inter_model_agreement_pct']:.2f}%</td>
                        <td class="aura-mono" style="padding:0.58rem;">{agree65_acc}</td>
                        <td class="aura-mono" style="padding:0.58rem; font-weight:800; color:{theme['accent']};">{sel_acc}</td>
                        <td class="aura-mono" style="padding:0.58rem;">{metrics['uncertain_abstention_pct']:.2f}%</td>
                    </tr>
                    """
                )

        lims_html = "".join(
            f"<li style='margin-bottom:0.35rem;'>{html.escape(str(item))}</li>"
            for item in benchmark_data.get("documented_limitations", [])
        )

        render_html(
            f"""
            <div class="aura-surface-card">
                <div class="aura-section-label">
                    <span>Empirical Evaluation · Diverse Indian Faces Benchmark</span>
                    <span class="aura-mono" style="font-size:0.7rem; color:{theme['accent']};">
                        n={benchmark_data['faces_detected_count']} detected / {benchmark_data['total_sampled_images']} held-out Indian faces
                    </span>
                </div>

                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(min(170px, 100%), 1fr)); gap:0.7rem; margin-bottom:1rem;">
                    <div style="background:#FAF8F5; border-radius:20px; padding:0.85rem 1rem;">
                        <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#7C778B;">Model 1 · Buffalo_L Acc</div>
                        <div class="aura-mono" style="font-size:1.15rem; font-weight:800; color:#2B2836; margin-top:0.2rem;">{ov['buffalo_l_accuracy_pct']:.2f}%</div>
                    </div>
                    <div style="background:#FAF8F5; border-radius:20px; padding:0.85rem 1rem;">
                        <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#7C778B;">Model 2 · FairFace Acc</div>
                        <div class="aura-mono" style="font-size:1.15rem; font-weight:800; color:#2B2836; margin-top:0.2rem;">{ov['fairface_resnet34_accuracy_pct']:.2f}%</div>
                    </div>
                    <div style="background:#FAF8F5; border-radius:20px; padding:0.85rem 1rem;">
                        <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#7C778B;">Consensus Selective Acc</div>
                        <div class="aura-mono" style="font-size:1.15rem; font-weight:800; color:{theme['accent']}; margin-top:0.2rem;">{ov['consensus_selective_accuracy_pct']:.2f}%</div>
                        <div style="font-size:0.65rem; color:#5E5A6E;">When both models agree (≥65%)</div>
                    </div>
                    <div style="background:#FAF8F5; border-radius:20px; padding:0.85rem 1rem;">
                        <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#7C778B;">Uncertain Abstention Rate</div>
                        <div class="aura-mono" style="font-size:1.15rem; font-weight:800; color:#2B2836; margin-top:0.2rem;">{ov['uncertain_abstention_pct']:.2f}%</div>
                        <div style="font-size:0.65rem; color:#5E5A6E;">{benchmark_data['disagreement_caught_error_rate_pct']:.1f}% of disagreements had a model error</div>
                    </div>
                </div>

                <div style="overflow-x:auto; margin-bottom:1rem;">
                    <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.76rem;">
                        <thead>
                            <tr style="border-bottom:2px solid rgba(43,40,54,0.08); font-size:0.66rem; text-transform:uppercase; color:#68637A;">
                                <th style="padding:0.5rem;">Dimension</th>
                                <th style="padding:0.5rem;">Subgroup</th>
                                <th style="padding:0.5rem;">Buffalo_L</th>
                                <th style="padding:0.5rem;">FairFace ResNet34</th>
                                <th style="padding:0.5rem;">Agreement</th>
                                <th style="padding:0.5rem;">Agreed ≥65% Acc</th>
                                <th style="padding:0.5rem;">Quality-Gated Acc</th>
                                <th style="padding:0.5rem;">Abstained (%)</th>
                            </tr>
                        </thead>
                        <tbody>{"".join(subgroup_rows)}</tbody>
                    </table>
                </div>

                <div style="padding:0.9rem 1.1rem; border-radius:20px; background:#FAF8F5; font-size:0.76rem; color:#4A4658;">
                    <strong>Documented Limitations:</strong>
                    <ul style="margin:0.4rem 0 0 1.1rem; padding:0;">{lims_html}</ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_clip_inspector_results(clip_res: Dict[str, Any], theme: Dict[str, str]) -> None:
    """Renders Clip Inspector summary cards and per-segment emotion timeline."""
    if not clip_res.get("available"):
        render_html(
            f"""
            <div class="aura-error-banner">
                <strong>⚠️ Clip Analysis Error:</strong> {html.escape(str(clip_res.get("error") or "Unknown error"))}
            </div>
            """
        )
        return

    conf_str = (
        f"{float(clip_res['avg_confidence_pct']):.1f}%"
        if clip_res.get("avg_confidence_pct") is not None
        else "—"
    )
    multi_note = " · Multiple faces seen (primary analyzed)" if clip_res.get("multiple_faces_seen") else ""
    audio_note = "Audio track present" if clip_res.get("has_audio_track") else "No audio track (facial-only)"

    render_html(
        f"""
        <div class="aura-hero-gender-card">
            <div class="aura-section-label">
                <span>🎞️ Clip Inspector Summary · {html.escape(str(clip_res.get("filename", "")))}</span>
                <span class="aura-mono" style="font-size:0.7rem; color:{theme['accent']};">
                    {clip_res.get('duration_sec', 0.0):.1f}s · {clip_res.get('segments_analyzed', 0)} segments · {audio_note}{multi_note}
                </span>
            </div>
            <div class="aura-grid-2" style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.75rem; margin-bottom:0.75rem;">
                <div style="background:rgba(255,255,255,0.9); border-radius:20px; padding:0.9rem 1rem;">
                    <div style="font-size:0.65rem; font-weight:800; text-transform:uppercase; color:#68637A;">Estimated Facial Presentation</div>
                    <div style="font-size:1.1rem; font-weight:800; color:#2B2836; margin-top:0.25rem;">{html.escape(str(clip_res.get("overall_facial_presentation", "Uncertain")))}</div>
                </div>
                <div style="background:rgba(255,255,255,0.9); border-radius:20px; padding:0.9rem 1rem;">
                    <div style="font-size:0.65rem; font-weight:800; text-transform:uppercase; color:#68637A;">Combined Expressed Emotion</div>
                    <div style="font-size:1.1rem; font-weight:800; color:{theme['accent']}; margin-top:0.25rem;">{html.escape(str(clip_res.get("overall_combined_emotion", "Insufficient data")))} <span class="aura-mono" style="font-size:0.8rem;">({conf_str})</span></div>
                </div>
            </div>
            <div class="aura-grid-2" style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.75rem;">
                <div style="background:rgba(255,255,255,0.85); border-radius:18px; padding:0.75rem 1rem;">
                    <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#68637A;">Facial Emotion</div>
                    <div style="font-size:0.98rem; font-weight:800; color:#2B2836;">{html.escape(str(clip_res.get("overall_facial_emotion", "—")))}</div>
                </div>
                <div style="background:rgba(255,255,255,0.85); border-radius:18px; padding:0.75rem 1rem;">
                    <div style="font-size:0.64rem; font-weight:800; text-transform:uppercase; color:#68637A;">Vocal Emotion</div>
                    <div style="font-size:0.98rem; font-weight:800; color:#2B2836;">{html.escape(str(clip_res.get("overall_vocal_emotion", "—")))}</div>
                </div>
            </div>
        </div>
        """
    )

    rows = []
    for seg in clip_res.get("timeline") or []:
        c_pct = f"{float(seg['confidence_pct']):.1f}%" if seg.get("confidence_pct") is not None else "—"
        rows.append(
            f"""
            <tr style="border-bottom:1px solid rgba(43,40,54,0.06);">
                <td class="aura-mono" style="padding:0.5rem;">{float(seg['timestamp_sec']):.1f}s</td>
                <td style="padding:0.5rem; font-weight:700;">{html.escape(str(seg['facial_emotion']))}</td>
                <td style="padding:0.5rem; font-weight:700;">{html.escape(str(seg['vocal_emotion']))}</td>
                <td style="padding:0.5rem; font-weight:800; color:{theme['accent']};">{html.escape(str(seg['combined_emotion']))}</td>
                <td class="aura-mono" style="padding:0.5rem;">{c_pct}</td>
                <td style="padding:0.5rem; font-size:0.72rem; color:#5E5A6E;">{html.escape(str(seg['facial_presentation']))}</td>
            </tr>
            """
        )

    render_html(
        f"""
        <div class="aura-surface-card">
            <div class="aura-section-label">
                <span>Emotion &amp; Presentation Timeline</span>
                <span class="aura-mono" style="font-size:0.68rem; color:#68637A;">Processed in {clip_res.get('processing_time_sec', 0.0):.2f}s</span>
            </div>
            <div style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.78rem;">
                    <thead>
                        <tr style="border-bottom:2px solid rgba(43,40,54,0.08); font-size:0.66rem; text-transform:uppercase; color:#68637A;">
                            <th style="padding:0.5rem;">Time</th>
                            <th style="padding:0.5rem;">Facial</th>
                            <th style="padding:0.5rem;">Vocal</th>
                            <th style="padding:0.5rem;">Combined</th>
                            <th style="padding:0.5rem;">Score</th>
                            <th style="padding:0.5rem;">Facial Presentation</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
        </div>
        """
    )
