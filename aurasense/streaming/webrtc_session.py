"""
Real-Time WebRTC Synchronized Webcam & Microphone Session Manager.

Audited Reliability, Memory, Privacy, and Lifecycle Guarantees:
1. Ephemeral In-Memory Processing Only:
   - Strictly bounded ring buffers (`maxlen=10` downscaled 640px video frames, `4.0s` 16 kHz float32 audio).
   - Zero frames, zero audio chunks, and zero personal predictions are ever written to disk.
   - Temporary frame/audio snapshots are explicitly discarded right after each inference cycle.
2. Non-Blocking Queue / Worker Architecture:
   - WebRTC callbacks never block on neural inference.
   - Uses a thread-safe `threading.Event` shutdown token + single-slot task queue (`queue.Queue(maxsize=1)`)
     so heavy inference never piles up or blocks the Streamlit main thread.
3. Clean Session Shutdown & Stream Interruption Recovery:
   - `shutdown()` and `reset_session()` cleanly cancel pending work, purge in-memory buffers,
     and reset state when a stream is stopped, interrupted, or restarted.
   - Detects partial device availability (e.g., webcam unavailable while mic streams, mic unavailable
     while webcam streams, or browser permission denied / stream timeout).
"""

from collections import deque
import logging
import queue
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import av
import cv2
import librosa
import numpy as np

from aurasense.config import LiveInferenceConfig, PASTEL_PALETTES
from aurasense.inference import (
    FusionRuntimeSettings,
    MultimodalEmotionFusionEngine,
    PretrainedModelRegistry,
    get_model_registry,
)

logger = logging.getLogger("AuraSense.WebRTCSession")


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (166, 111, 124)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


class LiveSessionCoordinator:
    """
    Thread-safe, memory-bounded coordinator connecting `streamlit-webrtc` callbacks,
    a single-slot background inference queue worker, and the live Streamlit UI.
    """

    def __init__(
        self,
        registry: Optional[PretrainedModelRegistry] = None,
        config: Optional[LiveInferenceConfig] = None,
    ) -> None:
        self.config = config or LiveInferenceConfig()
        self.registry = registry or get_model_registry(self.config)
        self.fusion_settings = FusionRuntimeSettings.from_config(self.config)
        self.fusion_engine = MultimodalEmotionFusionEngine(
            registry=self.registry,
            config=self.config,
            settings=self.fusion_settings,
        )

        self._buffer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._task_queue: "queue.Queue[Optional[Tuple[List[np.ndarray], np.ndarray]]]" = queue.Queue(
            maxsize=1
        )

        # Strictly bounded ephemeral ring buffers (discarded immediately on reset/shutdown)
        self._video_frames: Deque[np.ndarray] = deque(maxlen=self.config.max_video_buffer_frames)
        self._audio_samples_16k: Deque[np.ndarray] = deque()
        self._audio_total_samples: int = 0
        self._max_audio_samples: int = int(
            self.config.rolling_window_sec * self.config.target_sample_rate
        )

        # Live telemetry & hardware heartbeat state
        self._latest_result: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._audio_bars: List[int] = [12] * 24
        self._last_face_box: Optional[Tuple[int, int, int, int]] = None
        self._fast_face_detected: bool = False

        self._frame_counter: int = 0
        self._video_frames_received: int = 0
        self._audio_frames_received: int = 0
        self._last_video_rx_ts: float = 0.0
        self._last_audio_rx_ts: float = 0.0
        self._session_started_ts: float = 0.0
        self._completed_inferences: int = 0
        self._latency_history_ms: Deque[float] = deque(maxlen=25)
        self._face_latency_ms: float = 0.0
        self._voice_latency_ms: float = 0.0
        self._total_latency_ms: float = 0.0

        # Configurable runtime controls
        self.is_paused: bool = False
        self.mirror_video: bool = True
        self.show_face_overlay: bool = True
        self.inference_interval_sec: float = self.config.inference_interval_sec
        self.frame_sample_stride: int = self.config.frame_sample_stride

        self._is_inferring: bool = False
        self._last_inference_ts: float = 0.0

        # Dedicated background consumer thread reading from `_task_queue`
        self._worker_thread = threading.Thread(
            target=self._inference_worker_loop,
            name="AuraSense-InferenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def mark_session_started(self) -> None:
        """Records stream activation timestamp to detect missing/denied camera or mic tracks."""
        with self._state_lock:
            if self._session_started_ts == 0.0:
                self._session_started_ts = time.monotonic()

    def on_video_ended(self) -> None:
        """Callback invoked when browser webcam track stops or disconnects unexpectedly."""
        with self._buffer_lock:
            self._video_frames.clear()
        with self._state_lock:
            self._last_video_rx_ts = 0.0
            self._last_face_box = None
            self._fast_face_detected = False

    def on_audio_ended(self) -> None:
        """Callback invoked when browser microphone track stops or disconnects unexpectedly."""
        with self._buffer_lock:
            self._audio_samples_16k.clear()
            self._audio_total_samples = 0
        with self._state_lock:
            self._last_audio_rx_ts = 0.0
            self._audio_bars = [12] * 24

    def reset_session(self, clear_results: bool = True) -> None:
        """Purges all ephemeral in-memory audio/video buffers and resets smoothing state."""
        # Drain any pending queued task
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                break

        with self._buffer_lock:
            self._video_frames.clear()
            self._audio_samples_16k.clear()
            self._audio_total_samples = 0
            self._video_frames_received = 0
            self._audio_frames_received = 0

        with self._state_lock:
            self._audio_bars = [12] * 24
            self._last_face_box = None
            self._fast_face_detected = False
            self._last_video_rx_ts = 0.0
            self._last_audio_rx_ts = 0.0
            self._session_started_ts = 0.0
            self._last_error = None
            if clear_results:
                self._latest_result = None
                self._completed_inferences = 0
                self._latency_history_ms.clear()
                self._face_latency_ms = 0.0
                self._voice_latency_ms = 0.0
                self._total_latency_ms = 0.0

        self.fusion_engine.reset_smoothing()

    def shutdown(self) -> None:
        """Cleanly terminates background worker thread and purges all ephemeral memory."""
        self._shutdown_event.set()
        self.reset_session(clear_results=True)
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def set_paused(self, paused: bool) -> None:
        self.is_paused = bool(paused)

    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Downscales/buffers periodic webcam frames and enqueues non-blocking inference."""
        img = frame.to_ndarray(format="bgr24")
        if self.mirror_video:
            img = cv2.flip(img, 1)

        now = time.monotonic()
        with self._state_lock:
            self._last_video_rx_ts = now
            if self._session_started_ts == 0.0:
                self._session_started_ts = now
        self._frame_counter += 1

        stride = max(2, int(self.frame_sample_stride))
        if not self.is_paused:
            if self._frame_counter % stride == 1:
                # Cap stored buffer frame resolution at 640px width to strictly bound RAM usage
                h, w = img.shape[:2]
                if w > 640:
                    buffered_img = cv2.resize(img, (640, int(h * (640.0 / w))), interpolation=cv2.INTER_AREA)
                else:
                    buffered_img = img.copy()

                with self._buffer_lock:
                    self._video_frames.append(buffered_img)
                    self._video_frames_received += 1

                box = self.fusion_engine.face_analyzer.detect_face_fast(img)
                with self._state_lock:
                    if box is not None:
                        self._last_face_box = box
                        self._fast_face_detected = True
                    else:
                        self._fast_face_detected = False
                        if not (
                            self._latest_result
                            and self._latest_result.get("face_analysis", {}).get("available")
                        ):
                            self._last_face_box = None

            if (not self._is_inferring) and (
                now - self._last_inference_ts >= float(self.inference_interval_sec)
            ):
                self._enqueue_inference(now)

        annotated = self._draw_live_hud(img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def audio_frame_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Converts incoming microphone audio to 16 kHz mono float32 in a bounded ring buffer."""
        raw = frame.to_ndarray()
        sr = int(frame.sample_rate or 48000)
        now = time.monotonic()
        with self._state_lock:
            self._last_audio_rx_ts = now
            if self._session_started_ts == 0.0:
                self._session_started_ts = now

        if not self.is_paused:
            arr = np.asarray(raw)
            if np.issubdtype(arr.dtype, np.integer):
                max_int = float(np.iinfo(arr.dtype).max)
                wav = arr.astype(np.float32) / max_int
            else:
                wav = arr.astype(np.float32)

            if wav.ndim > 1:
                if wav.shape[0] <= 8 and wav.shape[-1] > wav.shape[0]:
                    wav = np.mean(wav, axis=0)
                else:
                    wav = np.mean(wav, axis=-1)
            wav = wav.reshape(-1)

            if wav.size > 0:
                if sr != self.config.target_sample_rate:
                    wav_16k = librosa.resample(
                        wav, orig_sr=sr, target_sr=self.config.target_sample_rate
                    ).astype(np.float32)
                else:
                    wav_16k = wav.astype(np.float32)

                bars = self._compute_spectrum_bars(wav_16k)

                with self._buffer_lock:
                    self._audio_samples_16k.append(wav_16k)
                    self._audio_total_samples += wav_16k.size
                    self._audio_frames_received += 1
                    while (
                        self._audio_total_samples > self._max_audio_samples
                        and self._audio_samples_16k
                    ):
                        popped = self._audio_samples_16k.popleft()
                        self._audio_total_samples -= popped.size

                with self._state_lock:
                    self._audio_bars = bars

            if (not self._is_inferring) and (
                now - self._last_inference_ts >= float(self.inference_interval_sec)
            ):
                self._enqueue_inference(now)

        silent_out = np.zeros_like(raw)
        out_frame = av.AudioFrame.from_ndarray(
            silent_out,
            format=frame.format.name,
            layout=frame.layout.name,
        )
        out_frame.sample_rate = frame.sample_rate
        if frame.pts is not None:
            out_frame.pts = frame.pts
        if frame.time_base is not None:
            out_frame.time_base = frame.time_base
        return out_frame

    def _compute_spectrum_bars(self, wav_16k: np.ndarray) -> List[int]:
        if wav_16k.size < 64:
            return [12] * 24
        spec = np.abs(np.fft.rfft(wav_16k * np.hanning(wav_16k.size)))
        if spec.size < 24:
            return [12] * 24
        usable = spec[1 : min(spec.size, 180)]
        splits = np.array_split(usable, 24)
        bars: List[int] = []
        for b in splits:
            mag = float(np.mean(b)) if b.size > 0 else 0.0
            h = int(np.clip(11 + np.sqrt(mag) * 15.0, 11, 36))
            bars.append(h)
        return bars

    def _enqueue_inference(self, now: float) -> None:
        if self._shutdown_event.is_set() or self._task_queue.full():
            return

        min_audio_samples = int(self.config.min_speech_window_sec * self.config.target_sample_rate)
        with self._buffer_lock:
            # Do not feed stale frames if webcam track disconnected (> 3.5s without frames)
            cam_fresh = (now - self._last_video_rx_ts) < 3.5
            mic_fresh = (now - self._last_audio_rx_ts) < 3.5

            frames_snapshot = list(self._video_frames) if cam_fresh else []
            audio_snapshot = (
                np.concatenate(list(self._audio_samples_16k))
                if (mic_fresh and self._audio_samples_16k)
                else np.zeros(0, dtype=np.float32)
            )
            if len(frames_snapshot) < 1 and audio_snapshot.size < min_audio_samples:
                return

        try:
            self._task_queue.put_nowait((frames_snapshot, audio_snapshot))
            self._last_inference_ts = now
        except queue.Full:
            pass

    def _inference_worker_loop(self) -> None:
        """Dedicated background queue consumer running heavy inference off the main & WebRTC threads."""
        while not self._shutdown_event.is_set():
            try:
                item = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._task_queue.task_done()
                break

            frames_snapshot, audio_snapshot = item
            self._is_inferring = True
            try:
                res = self.fusion_engine.analyze_live_window(
                    frames_bgr=frames_snapshot,
                    audio_waveform=audio_snapshot,
                    audio_sr=self.config.target_sample_rate,
                )
                face_reg = res.get("face_analysis", {}).get("diagnostics", {}).get("last_region")
                tot_ms = float(res.get("total_latency_ms", 0.0))
                f_ms = float(res.get("face_analysis", {}).get("latency_ms", 0.0))
                v_ms = float(res.get("voice_analysis", {}).get("latency_ms", 0.0))

                with self._state_lock:
                    if not self._shutdown_event.is_set():
                        self._latest_result = res
                        self._completed_inferences += 1
                        self._total_latency_ms = tot_ms
                        self._face_latency_ms = f_ms
                        self._voice_latency_ms = v_ms
                        self._latency_history_ms.append(tot_ms)
                        self._last_error = "; ".join(res["errors"]) if res.get("errors") else None

                        if face_reg:
                            self._last_face_box = (
                                int(face_reg["x"]),
                                int(face_reg["y"]),
                                int(face_reg["w"]),
                                int(face_reg["h"]),
                            )
                        elif not res.get("face_analysis", {}).get("available") and not self._fast_face_detected:
                            self._last_face_box = None
            except Exception as exc:
                err_str = f"Real-time inference pipeline error: {exc}"
                logger.exception(err_str)
                with self._state_lock:
                    self._last_error = err_str
            finally:
                # Explicitly release references to frame/audio arrays so GC reclaims RAM immediately
                del frames_snapshot
                del audio_snapshot
                self._is_inferring = False
                self._task_queue.task_done()

    def _draw_live_hud(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draws clean Material Design 3 pastel face tracking brackets and expressed emotion badge."""
        out = frame_bgr.copy()
        with self._state_lock:
            res = self._latest_result
            box = self._last_face_box
            inferring = self._is_inferring
            err = self._last_error

        theme_key = (res or {}).get("theme_key", "Neutral")
        palette = PASTEL_PALETTES.get(theme_key, PASTEL_PALETTES["Neutral"])
        accent_bgr = _hex_to_bgr(palette["accent"])

        if self.show_face_overlay and box is not None and not self.is_paused:
            x, y, w, h = box
            corner = max(16, min(w, h) // 6)
            thickness = 2
            cv2.line(out, (x, y), (x + corner, y), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x, y), (x, y + corner), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x + w, y), (x + w - corner, y), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x + w, y), (x + w, y + corner), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x, y + h), (x + corner, y + h), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x, y + h), (x, y + h - corner), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x + w, y + h), (x + w - corner, y + h), accent_bgr, thickness, cv2.LINE_AA)
            cv2.line(out, (x + w, y + h), (x + w, y + h - corner), accent_bgr, thickness, cv2.LINE_AA)

        if self.is_paused:
            hud_text = "Analysis Paused"
        elif err:
            hud_text = "Inference Error"
        elif res:
            dg = res.get("dual_facial_gender") or {}
            if not dg.get("is_uncertain", True) and dg.get("short_label"):
                g_part = f"Presentation: {dg['short_label']}"
            elif dg.get("status") in {"disagreement", "low_confidence", "quality_rejected"}:
                g_part = "Uncertain (Adjust Camera)"
            else:
                g_part = "No Face"
            emo_part = f" | {res['dominant_emotion']}" if res.get("available") else ""
            hud_text = f"{g_part}{emo_part}"
        elif inferring:
            hud_text = "Analyzing Face & Voice..."
        else:
            hud_text = "Live Stream Active"

        (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
        pad_x, pad_y = 14, 9
        x1, y1 = 16, 16
        x2, y2 = x1 + tw + pad_x * 2 + 14, y1 + th + pad_y * 2

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (252, 250, 248), -1)
        cv2.addWeighted(overlay, 0.86, out, 0.14, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), accent_bgr, 1, cv2.LINE_AA)
        cv2.circle(out, (x1 + 14, (y1 + y2) // 2), 4, accent_bgr, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            hud_text,
            (x1 + 26, y2 - pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (54, 40, 43),
            2,
            cv2.LINE_AA,
        )
        return out

    def get_snapshot(self, webrtc_playing: bool = False, desired_playing: Optional[bool] = None) -> Dict[str, Any]:
        """Thread-safe telemetry & diagnostics snapshot consumed by `@st.fragment` sections."""
        now = time.monotonic()
        with self._buffer_lock:
            buffered_frames = len(self._video_frames)
            buffered_audio_sec = round(
                self._audio_total_samples / float(self.config.target_sample_rate), 2
            )
            mem_bytes = sum(int(f.nbytes) for f in self._video_frames) + sum(
                int(a.nbytes) for a in self._audio_samples_16k
            )
            v_rx = self._video_frames_received
            a_rx = self._audio_frames_received

        with self._state_lock:
            res = self._latest_result
            # Treat camera/mic as connected if frames are actively arriving, even before a full Streamlit rerun
            vid_fresh = bool(self._last_video_rx_ts > 0.0 and (now - self._last_video_rx_ts < 3.5))
            aud_fresh = bool(self._last_audio_rx_ts > 0.0 and (now - self._last_audio_rx_ts < 3.5))
            cam_connected = bool(vid_fresh and (webrtc_playing or desired_playing is True or v_rx > 0))
            mic_connected = bool(aud_fresh and (webrtc_playing or desired_playing is True or a_rx > 0))
            stream_active = bool(webrtc_playing or cam_connected or mic_connected)

            # Detect missing/unavailable hardware, connecting state, or denied permissions
            device_alert: Optional[str] = None
            elapsed_since_start = (now - self._session_started_ts) if self._session_started_ts > 0.0 else 0.0
            if desired_playing is True and not stream_active:
                if elapsed_since_start > 10.0:
                    device_alert = (
                        "Connection Failed — WebRTC did not receive camera/microphone frames. "
                        "Check browser permissions, firewall/TURN settings, or click 'Select Camera / Mic'."
                    )
                elif elapsed_since_start > 2.0:
                    device_alert = (
                        "Camera Permission Required — Allow camera & microphone access in your browser "
                        "(or click 'Start Analysis' inside the video box)."
                    )
                else:
                    device_alert = "Connecting to camera and microphone..."
            elif stream_active and self._session_started_ts > 0.0 and elapsed_since_start > 3.5:
                if not cam_connected and mic_connected:
                    device_alert = "Webcam stream unavailable — continuing in vocal-only live mode."
                elif cam_connected and not mic_connected:
                    device_alert = "Microphone stream unavailable — continuing in facial-only live mode."
                elif not cam_connected and not mic_connected:
                    device_alert = (
                        "Connection Failed or stream interrupted. Check device connection or click Reset Session."
                    )

            df_face_ok = bool(
                cam_connected
                and res
                and res.get("face_analysis", {}).get("face_detected", res.get("face_analysis", {}).get("available"))
            )
            face_detected = bool(cam_connected and (df_face_ok or self._fast_face_detected))
            voice_detected = bool(
                mic_connected and res and res.get("voice_analysis", {}).get("available")
            )

            f_diag = (res or {}).get("face_analysis", {}).get("diagnostics") or {}
            if not stream_active:
                if desired_playing is True:
                    if elapsed_since_start > 10.0:
                        tracking_status = "Connection Failed"
                    elif elapsed_since_start > 2.0:
                        tracking_status = "Camera Permission Required"
                    else:
                        tracking_status = "Connecting..."
                else:
                    tracking_status = "Camera Standby"
            elif not cam_connected:
                tracking_status = "Webcam Unavailable"
            elif self.is_paused:
                tracking_status = "Tracking Paused"
            elif df_face_ok:
                if f_diag.get("rapid_motion_or_blur"):
                    tracking_status = "Tracking · Rapid Motion / Blur"
                elif f_diag.get("low_lighting"):
                    tracking_status = "Tracking · Low Light (CLAHE)"
                elif f_diag.get("multiple_faces"):
                    tracking_status = f"Primary Face ({f_diag.get('detected_face_count', 2)} in view)"
                elif not f_diag.get("passed_gender_gate", True):
                    tracking_status = "Tracking · Adjust Camera Position"
                else:
                    tracking_status = "Locked · SCRFD 5-pt Aligned"
            elif self._fast_face_detected:
                tracking_status = "Tracking Face · Acquiring Expression"
            else:
                tracking_status = "Searching for Face..."

            if not webrtc_playing:
                inference_status = "Standby"
            elif self.is_paused:
                inference_status = "Paused"
            elif self._last_error:
                inference_status = "Error"
            elif self._is_inferring:
                inference_status = "Running Inference"
            elif self._completed_inferences > 0:
                inference_status = f"Active ({self._completed_inferences} cycles)"
            else:
                inference_status = "Buffering Initial Window"

            avg_latency_ms = (
                round(float(np.mean(self._latency_history_ms)), 1)
                if self._latency_history_ms
                else 0.0
            )

            return {
                "result": res,
                "last_error": self._last_error,
                "device_alert": device_alert,
                "audio_bars": list(self._audio_bars) if mic_connected and not self.is_paused else [12] * 24,
                "is_inferring": self._is_inferring,
                "is_paused": self.is_paused,
                "buffered_frames": buffered_frames,
                "buffered_audio_sec": buffered_audio_sec,
                "buffer_memory_mb": round(mem_bytes / (1024.0 * 1024.0), 2),
                "rolling_window_target_sec": self.config.rolling_window_sec,
                "camera_connected": cam_connected,
                "microphone_connected": mic_connected,
                "face_detected": face_detected,
                "face_tracking_status": tracking_status,
                "voice_detected": voice_detected,
                "inference_status": inference_status,
                "completed_inferences": self._completed_inferences,
                "face_latency_ms": self._face_latency_ms,
                "voice_latency_ms": self._voice_latency_ms,
                "total_latency_ms": self._total_latency_ms,
                "avg_latency_ms": avg_latency_ms,
                "video_frames_received": v_rx,
                "audio_frames_received": a_rx,
                "last_video_rx_age_s": round(now - self._last_video_rx_ts, 1) if self._last_video_rx_ts > 0.0 else None,
                "last_audio_rx_age_s": round(now - self._last_audio_rx_ts, 1) if self._last_audio_rx_ts > 0.0 else None,
            }
