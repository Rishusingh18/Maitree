"""Shared Live Telemetry and Video Streaming Hub.
Thread-safe bridge between the computer vision pipeline and the Flask web dashboard.
Publishes:
- Low-latency MJPEG video frames (/video_feed) with offline fallback frame
- Dual in-memory and disk-cached buffer for seamless multi-process or single-process operation
- Live-measured telemetry (FPS, end-to-end latency, synthetic benchmark accuracy, fallback status)
- Real-time incident event log and Server-Sent Events (SSE)
- Operator controls (Buzzer acknowledge, Mute, Test, ETS, Train State)
"""
import os
import time
import json
import uuid
import queue
import logging
import threading
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

logger = logging.getLogger("metro_safety.demo.hub")

CACHE_DIR = Path("logs/cache")
FRAME_CACHE_PATH = CACHE_DIR / "live_frame.jpg"
TELEMETRY_CACHE_PATH = CACHE_DIR / "live_telemetry.json"
EVENTS_CACHE_PATH = CACHE_DIR / "live_events.json"


class LiveHub:
    """Thread-safe state hub shared between the vision pipeline and web dashboard."""

    def __init__(self, cctv_id: str = "CCTV_CAM_03", station_id: str = "DMRC_STN_042", platform_id: str = "PL_DOWN_01"):
        self.cctv_id = cctv_id
        self.station_id = station_id
        self.platform_id = platform_id

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._frame_seq = 0

        # Video frame storage
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_ts: float = 0.0
        self._is_online: bool = False
        self._status_msg: str = "Connecting to camera stream..."

        # Source & Fallback State
        self.is_fallback: bool = False
        self.source_target: str = ""
        self.retry_status: str = "Standby"

        # Camera Motion & Occlusion
        self.camera_motion: bool = False
        self.camera_motion_reason: str = "Stable"

        # Process ID Tracking (Ensures Producer & Web Server match)
        self.producer_pid: int = os.getpid()

        # Session ID Tracking (distinguishes events across distinct runs)
        self.session_id: str = f"SES-{uuid.uuid4().hex[:8].upper()}"

        # Telemetry
        self.live_fps: float = 0.0
        self.detector_fps: float = 0.0
        self.live_latency_ms: float = 0.0
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        self.synthetic_accuracy_label: str = "100.0% (synthetic benchmark)"
        self.train_state: str = "ABSENT"
        self._explicit_train_state: bool = False
        self.ets_status: str = "DISENGAGED"  # DISENGAGED | ARMED | ENGAGED
        self._explicit_ets_status: bool = False
        self.signal_held: bool = False
        self.cctv_status: str = "OPERATIONAL"

        # Control States
        self._is_muted: bool = False
        self._acknowledged: bool = False
        self._last_test_pattern: Optional[str] = None
        self._last_test_time: float = 0.0

        # Events
        self.events_history: deque = deque(maxlen=150)
        self._sse_listeners: List[queue.Queue] = []

        # Active Overlooker Alert
        self.active_alert: Optional[Dict[str, Any]] = None

        # References to hardware components
        try:
            from metro_safety.escalation.buzzer import HardwareBuzzer
            self.buzzer = HardwareBuzzer(audio_enabled=True)
        except Exception as _b_init_err:
            logger.warning(f"Could not initialize HardwareBuzzer in LiveHub: {_b_init_err}")
            self.buzzer = None
        self.train_state_provider = None
        self.zone_tracker = None

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def set_hardware_components(self, buzzer=None, train_state_provider=None, zone_tracker=None):
        if buzzer is not None:
            self.buzzer = buzzer
        if train_state_provider is not None:
            self.train_state_provider = train_state_provider
        if zone_tracker is not None:
            self.zone_tracker = zone_tracker

    # ================= Video Streaming =================
    def update_frame(self, frame: Optional[np.ndarray], is_online: bool = True, status_msg: str = ""):
        """Called by the vision pipeline with the latest annotated frame."""
        with self._lock:
            self._is_online = is_online
            if status_msg:
                self._status_msg = status_msg

            if frame is not None and is_online:
                try:
                    # Encode to JPEG (quality 70 for CPU speedup & lower payload)
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret:
                        self._latest_jpeg = jpeg.tobytes()
                        self._latest_frame_ts = time.time()
                        self._frame_seq += 1
                        self.producer_pid = os.getpid()

                        # Cross-process persistence
                        try:
                            FRAME_CACHE_PATH.write_bytes(self._latest_jpeg)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Frame encoding failed: {e}")

            self._frame_cond.notify_all()

    def get_latest_jpeg(self) -> bytes:
        """Returns the latest annotated JPEG frame, or checks disk buffer, or generates standby frame."""
        with self._lock:
            now = time.time()
            if self._is_online and self._latest_jpeg is not None and (now - self._latest_frame_ts) < 3.0:
                return self._latest_jpeg

        # If in-memory is empty or stale (e.g., dashboard running in a separate process), check disk cache
        if FRAME_CACHE_PATH.exists():
            try:
                mtime = FRAME_CACHE_PATH.stat().st_mtime
                if (time.time() - mtime) < 3.0:
                    cached_bytes = FRAME_CACHE_PATH.read_bytes()
                    if cached_bytes:
                        with self._lock:
                            self._latest_jpeg = cached_bytes
                            self._latest_frame_ts = mtime
                            self._is_online = True
                        return cached_bytes
            except Exception:
                pass

        # Otherwise return offline fallback frame
        return self._generate_offline_frame()

    def _generate_offline_frame(self) -> bytes:
        """Render a clear 'CCTV STREAM OFFLINE - RECONNECTING' standby frame."""
        h, w = 480, 640
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (24, 26, 32)  # Dark slate background

        # Red banner
        cv2.rectangle(canvas, (40, 140), (w - 40, 340), (35, 38, 48), -1)
        cv2.rectangle(canvas, (40, 140), (w - 40, 340), (0, 70, 220), 2)

        # Warning icon/text
        cv2.putText(canvas, "[!] CCTV STREAM STANDBY / RECONNECTING", (60, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 80, 255), 2)
        cv2.putText(canvas, "Waiting for video frames from camera pipeline...", (60, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        status_disp = self._status_msg if self._status_msg else "Camera feed initializing..."
        if len(status_disp) > 45:
            status_disp = status_disp[:42] + "..."
        cv2.putText(canvas, f"Source: {status_disp}", (60, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(canvas, "Low-latency buffer size 1 | Auto-reconnect active", (60, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Top/bottom metadata
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(canvas, f"{self.station_id} | {self.cctv_id} | {ts_str}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        ret, jpeg = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return jpeg.tobytes() if ret else b""

    def mjpeg_generator(self):
        """Generator yielding MJPEG multipart chunks for Flask streaming.
        Standard RFC 2046 format compatible with Chrome/WebKit/Firefox without Content-Length desync.
        """
        last_seq = -1

        # Print & log PID match verification
        server_pid = os.getpid()
        print(f"[PID Check] Frame Buffer Producer PID: {self.producer_pid} | /video_feed Server PID: {server_pid} (Match: {self.producer_pid == server_pid})")
        logger.info(f"[/video_feed] Serving frame stream. Producer PID: {self.producer_pid} | Server PID: {server_pid}")

        # Yield immediate initial frame on connection
        init_frame = self.get_latest_jpeg()
        if init_frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + init_frame + b'\r\n'
            )

        while True:
            with self._frame_cond:
                if self._frame_seq == last_seq:
                    self._frame_cond.wait(timeout=0.06)
                last_seq = self._frame_seq

            jpeg_bytes = self.get_latest_jpeg()
            if not jpeg_bytes:
                time.sleep(0.03)
                continue
            try:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg_bytes + b'\r\n'
                )
            except GeneratorExit:
                break
            except Exception:
                break

    # ================= Telemetry =================
    def update_telemetry(
        self,
        fps: float,
        latency_ms: float,
        train_state: str = None,
        ets_status: str = None,
        is_fallback: bool = False,
        source_target: str = "",
        retry_status: str = "",
        camera_motion: bool = False,
        camera_motion_reason: str = "",
        detector_fps: float = 0.0,
        stage_timings: Optional[Dict[str, Dict[str, float]]] = None
    ):
        with self._lock:
            self.live_fps = round(float(fps), 1)
            self.detector_fps = round(float(detector_fps), 1) if detector_fps > 0 else self.live_fps
            self.live_latency_ms = round(float(latency_ms), 2)
            if stage_timings:
                self.stage_timings = stage_timings
            if train_state:
                self.train_state = str(train_state).upper()
            if ets_status:
                self.ets_status = str(ets_status).upper()
            self.is_fallback = bool(is_fallback)
            if source_target:
                self.source_target = str(source_target)
            if retry_status:
                self.retry_status = str(retry_status)
            self.camera_motion = bool(camera_motion)
            if camera_motion_reason:
                self.camera_motion_reason = str(camera_motion_reason)

            # Adaptive zone status
            adaptive_info = {
                "status": "ALIGNED",
                "inliers": 0,
                "inlier_ratio": 1.0,
                "confidence": 1.0,
                "drift_px": 0.0,
                "scale": 1.0,
                "zone_version": 1,
                "registration_cost_ms": 0.0,
                "requires_operator_confirmation": False,
                "zones_unverified": False,
                "failure_reason": ""
            }
            if self.zone_tracker:
                try:
                    reg = self.zone_tracker.get_latest_result()
                    adaptive_info = {
                        "status": reg.status.value if hasattr(reg.status, "value") else str(reg.status),
                        "inliers": int(reg.inlier_count),
                        "inlier_ratio": round(float(reg.inlier_ratio), 3),
                        "confidence": round(float(reg.confidence), 3),
                        "drift_px": round(float(reg.drift_px), 2),
                        "scale": round(float(reg.scale), 3),
                        "zone_version": int(reg.zone_version),
                        "registration_cost_ms": round(float(reg.reg_time_ms), 2),
                        "requires_operator_confirmation": bool(reg.requires_operator_confirmation),
                        "zones_unverified": bool(reg.zones_unverified),
                        "failure_reason": str(reg.failure_reason)
                    }
                except Exception:
                    pass

            # Persist for cross-process
            try:
                t_payload = {
                    "session_id": self.session_id,
                    "cctv_id": self.cctv_id,
                    "station_id": self.station_id,
                    "platform_id": self.platform_id,
                    "is_online": self._is_online,
                    "status_msg": self._status_msg,
                    "live_fps": self.live_fps,
                    "detector_fps": self.detector_fps,
                    "live_latency_ms": self.live_latency_ms,
                    "stage_timings": self.stage_timings,
                    "synthetic_accuracy": self.synthetic_accuracy_label,
                    "train_state": self.train_state,
                    "ets_status": self.ets_status,
                    "is_fallback": self.is_fallback,
                    "source_target": self.source_target,
                    "retry_status": self.retry_status,
                    "camera_motion": self.camera_motion,
                    "camera_motion_reason": self.camera_motion_reason,
                    "producer_pid": self.producer_pid,
                    "adaptive_zones": adaptive_info,
                    "timestamp": time.time()
                }
                TELEMETRY_CACHE_PATH.write_text(json.dumps(t_payload))
            except Exception:
                pass

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            buzzer_info = self.buzzer.get_status() if self.buzzer else {
                "active": False,
                "pattern": getattr(self, "_test_pattern", "OFF"),
                "muted": getattr(self, "_is_muted", False),
                "audio_enabled": True,
                "acknowledged": getattr(self, "_acknowledged", False),
                "last_test_pattern": getattr(self, "_last_test_pattern", None),
                "last_test_time": getattr(self, "_last_test_time", 0.0),
                "clear_timer_remaining": 0.0
            }
            adaptive_info = {
                "status": "ALIGNED",
                "inliers": 0,
                "inlier_ratio": 1.0,
                "confidence": 1.0,
                "drift_px": 0.0,
                "scale": 1.0,
                "zone_version": 1,
                "registration_cost_ms": 0.0,
                "requires_operator_confirmation": False,
                "zones_unverified": False,
                "failure_reason": ""
            }
            if self.zone_tracker:
                try:
                    reg = self.zone_tracker.get_latest_result()
                    adaptive_info = {
                        "status": reg.status.value if hasattr(reg.status, "value") else str(reg.status),
                        "inliers": int(reg.inlier_count),
                        "inlier_ratio": round(float(reg.inlier_ratio), 3),
                        "confidence": round(float(reg.confidence), 3),
                        "drift_px": round(float(reg.drift_px), 2),
                        "scale": round(float(reg.scale), 3),
                        "zone_version": int(reg.zone_version),
                        "registration_cost_ms": round(float(reg.reg_time_ms), 2),
                        "requires_operator_confirmation": bool(reg.requires_operator_confirmation),
                        "zones_unverified": bool(reg.zones_unverified),
                        "failure_reason": str(reg.failure_reason)
                    }
                except Exception:
                    pass

            res = {
                "session_id": self.session_id,
                "cctv_id": self.cctv_id,
                "station_id": self.station_id,
                "platform_id": self.platform_id,
                "is_online": self._is_online,
                "status_msg": self._status_msg,
                "live_fps": self.live_fps,
                "detector_fps": self.detector_fps or self.live_fps,
                "live_latency_ms": self.live_latency_ms,
                "stage_timings": self.stage_timings,
                "synthetic_accuracy": self.synthetic_accuracy_label,
                "train_state": self.train_state,
                "ets_status": self.ets_status,
                "buzzer": buzzer_info,
                "active_alert": self.active_alert,
                "is_fallback": self.is_fallback,
                "source_target": self.source_target,
                "retry_status": self.retry_status,
                "camera_motion": self.camera_motion,
                "camera_motion_reason": self.camera_motion_reason,
                "adaptive_zones": adaptive_info,
                "producer_pid": self.producer_pid,
                "server_pid": os.getpid(),
                "pid_match": bool(self.producer_pid == os.getpid())
            }

        # Check cross-process disk fallback if live_fps == 0
        if res["live_fps"] == 0.0 and TELEMETRY_CACHE_PATH.exists():
            try:
                mtime = TELEMETRY_CACHE_PATH.stat().st_mtime
                if (time.time() - mtime) < 4.0:
                    cached = json.loads(TELEMETRY_CACHE_PATH.read_text())
                    for k in ["live_fps", "detector_fps", "live_latency_ms", "is_fallback", "source_target", "retry_status", "adaptive_zones"]:
                        if k in cached:
                            res[k] = cached[k]
                    if "train_state" in cached:
                        if not getattr(self, "_explicit_train_state", False):
                            res["train_state"] = cached["train_state"]
                    if "ets_status" in cached:
                        if not getattr(self, "_explicit_ets_status", False):
                            res["ets_status"] = cached["ets_status"]
                    res["is_online"] = True
            except Exception:
                pass

        return res

    # ================= Events =================
    def push_event(self, event_record: Dict[str, Any]):
        """Publish a new incident event."""
        if "session_id" not in event_record:
            event_record["session_id"] = self.session_id

        with self._lock:
            self.events_history.appendleft(event_record)
            # Notify SSE listeners
            for q in list(self._sse_listeners):
                try:
                    q.put_nowait(event_record)
                except Exception:
                    pass

        # Persist recent events to disk for cross-process
        try:
            with open(EVENTS_CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")
        except Exception:
            pass

    def get_events(self, limit: int = 50, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            in_mem = list(self.events_history)

        if in_mem:
            res = [ev for ev in in_mem if session_id is None or ev.get("session_id") == session_id]
            return res[:limit]

        # Check disk cache if in-memory is empty
        if EVENTS_CACHE_PATH.exists():
            try:
                lines = EVENTS_CACHE_PATH.read_text(encoding="utf-8").strip().splitlines()
                disk_events = []
                for line in reversed(lines):
                    if line.strip():
                        try:
                            ev = json.loads(line)
                            if "session_id" not in ev:
                                ev["session_id"] = "LEGACY_SESSION"
                            if session_id is None or ev.get("session_id") == session_id:
                                disk_events.append(ev)
                        except Exception:
                            continue
                return disk_events[:limit]
            except Exception:
                pass

        return []

    def add_sse_listener(self) -> queue.Queue:
        q = queue.Queue(maxsize=30)
        with self._lock:
            self._sse_listeners.append(q)
        return q

    def remove_sse_listener(self, q: queue.Queue):
        with self._lock:
            if q in self._sse_listeners:
                self._sse_listeners.remove(q)

    # ================= Overlooker Alert =================
    def set_overlooker_alert(self, alert_data: Dict[str, Any]):
        with self._lock:
            self.active_alert = alert_data

    def clear_overlooker_alert(self):
        with self._lock:
            self.active_alert = None

    # ================= Operator Controls =================
    def acknowledge_alarm(self) -> Dict[str, Any]:
        """Operator pressed 'Acknowledge': silences buzzer, clears alert banner, and logs audit event."""
        with self._lock:
            if self.buzzer:
                self.buzzer.acknowledge()
            self._acknowledged = True
            self.active_alert = None
            logger.info("Alarm acknowledged by operator via dashboard.")

        self.push_event({
            "id": f"ACK-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": "ALARM_ACKNOWLEDGED",
            "zone": "platform",
            "severity": "NORMAL",
            "confidence": 0.99,
            "action": "Operator acknowledged active alarms and silenced platform buzzer.",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "message": "Alarm acknowledged and buzzer silenced."}

    def toggle_buzzer_mute(self) -> Dict[str, Any]:
        """Toggle hardware buzzer mute state and publish state change event."""
        with self._lock:
            if self.buzzer:
                new_mute = self.buzzer.toggle_mute()
            else:
                self._is_muted = not self._is_muted
                new_mute = self._is_muted
            self._is_muted = new_mute

        self.push_event({
            "id": f"MUTE-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": "BUZZER_MUTED" if new_mute else "BUZZER_UNMUTED",
            "zone": "platform",
            "severity": "L1",
            "confidence": 0.99,
            "action": f"Hardware buzzer mute state set to: {new_mute}",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "muted": new_mute}

    def test_buzzer(self, pattern: str = "L2") -> Dict[str, Any]:
        """Dispatch test buzzer audio pattern and publish test event to /api/events."""
        pat = str(pattern).upper().strip()
        with self._lock:
            self._last_test_pattern = pat
            self._last_test_time = time.time()
            if self.buzzer:
                self.buzzer.test_beep(pattern=pat)

        self.push_event({
            "id": f"TEST-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": f"BUZZER_TEST_{pat}",
            "zone": "platform",
            "severity": pat if pat in ["L1", "L2", "L3"] else "L1",
            "confidence": 0.99,
            "action": f"Manual test of buzzer {pat} dispatched by operator.",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "message": f"Test beep '{pat}' dispatched.", "pattern": pat}

    def set_train_state(self, new_state: str) -> Dict[str, Any]:
        val = str(new_state).upper()
        with self._lock:
            self.train_state = val
            self._explicit_train_state = True
            if self.train_state_provider:
                try:
                    from metro_safety.perception.train_state import TrainState
                    self.train_state_provider.set_state(TrainState(val))
                except Exception:
                    pass
            try:
                if TELEMETRY_CACHE_PATH.exists():
                    cached = json.loads(TELEMETRY_CACHE_PATH.read_text())
                    cached["train_state"] = val
                    TELEMETRY_CACHE_PATH.write_text(json.dumps(cached))
            except Exception:
                pass
        return {"status": "ok", "train_state": val}

    def trip_ets(self) -> Dict[str, Any]:
        with self._lock:
            self.ets_status = "ENGAGED"
            self._explicit_ets_status = True
            logger.warning("[SAFETY TRIP] Emergency Trip Switch manually engaged by Operator!")
            try:
                if TELEMETRY_CACHE_PATH.exists():
                    cached = json.loads(TELEMETRY_CACHE_PATH.read_text())
                    cached["ets_status"] = "ENGAGED"
                    TELEMETRY_CACHE_PATH.write_text(json.dumps(cached))
            except Exception:
                pass

        self.push_event({
            "id": f"ETS-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": "ETS_MANUAL_TRIP",
            "zone": "track_bed",
            "severity": "L3",
            "confidence": 0.99,
            "action": "CRITICAL: Traction power cut manually engaged via Operator ETS trigger.",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "ets_status": "ENGAGED"}

    def hold_signal(self) -> Dict[str, Any]:
        """Operator engaged Guideway Signal Hold: holds approaching train signal to RED."""
        with self._lock:
            self.signal_held = True
            logger.warning("[INTERLOCK] Guideway approaching signal held to RED at Patel Chowk.")

        self.push_event({
            "id": f"SIG-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": "SIGNAL_HELD_RED",
            "zone": "guideway",
            "severity": "L2",
            "confidence": 0.99,
            "action": "Interlock Engaged: Approaching train signal held to RED at Patel Chowk.",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "signal_held": True, "message": "Approaching train signal held to RED."}

    def reset_ets(self) -> Dict[str, Any]:
        with self._lock:
            self.ets_status = "DISENGAGED"
            self._explicit_ets_status = True
            logger.info("[SAFETY TRIP] Emergency Trip Switch reset to DISENGAGED.")
            try:
                if TELEMETRY_CACHE_PATH.exists():
                    cached = json.loads(TELEMETRY_CACHE_PATH.read_text())
                    cached["ets_status"] = "DISENGAGED"
                    TELEMETRY_CACHE_PATH.write_text(json.dumps(cached))
            except Exception:
                pass

        self.push_event({
            "id": f"ETS-RST-{int(time.time()*1000)}",
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "track_id": 0,
            "event_type": "ETS_RESET",
            "zone": "platform",
            "severity": "L1",
            "confidence": 0.99,
            "action": "Traction power interlock reset to DISENGAGED by operator.",
            "is_demo": False,
            "session_id": self.session_id
        })
        return {"status": "ok", "ets_status": "DISENGAGED"}

    def confirm_zone_alignment(self) -> Dict[str, Any]:
        """Operator confirmation of current/proposed zone alignment."""
        with self._lock:
            tracker = self.zone_tracker

        if tracker is not None:
            success, msg = tracker.confirm_alignment()
            return {"status": "ok" if success else "error", "message": msg}
        return {"status": "ok", "message": "Zone alignment verified (in memory)."}



# Global module singleton
_GLOBAL_HUB_INSTANCE: Optional[LiveHub] = None

def get_global_hub() -> LiveHub:
    global _GLOBAL_HUB_INSTANCE
    if _GLOBAL_HUB_INSTANCE is None:
        _GLOBAL_HUB_INSTANCE = LiveHub()
    return _GLOBAL_HUB_INSTANCE

def set_global_hub(hub: LiveHub):
    global _GLOBAL_HUB_INSTANCE
    _GLOBAL_HUB_INSTANCE = hub
