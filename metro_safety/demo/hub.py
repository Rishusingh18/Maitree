"""Shared Live Telemetry and Video Streaming Hub.
Thread-safe bridge between the computer vision pipeline and the Flask web dashboard.
Publishes:
- Low-latency MJPEG video frames (/video_feed) with offline fallback frame
- Live-measured telemetry (FPS, end-to-end latency, synthetic benchmark accuracy)
- Real-time incident event log and Server-Sent Events (SSE)
- Operator controls (Buzzer acknowledge, Mute, Test, ETS, Train State)
"""
import time
import queue
import logging
import threading
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

logger = logging.getLogger("metro_safety.demo.hub")


class LiveHub:
    """Thread-safe state hub shared between the vision pipeline and web dashboard."""

    def __init__(self, cctv_id: str = "CCTV_CAM_03", station_id: str = "DMRC_STN_042", platform_id: str = "PL_DOWN_01"):
        self.cctv_id = cctv_id
        self.station_id = station_id
        self.platform_id = platform_id

        self._lock = threading.Lock()

        # Video frame storage
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_ts: float = 0.0
        self._is_online: bool = False
        self._status_msg: str = "Connecting to camera stream..."
        self._frame_event = threading.Event()

        # Telemetry
        self.live_fps: float = 0.0
        self.live_latency_ms: float = 0.0
        self.synthetic_accuracy_label: str = "100.0% (synthetic benchmark)"
        self.train_state: str = "ABSENT"
        self.ets_status: str = "DISENGAGED"  # DISENGAGED | ARMED | ENGAGED
        self.cctv_status: str = "OPERATIONAL"

        # Events
        self.events_history: deque = deque(maxlen=150)
        self._sse_listeners: List[queue.Queue] = []

        # Active Overlooker Alert
        self.active_alert: Optional[Dict[str, Any]] = None

        # References to hardware components
        self.buzzer = None
        self.train_state_provider = None

    def set_hardware_components(self, buzzer=None, train_state_provider=None):
        self.buzzer = buzzer
        self.train_state_provider = train_state_provider

    # ================= Video Streaming =================
    def update_frame(self, frame: Optional[np.ndarray], is_online: bool = True, status_msg: str = ""):
        """Called by the vision pipeline with the latest annotated frame."""
        with self._lock:
            self._is_online = is_online
            if status_msg:
                self._status_msg = status_msg

            if frame is not None and is_online:
                try:
                    # Encode to JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ret:
                        self._latest_jpeg = jpeg.tobytes()
                        self._latest_frame_ts = time.time()
                except Exception as e:
                    logger.debug(f"Frame encoding failed: {e}")
            self._frame_event.set()

    def get_latest_jpeg(self) -> bytes:
        """Returns the latest annotated JPEG frame, or generates an offline standby frame."""
        with self._lock:
            now = time.time()
            if self._is_online and self._latest_jpeg is not None and (now - self._latest_frame_ts) < 2.5:
                return self._latest_jpeg

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
        """Generator yielding MJPEG multipart chunks for Flask streaming."""
        while True:
            # Wait for next frame event or timeout to emit latest frame
            self._frame_event.wait(timeout=0.06)
            self._frame_event.clear()
            jpeg_bytes = self.get_latest_jpeg()
            if not jpeg_bytes:
                time.sleep(0.04)
                continue
            try:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(jpeg_bytes)).encode('ascii') + b'\r\n\r\n'
                    + jpeg_bytes + b'\r\n'
                )
            except GeneratorExit:
                break
            except Exception:
                break

    # ================= Telemetry =================
    def update_telemetry(self, fps: float, latency_ms: float, train_state: str = None, ets_status: str = None):
        with self._lock:
            self.live_fps = round(float(fps), 1)
            self.live_latency_ms = round(float(latency_ms), 2)
            if train_state:
                self.train_state = str(train_state).upper()
            if ets_status:
                self.ets_status = str(ets_status).upper()

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            buzzer_info = self.buzzer.get_status() if self.buzzer else {
                "active": False, "pattern": "OFF", "muted": False, "audio_enabled": True
            }
            return {
                "cctv_id": self.cctv_id,
                "station_id": self.station_id,
                "platform_id": self.platform_id,
                "is_online": self._is_online,
                "status_msg": self._status_msg,
                "live_fps": self.live_fps,
                "live_latency_ms": self.live_latency_ms,
                "synthetic_accuracy": self.synthetic_accuracy_label,
                "train_state": self.train_state,
                "ets_status": self.ets_status,
                "buzzer": buzzer_info,
                "active_alert": self.active_alert
            }

    # ================= Events =================
    def push_event(self, event_record: Dict[str, Any]):
        """Publish a new incident event."""
        with self._lock:
            self.events_history.appendleft(event_record)
            # Notify SSE listeners
            for q in list(self._sse_listeners):
                try:
                    q.put_nowait(event_record)
                except Exception:
                    pass

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events_history)[:limit]

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
        """Operator pressed 'Acknowledge': silences buzzer and clears alert banner."""
        with self._lock:
            if self.buzzer:
                self.buzzer.acknowledge()
            self.active_alert = None
            logger.info("Alarm acknowledged by operator via dashboard.")
            return {"status": "ok", "message": "Alarm acknowledged and buzzer silenced."}

    def toggle_buzzer_mute(self) -> Dict[str, Any]:
        if self.buzzer:
            new_mute = self.buzzer.toggle_mute()
            return {"status": "ok", "muted": new_mute}
        return {"status": "error", "message": "Buzzer not initialized"}

    def test_buzzer(self, pattern: str = "L2") -> Dict[str, Any]:
        if self.buzzer:
            self.buzzer.test_beep(pattern=pattern)
            return {"status": "ok", "message": f"Test beep '{pattern}' dispatched."}
        return {"status": "error", "message": "Buzzer not initialized"}

    def set_train_state(self, new_state: str) -> Dict[str, Any]:
        val = str(new_state).upper()
        with self._lock:
            self.train_state = val
            if self.train_state_provider:
                try:
                    from metro_safety.perception.train_state import TrainState
                    self.train_state_provider.set_state(TrainState(val))
                except Exception:
                    pass
        return {"status": "ok", "train_state": val}

    def trip_ets(self) -> Dict[str, Any]:
        with self._lock:
            self.ets_status = "ENGAGED"
            logger.warning("[SAFETY TRIP] Emergency Trip Switch manually engaged by Operator!")
            return {"status": "ok", "ets_status": "ENGAGED"}

    def reset_ets(self) -> Dict[str, Any]:
        with self._lock:
            self.ets_status = "DISENGAGED"
            logger.info("[SAFETY TRIP] Emergency Trip Switch reset to DISENGAGED.")
            return {"status": "ok", "ets_status": "DISENGAGED"}

