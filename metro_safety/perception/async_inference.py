"""Asynchronous threaded inference engine for YOLO pose estimation.
Runs heavy deep neural network inference on a dedicated background worker thread,
allowing the main video capture, tracking, escalation, rendering, and web streaming loops
to execute at full camera frame rate (25-30+ FPS) without CPU stalls.
"""
import time
import logging
import threading
from typing import List, Optional, Tuple
import numpy as np

from metro_safety.perception.base import Detection
from metro_safety.perception.yolo_pose import YOLOPoseDetector

logger = logging.getLogger("metro_safety.perception.async_inference")


class AsyncInferenceWorker:
    """Threaded pose inference worker decoupling model execution from video rendering."""

    def __init__(self, detector: Optional[YOLOPoseDetector], frame_skip: int = 1):
        self.detector = detector
        self.frame_skip = max(1, frame_skip)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_frame_idx: int = 0
        self._latest_detections: List[Detection] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Honest telemetry
        self.inference_fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self._infer_history: List[float] = []
        self._fps_history: List[float] = []
        self._last_infer_time: float = time.perf_counter()

    def start(self):
        """Start the background inference worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="YOLOInferenceWorker")
        self._thread.start()
        logger.info("[AsyncInference] Background pose inference worker thread started.")

    def stop(self):
        """Stop the background inference worker thread."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("[AsyncInference] Inference worker stopped.")

    def submit_frame(self, frame: np.ndarray, frame_idx: int):
        """Submit the latest video frame for background inference.
        Non-blocking: if a frame is already queued, replaces it with the newer frame.
        """
        if not self._running:
            return
        if frame_idx % self.frame_skip != 0:
            return

        with self._cond:
            self._pending_frame = frame.copy()
            self._pending_frame_idx = frame_idx
            self._cond.notify()

    def get_latest_detections(self) -> List[Detection]:
        """Fetch the latest available detections in O(1) time without blocking."""
        with self._lock:
            return list(self._latest_detections)

    def get_telemetry(self) -> Tuple[float, float]:
        """Returns measured (inference_fps, inference_latency_ms)."""
        with self._lock:
            return self.inference_fps, self.inference_latency_ms

    def _worker_loop(self):
        while self._running:
            with self._cond:
                while self._pending_frame is None and self._running:
                    self._cond.wait(timeout=0.04)
                if not self._running:
                    break
                frame_to_process = self._pending_frame
                self._pending_frame = None

            if frame_to_process is None:
                continue

            t0 = time.perf_counter()
            dets: List[Detection] = []
            try:
                if self.detector and self.detector._model is not None:
                    dets = self.detector.detect(frame_to_process)
            except Exception as e:
                logger.exception(f"[AsyncInference] Inference execution error: {e}")

            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0
            dt = t1 - self._last_infer_time
            self._last_infer_time = t1

            with self._lock:
                self._latest_detections = dets
                self._infer_history.append(lat)
                if len(self._infer_history) > 20:
                    self._infer_history.pop(0)
                if dt > 0:
                    self._fps_history.append(1.0 / dt)
                    if len(self._fps_history) > 20:
                        self._fps_history.pop(0)

                self.inference_latency_ms = float(np.mean(self._infer_history))
                self.inference_fps = float(np.mean(self._fps_history))

