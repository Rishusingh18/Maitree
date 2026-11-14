"""FrameSource abstraction and implementations:
- VideoFileSource: Reads video from file path with optional looping
- WebcamSource: Reads from local device index (e.g. 0, 1)
- NetworkStreamSource: Reads from HTTP MJPEG / RTSP streams (e.g. phone IP camera apps)
Features:
- Threaded background frame grabber keeping strictly the latest frame (buffer size 1) to eliminate latency
- Auto-reconnect with exponential backoff on dropped streams
- Logging of reconnect attempts and stream health
- Seamless fallback to pre-recorded video or synthetic clip if network source is unavailable
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Union
import threading
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger("metro_safety.perception.frame_source")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [FrameSource] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FrameSource(ABC):
    """Abstract interface for video frame sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns (success, frame, timestamp_sec)."""
        pass

    @abstractmethod
    def release(self):
        """Release underlying camera or stream resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is operational."""
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name/Identifier of the source."""
        pass


class VideoFileSource(FrameSource):
    """Reads frames from a local video file with optional looping."""

    def __init__(self, file_path: Union[str, Path], loop: bool = True):
        self.file_path = Path(file_path)
        self.loop = loop
        self._name = str(self.file_path)
        self.cap = cv2.VideoCapture(str(self.file_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_idx = 0

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_opened():
            return False, None, 0.0

        ret, frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            self.frame_idx = 0

        if not ret or frame is None:
            return False, None, 0.0

        self.frame_idx += 1
        ts = self.frame_idx / float(self.fps)
        return True, frame, ts

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def source_name(self) -> str:
        return self._name


class ThreadedStreamSource(FrameSource):
    """Low-latency threaded stream reader for Webcams and Network (HTTP/RTSP) streams.
    Always preserves only the newest frame (buffer size 1) and drops stale frames.
    Includes auto-reconnect with exponential backoff on network dropouts.
    """

    def __init__(
        self,
        stream_target: Union[str, int],
        is_network: bool = False,
        reconnect_interval_base: float = 1.0,
        max_reconnect_interval: float = 15.0,
        connection_timeout_sec: float = 8.0,
        fallback_source: Optional[FrameSource] = None
    ):
        self.stream_target = stream_target
        self.is_network = is_network
        self.base_backoff = reconnect_interval_base
        self.max_backoff = max_reconnect_interval
        self.conn_timeout = connection_timeout_sec
        self.fallback = fallback_source
        self.using_fallback = False

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_timestamp: float = 0.0
        self.frame_count: int = 0
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.reconnect_count = 0
        self._connected = False

        # Attempt initial connection
        self._init_capture()

    def _init_capture(self) -> bool:
        target_display = self.stream_target
        logger.info(f"Opening capture source: {target_display}")
        try:
            if isinstance(self.stream_target, int):
                # Webcam
                self.cap = cv2.VideoCapture(self.stream_target, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else cv2.CAP_ANY)
            else:
                target_str = str(self.stream_target)
                if self.is_network:
                    # Optimize latency for network streams
                    self.cap = cv2.VideoCapture(target_str, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    self.cap = cv2.VideoCapture(target_str)

            if self.cap and self.cap.isOpened():
                # Read 1 frame to verify real stream
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_timestamp = time.time()
                    self._connected = True
                    self._start_worker()
                    logger.info(f"Successfully connected to {target_display}")
                    return True
        except Exception as e:
            logger.warning(f"Failed to connect to {target_display}: {e}")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self._connected = False

        if self.fallback is not None:
            logger.warning(f"Primary source {target_display} unreachable. Engaging fallback source: {self.fallback.source_name}")
            self.using_fallback = True
            return True

        # If network stream without fallback, start worker anyway to attempt auto-reconnect in background
        if self.is_network:
            logger.info("Starting background auto-reconnect worker...")
            self._start_worker()

        return False

    def _start_worker(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        backoff = self.base_backoff
        while self.running:
            if self.using_fallback:
                time.sleep(0.05)
                continue

            if self.cap is None or not self.cap.isOpened():
                self.reconnect_count += 1
                logger.info(f"Reconnecting to {self.stream_target} (attempt #{self.reconnect_count}, backoff: {backoff:.1f}s)...")
                time.sleep(backoff)
                backoff = min(self.max_backoff, backoff * 1.5)
                try:
                    target_str = str(self.stream_target)
                    new_cap = cv2.VideoCapture(target_str, cv2.CAP_FFMPEG if self.is_network else cv2.CAP_ANY)
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if new_cap.isOpened():
                        ret, frame = new_cap.read()
                        if ret and frame is not None:
                            self.cap = new_cap
                            self._connected = True
                            backoff = self.base_backoff
                            logger.info(f"Reconnected successfully to {self.stream_target}!")
                            continue
                    new_cap.release()
                except Exception as err:
                    logger.warning(f"Reconnect error: {err}")
                continue

            # Read frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Stream dropped from {self.stream_target}. Initiating reconnect...")
                self._connected = False
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

                # Fallback to secondary source if available
                if self.fallback is not None:
                    logger.warning(f"Stream lost. Falling back to {self.fallback.source_name}")
                    self.using_fallback = True
                continue

            # Keep strictly latest frame (drop older frames to guarantee 0 buffer latency)
            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.frame_count += 1
            # Sleep very briefly to avoid spinning CPU
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.using_fallback and self.fallback is not None:
            return self.fallback.read()

        with self.lock:
            if self.latest_frame is None:
                return False, None, 0.0
            # Return copy of latest frame
            return True, self.latest_frame.copy(), self.latest_timestamp

    def is_opened(self) -> bool:
        if self.using_fallback and self.fallback is not None:
            return self.fallback.is_opened()
        return self._connected or self.running

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.fallback is not None:
            self.fallback.release()
        self._connected = False

    @property
    def source_name(self) -> str:
        if self.using_fallback and self.fallback is not None:
            return f"{self.stream_target} (FALLBACK: {self.fallback.source_name})"
        return str(self.stream_target)


def create_frame_source(
    source_uri: Union[str, int],
    fallback_path: Optional[Union[str, Path]] = "demo_clips/demo_fall.mp4",
    allow_fallback: bool = True,
    loop_video: bool = False
) -> FrameSource:
    """Factory creating appropriate FrameSource based on source URI format.
    - Integer: Webcam index (0, 1, ...)
    - 'http://' or 'rtsp://': Network stream (phone IP camera app)
    - File path: Video file
    """
    fallback_src = None
    if allow_fallback and fallback_path and Path(fallback_path).exists():
        fallback_src = VideoFileSource(fallback_path, loop=True)

    # Check for webcam index
    if isinstance(source_uri, int) or (isinstance(source_uri, str) and source_uri.strip().isdigit()):
        cam_idx = int(source_uri)
        return ThreadedStreamSource(
            stream_target=cam_idx,
            is_network=False,
            fallback_source=fallback_src
        )

    uri_str = str(source_uri).strip()
    if uri_str.lower().startswith("http://") or uri_str.lower().startswith("https://") or uri_str.lower().startswith("rtsp://"):
        return ThreadedStreamSource(
            stream_target=uri_str,
            is_network=True,
            fallback_source=fallback_src
        )

    # Local file path or pre-recorded clip
    path = Path(uri_str)
    if path.exists():
        return VideoFileSource(path, loop=loop_video)

    # Check known demo aliases
    alias_map = {
        "fall": "demo_clips/demo_fall.mp4",
        "jump": "demo_clips/demo_jump.mp4",
        "distress": "demo_clips/demo_distress.mp4",
        "synthetic": "demo_clips/demo_fall.mp4"
    }
    if uri_str.lower() in alias_map and Path(alias_map[uri_str.lower()]).exists():
        return VideoFileSource(alias_map[uri_str.lower()], loop=loop_video)

    # If file not found and fallback enabled
    if fallback_src:
        logger.warning(f"Specified source '{uri_str}' not found. Falling back to {fallback_src.source_name}")
        return fallback_src

    raise FileNotFoundError(f"Could not open source: {uri_str}")

