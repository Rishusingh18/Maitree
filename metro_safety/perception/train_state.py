"""Train state provider interfaces and implementations."""
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import numpy as np
import cv2

class TrainState(str, Enum):
    ABSENT = "ABSENT"
    APPROACHING = "APPROACHING"
    PRESENT = "PRESENT"

class BaseTrainStateProvider(ABC):
    """Abstract train state provider interface."""

    @abstractmethod
    def get_state(self, timestamp: Optional[float] = None, frame: Optional[np.ndarray] = None) -> TrainState:
        """Returns the current train state."""
        pass

class SimulatedTrainStateProvider(BaseTrainStateProvider):
    """Provider for programmatic simulation or manual state setting."""

    def __init__(self, initial_state: TrainState = TrainState.ABSENT):
        self._current_state = initial_state

    def set_state(self, state: TrainState):
        self._current_state = state

    def get_state(self, timestamp: Optional[float] = None, frame: Optional[np.ndarray] = None) -> TrainState:
        return self._current_state

class ScheduleFileTrainStateProvider(BaseTrainStateProvider):
    """Provider reading train arrival / departure timetable intervals."""

    def __init__(self, schedule_file: Path | str):
        self.schedule_file = Path(schedule_file)
        self.intervals = []
        self._load_schedule()

    def _load_schedule(self):
        if not self.schedule_file.exists():
            return
        with open(self.schedule_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # data format: [{"approaching": [10.0, 20.0], "present": [20.0, 45.0]}, ...]
            self.intervals = data.get("train_events", [])

    def get_state(self, timestamp: Optional[float] = None, frame: Optional[np.ndarray] = None) -> TrainState:
        t = timestamp if timestamp is not None else time.time()
        for item in self.intervals:
            app = item.get("approaching")
            pres = item.get("present")
            if pres and pres[0] <= t <= pres[1]:
                return TrainState.PRESENT
            if app and app[0] <= t <= app[1]:
                return TrainState.APPROACHING
        return TrainState.ABSENT

class VisionTrainStateProvider(BaseTrainStateProvider):
    """Vision-based train presence detector using motion / large foreground blob in track bed."""

    def __init__(self, track_polygon: np.ndarray, area_threshold: float = 0.25):
        self.track_polygon = track_polygon
        self.area_thresh = area_threshold
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        self.prev_gray = None
        self.state_history = []

    def get_state(self, timestamp: Optional[float] = None, frame: Optional[np.ndarray] = None) -> TrainState:
        if frame is None:
            return TrainState.ABSENT

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mask to track zone
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.track_polygon], 255)
        track_area = max(1.0, float(np.sum(mask > 0)))

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        fg_track = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
        motion_ratio = float(np.sum(fg_track > 0)) / track_area

        if motion_ratio > self.area_thresh:
            curr_state = TrainState.PRESENT
        elif motion_ratio > (self.area_thresh * 0.4):
            curr_state = TrainState.APPROACHING
        else:
            curr_state = TrainState.ABSENT

        self.state_history.append(curr_state)
        if len(self.state_history) > 10:
            self.state_history.pop(0)

        # Majority vote over history for temporal smoothing
        from collections import Counter
        counts = Counter(self.state_history)
        return counts.most_common(1)[0][0]
