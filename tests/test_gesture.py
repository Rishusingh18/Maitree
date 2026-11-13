"""Unit tests for Silent Distress Gesture detector."""
import pytest
import numpy as np
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.events.models import EventType
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState

@pytest.fixture
def gesture_env():
    cfg = {"platform": [[0,0],[640,0],[640,480],[0,480]]}
    zm = ZoneManager(cfg)
    detector = DistressGestureDetector(min_persistence=3, use_macro_fallback=True)
    return zm, detector

def test_wrists_crossed_overhead(gesture_env):
    zm, detector = gesture_env
    events = []
    # Feed 4 frames with wrists crossed above nose
    for f in range(4):
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[0] = [320, 150, 0.9]  # nose
        kpts[5] = [300, 200, 0.9]  # L sh
        kpts[6] = [340, 200, 0.9]  # R sh
        kpts[9] = [325, 120, 0.9]  # L wrist above head
        kpts[10] = [315, 120, 0.9] # R wrist above head
        kpts[15] = [310, 350, 0.9]
        kpts[16] = [330, 350, 0.9]
        det = Detection(box=np.array([290, 100, 350, 360]), confidence=0.9, keypoints=kpts, track_id=1)
        evs = detector.process_frame(f, f*0.04, [det], zm, TrainState.ABSENT)
        events.extend(evs)

    assert len(events) >= 1
    assert events[0].event_type == EventType.DISTRESS_GESTURE
    assert events[0].metadata["gesture_variant"] == "WRISTS_CROSSED_OVERHEAD"
