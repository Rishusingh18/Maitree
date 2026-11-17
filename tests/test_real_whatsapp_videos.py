"""Integration test suite using real-world WhatsApp distress gesture video captures."""
import os
import cv2
import pytest
from pathlib import Path

from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState
from metro_safety.events.models import EventType

V1_PATH = Path("tests/WhatsApp Video 2026-11-15 at 11.02.40.mp4")
V2_PATH = Path("tests/WhatsApp Video 2026-11-15 at 11.02.41.mp4")

@pytest.fixture
def zm():
    return ZoneManager({"platform": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]})

@pytest.fixture
def pose_detector():
    return YOLOPoseDetector()

def test_whatsapp_video_1_distress_detected(pose_detector, zm):
    """Verify Video 1 triggers DISTRESS_GESTURE with HIGH severity on real CCTV/phone capture."""
    if not V1_PATH.exists():
        pytest.skip(f"Video file not found: {V1_PATH}")

    cap = cv2.VideoCapture(str(V1_PATH))
    detector = DistressGestureDetector(min_persistence=2)
    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = pose_detector.detect(frame)
        for d in dets:
            d.track_id = 1
        evs = detector.process_frame(frame_idx, frame_idx * 0.033, dets, zm, TrainState.ABSENT)
        events.extend(evs)
        frame_idx += 1
    cap.release()

    assert len(events) >= 1, f"Expected at least 1 distress gesture event in {V1_PATH.name}, got 0"
    assert events[0].event_type == EventType.DISTRESS_GESTURE
    assert events[0].metadata.get("severity") == "HIGH"
    assert events[0].metadata.get("gesture_variant") in ["ARMS_CROSSED_CHEST", "WRISTS_CROSSED_OVERHEAD"]

def test_whatsapp_video_2_distress_detected(pose_detector, zm):
    """Verify Video 2 triggers DISTRESS_GESTURE with HIGH severity on real CCTV/phone capture."""
    if not V2_PATH.exists():
        pytest.skip(f"Video file not found: {V2_PATH}")

    cap = cv2.VideoCapture(str(V2_PATH))
    detector = DistressGestureDetector(min_persistence=2)
    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = pose_detector.detect(frame)
        for d in dets:
            d.track_id = 1
        evs = detector.process_frame(frame_idx, frame_idx * 0.033, dets, zm, TrainState.ABSENT)
        events.extend(evs)
        frame_idx += 1
    cap.release()

    assert len(events) >= 1, f"Expected at least 1 distress gesture event in {V2_PATH.name}, got 0"
    assert events[0].event_type == EventType.DISTRESS_GESTURE
    assert events[0].metadata.get("severity") == "HIGH"
    assert events[0].metadata.get("gesture_variant") in ["ARMS_CROSSED_CHEST", "WRISTS_CROSSED_OVERHEAD"]
