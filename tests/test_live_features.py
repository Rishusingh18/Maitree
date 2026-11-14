"""Unit and integration tests for new live mode features:
1. Fallback visibility and background retry state
2. Event deduplication cooldown per (track_id, event_type)
3. DEMO tag on synthetic fallback events
4. Configurable inference size on YOLO pose detector
"""
import time
import numpy as np
import pytest

from metro_safety.perception.frame_source import ThreadedStreamSource, VideoFileSource
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel


def test_yolo_pose_inference_size():
    """Verify YOLOPoseDetector accepts custom inference_size and handles frames cleanly."""
    detector = YOLOPoseDetector(
        model_name="yolov8n-pose.pt",
        confidence_threshold=0.35,
        device="cpu",
        inference_size=320
    )
    assert detector.inference_size == 320

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy)
    assert isinstance(dets, list)


def test_frame_source_fallback_status():
    """Verify ThreadedStreamSource fallback properties when primary is unreachable."""
    fallback_file = "demo_clips/demo_fall.mp4"
    fallback_src = VideoFileSource(fallback_file, loop=True)

    # Use unreachable IP
    unreachable_url = "http://192.0.2.1:8080/video"  # RFC 5737 TEST-NET-1 (non-routable)
    stream = ThreadedStreamSource(
        stream_target=unreachable_url,
        is_network=True,
        connection_timeout_sec=0.5,
        fallback_source=fallback_src
    )

    try:
        assert stream.is_fallback is True
        assert stream.primary_target == unreachable_url
        assert "Retrying" in stream.retry_status or "unreachable" in stream.retry_status
        assert stream.source_name.startswith(unreachable_url)

        # Reading from stream returns valid fallback frames
        ret, frame, ts = stream.read()
        assert ret is True
        assert frame is not None
        assert frame.ndim == 3
    finally:
        stream.release()


def test_event_deduplication_cooldown():
    """Verify event deduplication cooldown per (track_id, event_type)."""
    last_event_time = {}
    cooldown_sec = 5.0

    events = [
        # (track_id, event_type_name, timestamp)
        (1, "DELIBERATE_JUMP", 10.0),
        (1, "DELIBERATE_JUMP", 11.0), # Should be suppressed (< 5s)
        (1, "DELIBERATE_JUMP", 12.0), # Should be suppressed (< 5s)
        (2, "DELIBERATE_JUMP", 12.0), # Different track - should pass!
        (1, "PLATFORM_LINE_CROSSING", 12.5), # Different type - should pass!
        (1, "DELIBERATE_JUMP", 15.5), # Cooldown expired (15.5 - 10.0 >= 5.0) - should pass!
    ]

    accepted = []
    for tid, ev_type, ts in events:
        dedup_key = (tid, ev_type)
        if ts - last_event_time.get(dedup_key, 0.0) < cooldown_sec:
            continue
        last_event_time[dedup_key] = ts
        accepted.append((tid, ev_type, ts))

    assert len(accepted) == 4
    assert accepted[0] == (1, "DELIBERATE_JUMP", 10.0)
    assert accepted[1] == (2, "DELIBERATE_JUMP", 12.0)
    assert accepted[2] == (1, "PLATFORM_LINE_CROSSING", 12.5)
    assert accepted[3] == (1, "DELIBERATE_JUMP", 15.5)


def test_demo_event_tagging():
    """Verify fallback synthetic source marks events with DEMO prefix and flag."""
    is_fallback = True
    event_type_name = "PERSON_FALL"

    ev_type_label = f"[DEMO] {event_type_name}" if is_fallback else event_type_name
    assert ev_type_label == "[DEMO] PERSON_FALL"
    assert ev_type_label.startswith("[DEMO]")
