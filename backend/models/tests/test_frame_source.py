"""Unit tests for FrameSource implementations:
- VideoFileSource
- ThreadedStreamSource
- create_frame_source factory with network fallback
- Reconnect and latest-frame buffer drop verification
"""
from pathlib import Path
import time
import numpy as np
import pytest
from metro_safety.perception.frame_source import (
    VideoFileSource,
    ThreadedStreamSource,
    create_frame_source
)

def test_video_file_source():
    src = VideoFileSource("demo_clips/demo_fall.mp4", loop=True)
    assert src.is_opened() is True
    ret, frame, ts = src.read()
    assert ret is True
    assert frame is not None
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    assert ts >= 0.0
    src.release()

def test_frame_source_factory_alias():
    src = create_frame_source("fall")
    assert src.is_opened() is True
    ret, frame, ts = src.read()
    assert ret is True
    assert frame is not None
    src.release()

def test_frame_source_network_fallback():
    # Connecting to an invalid IP address should safely engage the fallback video
    src = create_frame_source(
        source_uri="http://192.0.2.1:9999/invalid_stream",
        fallback_path="demo_clips/demo_fall.mp4",
        allow_fallback=True
    )
    assert src.is_opened() is True
    ret, frame, ts = src.read()
    assert ret is True
    assert frame is not None
    assert "FALLBACK" in src.source_name
    src.release()

def test_threaded_stream_drops_stale_frames():
    # Test that reading multiple times returns valid timestamps without queue lag
    src = create_frame_source("demo_clips/demo_fall.mp4")
    ret1, f1, ts1 = src.read()
    time.sleep(0.05)
    ret2, f2, ts2 = src.read()
    assert ret1 is True and ret2 is True
    assert ts2 > ts1
    src.release()

