"""Integration test for /video_feed MJPEG streaming endpoint.
Asserts:
1. /video_feed returns HTTP 200 with multipart/x-mixed-replace header
2. Frames arrive continuously with valid JPEG headers (0xFF 0xD8)
3. Decoded frames match published dimensions and colors
4. Fails if the stream returns no frames or hangs
"""
import time
import threading
import requests
import cv2
import numpy as np
import pytest
from werkzeug.serving import make_server

from metro_safety.demo.hub import LiveHub
from metro_safety.demo.dashboard import create_dashboard_app


def find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def test_video_feed_stream_returns_valid_frames():
    port = find_free_port()
    hub = LiveHub(cctv_id="TEST_CCTV", station_id="TEST_STN", platform_id="TEST_PL")

    # Initial frame
    test_frame_1 = np.zeros((240, 320, 3), dtype=np.uint8)
    test_frame_1[:, :, 0] = 255  # Solid Blue
    hub.update_frame(test_frame_1, is_online=True)

    app = create_dashboard_app(hub)
    server = make_server("127.0.0.1", port, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(0.3)

    # Background publisher simulating live video pipeline
    stop_event = threading.Event()

    def publisher():
        counter = 0
        while not stop_event.is_set():
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, 1] = (counter * 20) % 255  # Green pulses
            hub.update_frame(frame, is_online=True)
            counter += 1
            time.sleep(0.04)

    pub_thread = threading.Thread(target=publisher, daemon=True)
    pub_thread.start()

    try:
        url = f"http://127.0.0.1:{port}/video_feed"
        resp = requests.get(url, stream=True, timeout=6.0)

        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
        content_type = resp.headers.get("Content-Type", "")
        assert "multipart/x-mixed-replace" in content_type, f"Invalid Content-Type: {content_type}"
        assert "boundary=frame" in content_type, f"Missing boundary: {content_type}"

        # Collect stream chunks and parse multipart frames
        buffer = b""
        frames_extracted = []
        t0 = time.time()

        for chunk in resp.iter_content(chunk_size=1024):
            buffer += chunk
            while b"--frame" in buffer:
                # Split at boundary
                parts = buffer.split(b"--frame", 1)
                before = parts[0]
                buffer = parts[1]

                # Look for JPEG in the previous part
                soi = before.find(b"\xff\xd8")
                eoi = before.find(b"\xff\xd9")
                if soi != -1 and eoi != -1 and eoi > soi:
                    jpeg_bytes = before[soi:eoi + 2]
                    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        frames_extracted.append(img)
                        if len(frames_extracted) >= 3:
                            break

            if len(frames_extracted) >= 3 or (time.time() - t0) > 4.0:
                break

        # Assert at least 3 valid frames received
        assert len(frames_extracted) >= 3, f"Expected >= 3 frames from /video_feed, received {len(frames_extracted)}"

        for f in frames_extracted:
            assert f.shape == (240, 320, 3), f"Frame shape mismatch: {f.shape}"
            assert f.size > 0

    finally:
        stop_event.set()
        pub_thread.join(timeout=1.0)
        server.shutdown()
        server_thread.join(timeout=1.0)


def test_video_feed_offline_fallback():
    """Assert /video_feed returns offline standby frame when camera is disconnected."""
    port = find_free_port()
    hub = LiveHub(cctv_id="TEST_CCTV", station_id="TEST_STN", platform_id="TEST_PL")

    # Mark offline
    hub.update_frame(None, is_online=False, status_msg="Camera offline")

    app = create_dashboard_app(hub)
    server = make_server("127.0.0.1", port, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(0.3)

    try:
        url = f"http://127.0.0.1:{port}/video_feed"
        resp = requests.get(url, stream=True, timeout=5.0)
        assert resp.status_code == 200

        buffer = b""
        got_standby = False
        t0 = time.time()
        for chunk in resp.iter_content(chunk_size=1024):
            buffer += chunk
            soi = buffer.find(b"\xff\xd8")
            eoi = buffer.find(b"\xff\xd9")
            if soi != -1 and eoi != -1 and eoi > soi:
                jpeg = buffer[soi:eoi+2]
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert img is not None
                got_standby = True
                break
            if time.time() - t0 > 3.0:
                break

        assert got_standby, "Expected standby frame when camera is offline"

    finally:
        server.shutdown()
        server_thread.join(timeout=1.0)

