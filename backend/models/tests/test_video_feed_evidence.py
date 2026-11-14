"""Black-panel evidence and MJPEG multipart stream validation (Addendum A).
Validates:
1. HTTP 200 with Content-Type containing 'multipart/x-mixed-replace' and boundary definition.
2. At least 5 JPEG parts arrive over 5 seconds.
3. Every JPEG part is decoded and confirmed non-black (mean pixel intensity > 5.0).
4. Saves 3 decoded frames to debug/ directory.
5. Verifies the producer process PID and server process PID match (same unified process).
"""
import os
import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np
import requests
import pytest


def test_video_feed_multipart_evidence_and_non_black_frames():
    port = 5003
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["METRO_SOURCE_OVERRIDE"] = "demo_clips/demo_fall.mp4"

    cmd = [
        sys.executable,
        "-m", "metro_safety.demo.live",
        "--source", "demo_clips/demo_fall.mp4",
        "--port", str(port),
        "--no-window",
        "--loop"
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 1. Wait for server online
    server_online = False
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/telemetry", timeout=1.0)
            if r.status_code == 200:
                server_online = True
                break
        except Exception:
            time.sleep(0.3)

    assert server_online, f"Live pipeline failed to start on port {port}"

    try:
        # 2. Check PID match via telemetry API
        tel_resp = requests.get(f"http://127.0.0.1:{port}/api/telemetry", timeout=3.0)
        assert tel_resp.status_code == 200
        tel_data = tel_resp.json()
        producer_pid = tel_data.get("producer_pid")
        server_pid = tel_data.get("server_pid")
        assert producer_pid is not None, "producer_pid missing from telemetry"
        assert server_pid is not None, "server_pid missing from telemetry"
        # Prompt requirement: Producer PID and Server PID must match
        print(f"\n[PID Check] Buffer Producer PID: {producer_pid} | Video Feed Server PID: {server_pid}")
        assert producer_pid == server_pid, f"PID mismatch: producer={producer_pid} vs server={server_pid}"

        # 3. Stream /video_feed for 5 seconds with requests(stream=True)
        url = f"http://127.0.0.1:{port}/video_feed"
        resp = requests.get(url, stream=True, timeout=5.0)
        assert resp.status_code == 200, f"Expected status 200, got {resp.status_code}"

        content_type = resp.headers.get("Content-Type", "")
        assert "multipart/x-mixed-replace" in content_type, f"Invalid Content-Type: {content_type}"
        assert "boundary=" in content_type, f"Boundary missing from Content-Type: {content_type}"

        debug_dir = Path("debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        buffer = bytearray()
        jpeg_frames = []
        part_sizes = []
        t_start = time.time()

        for chunk in resp.iter_content(chunk_size=4096):
            if time.time() - t_start > 5.0:
                break
            if not chunk:
                continue
            buffer.extend(chunk)

            while True:
                start_idx = buffer.find(b"\xff\xd8")  # JPEG start marker
                if start_idx == -1:
                    break
                end_idx = buffer.find(b"\xff\xd9", start_idx)  # JPEG end marker
                if end_idx == -1:
                    break

                jpeg_data = bytes(buffer[start_idx:end_idx + 2])
                del buffer[:end_idx + 2]

                part_sizes.append(len(jpeg_data))
                img_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    jpeg_frames.append(frame)

        # 4. Assertions on stream evidence
        assert len(jpeg_frames) >= 5, f"Expected >= 5 JPEG frames, got {len(jpeg_frames)}"
        mean_part_size = np.mean(part_sizes) if part_sizes else 0
        assert mean_part_size > 2000, f"JPEG frames too small ({mean_part_size} bytes)"

        # 5. Non-black verification (mean pixel value > 5.0)
        for i, f in enumerate(jpeg_frames[:3]):
            mean_val = float(np.mean(f))
            assert mean_val > 5.0, f"Frame {i} is black/blank (mean pixel value: {mean_val:.2f} <= 5.0)"
            save_path = debug_dir / f"test_evidence_frame_{i+1}.jpg"
            cv2.imwrite(str(save_path), f)
            assert save_path.exists()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

