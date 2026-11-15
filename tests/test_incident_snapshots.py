import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from metro_safety.demo.hub import LiveHub, save_incident_snapshot, SNAPSHOTS_DIR
from backend.app.main import app

client = TestClient(app)


def test_save_incident_snapshot():
    """Verify that save_incident_snapshot writes a valid JPEG to disk and returns metadata."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 50, (0, 255, 0), -1)

    event_id = "EV-TEST-SNAPSHOT-001"
    res = save_incident_snapshot(event_id, dummy_frame)

    assert "snapshot_path" in res
    assert "snapshot_url" in res
    assert "snapshot_data" in res

    file_path = Path(res["snapshot_path"])
    assert file_path.exists()
    assert file_path.stat().st_size > 0

    assert res["snapshot_url"] == f"/api/snapshots/{event_id}.jpg"
    assert res["snapshot_data"].startswith("data:image/jpeg;base64,")

    # Clean up test file
    if file_path.exists():
        file_path.unlink()


def test_hub_push_event_with_frame_auto_captures():
    """Verify LiveHub.push_event auto-captures and persists screenshot when frame is provided."""
    hub = LiveHub(cctv_id="CAM-101", station_id="Rajiv Chowk", platform_id="Platform 2")
    dummy_frame = np.ones((200, 200, 3), dtype=np.uint8) * 128

    ev_id = "EV-TEST-AUTO-002"
    event_record = {
        "id": ev_id,
        "event_type": "DELIBERATE_JUMP",
        "severity": "CRITICAL"
    }

    hub.push_event(event_record, frame=dummy_frame)

    assert event_record["snapshot_url"] == f"/api/snapshots/{ev_id}.jpg"
    assert "snapshot_data" in event_record
    assert Path(event_record["snapshot_path"]).exists()

    # Clean up
    if Path(event_record["snapshot_path"]).exists():
        Path(event_record["snapshot_path"]).unlink()


def test_hub_push_event_with_latest_jpeg_fallback():
    """Verify LiveHub.push_event falls back to _latest_jpeg when no frame is explicitly passed."""
    hub = LiveHub(cctv_id="CAM-101", station_id="Rajiv Chowk", platform_id="Platform 2")
    dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
    hub.update_frame(dummy_frame)

    ev_id = "EV-TEST-FALLBACK-003"
    event_record = {
        "id": ev_id,
        "event_type": "TRACK_INTRUSION",
        "severity": "L3"
    }

    hub.push_event(event_record)

    assert event_record["snapshot_url"] == f"/api/snapshots/{ev_id}.jpg"
    assert Path(event_record["snapshot_path"]).exists()

    # Clean up
    if Path(event_record["snapshot_path"]).exists():
        Path(event_record["snapshot_path"]).unlink()


def test_api_snapshots_endpoint():
    """Verify FastAPI /api/snapshots/{snapshot_id} endpoint serves saved snapshots."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ev_id = "EV-TEST-ENDPOINT-004"
    snap_info = save_incident_snapshot(ev_id, dummy_frame)

    # Test retrieval with .jpg extension
    resp1 = client.get(f"/api/snapshots/{ev_id}.jpg")
    assert resp1.status_code == 200
    assert resp1.headers["content-type"] == "image/jpeg"
    assert len(resp1.content) > 0

    # Test retrieval without .jpg extension
    resp2 = client.get(f"/api/snapshots/{ev_id}")
    assert resp2.status_code == 200
    assert resp2.headers["content-type"] == "image/jpeg"

    # Test non-existent snapshot returns 404
    resp_404 = client.get("/api/snapshots/NON_EXISTENT_SNAPSHOT.jpg")
    assert resp_404.status_code == 404

    # Test path sanitization prevents directory traversal
    resp_traversal = client.get("/api/snapshots/../../etc/passwd")
    assert resp_traversal.status_code == 404

    # Clean up
    if Path(snap_info["snapshot_path"]).exists():
        Path(snap_info["snapshot_path"]).unlink()
