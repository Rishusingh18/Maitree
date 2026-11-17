"""End-to-end tests for the Video Forensic & Upload pipeline and API endpoints."""
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from metro_safety.tools.video_processor import (
    PROCESSED_FRAME_CACHE,
    get_cached_processed_stream,
)


def _cleanup_video_artifacts(video_id: str, incidents: list):
    """Remove generated video frames, MP4, and snapshots created during tests."""
    uploads = Path("data/uploads")
    shutil.rmtree(uploads / f"frames_{video_id}", ignore_errors=True)
    (uploads / f"processed_{video_id}.mp4").unlink(missing_ok=True)
    snapshots = Path("data/snapshots")
    for inc in incidents:
        ev_id = inc.get("id")
        if ev_id:
            (snapshots / f"{ev_id}.jpg").unlink(missing_ok=True)


def test_demo_clip_formdata_and_unregistered_snapshots():
    """Demo clip endpoint honors FormData fields, saves snapshots even when unregistered, and populates metadata."""
    client = TestClient(app)
    vid_id = None
    incidents = []
    try:
        resp = client.post(
            "/api/video/demo/demo_fall.mp4",
            data={"register_events": "false", "platform_id": "kashmere_gate_p1"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        vid_id = data["video_id"]
        incidents = data["incidents"]

        assert data["status"] == "ok"
        assert data["total_frames"] > 0
        assert data["processed_fps"] > 0
        assert data["incidents_count"] == len(incidents)
        assert data["incidents_count"] > 0

        first_inc = incidents[0]
        assert first_inc["platform_id"] == "kashmere_gate_p1"
        assert first_inc["registered_to_hub"] is False
        assert first_inc["frame_idx"] == first_inc["frame_index"]
        assert first_inc.get("snapshot_url")
        assert first_inc.get("bbox") is not None

        # Snapshot must be downloadable even when register_events=false
        snap_resp = client.get(first_inc["snapshot_url"])
        assert snap_resp.status_code == 200
        assert snap_resp.headers["content-type"].startswith("image/jpeg")
        assert len(snap_resp.content) > 1000

        mp4_resp = client.get(f"/api/processed_video/{vid_id}")
        assert mp4_resp.status_code == 200
        assert len(mp4_resp.content) > 1000

        frame_resp = client.get(f"/api/processed_frame/{vid_id}/0")
        assert frame_resp.status_code == 200
        assert frame_resp.headers["content-type"].startswith("image/jpeg")

        # Verify MJPEG generator yields valid JPEG chunk and closes cleanly
        gen = get_cached_processed_stream(vid_id)
        chunk = next(gen)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        gen.close()
    finally:
        if vid_id:
            _cleanup_video_artifacts(vid_id, incidents)


def test_video_upload_preserves_previous_cache_and_registers_events():
    """Uploading a video processes cleanly and does not wipe previous videos from cache."""
    client = TestClient(app)
    demo_clip = Path("demo_clips/demo_jump.mp4")
    assert demo_clip.exists()

    uploads_before = set(Path("data/uploads").glob("raw_*demo_jump*.mp4"))
    vid1 = vid2 = None
    inc1 = inc2 = []
    try:
        with open(demo_clip, "rb") as f:
            r1 = client.post(
                "/api/video/upload",
                files={"file": ("demo_jump.mp4", f, "video/mp4")},
                data={"register_events": "true", "platform_id": "mandi_house_p3"},
            )
        assert r1.status_code == 200, r1.text
        d1 = r1.json()
        vid1, inc1 = d1["video_id"], d1["incidents"]
        assert vid1 in PROCESSED_FRAME_CACHE

        with open(demo_clip, "rb") as f:
            r2 = client.post(
                "/api/video/upload",
                files={"file": ("demo_jump_2.mp4", f, "video/mp4")},
                data={"register_events": "true", "platform_id": "rajiv_chowk_p2"},
            )
        assert r2.status_code == 200, r2.text
        d2 = r2.json()
        vid2, inc2 = d2["video_id"], d2["incidents"]
        assert vid1 in PROCESSED_FRAME_CACHE
        assert vid2 in PROCESSED_FRAME_CACHE
    finally:
        if vid1:
            _cleanup_video_artifacts(vid1, inc1)
        if vid2:
            _cleanup_video_artifacts(vid2, inc2)
        for p in set(Path("data/uploads").glob("raw_*demo_jump*.mp4")) - uploads_before:
            p.unlink(missing_ok=True)


def test_empty_upload_and_missing_demo_return_proper_errors():
    """Empty uploads return 400 and unknown demo clips return 404 with JSON messages."""
    client = TestClient(app)

    r_empty = client.post(
        "/api/video/upload",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r_empty.status_code == 400
    assert r_empty.json()["status"] == "error"

    r_missing = client.post("/api/video/demo/does_not_exist_xyz.mp4")
    assert r_missing.status_code == 404
    assert r_missing.json()["status"] == "error"
