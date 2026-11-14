"""End-to-end integration test running the complete pipeline on a synthetic clip."""
from pathlib import Path
import json
import pytest
from metro_safety.demo.run import run_pipeline
from metro_safety.simulation.generator import ScenarioGenerator

def test_pipeline_end_to_end(tmp_path):
    gen = ScenarioGenerator()
    traj = gen.generate_person_trajectory("ACCIDENTAL_FALL", num_frames=35)
    video_path = tmp_path / "test_clip.mp4"
    out_video = tmp_path / "test_out.mp4"
    out_log = tmp_path / "test_log.json"

    gen.render_scenario_video(traj, video_path)
    assert video_path.exists()

    run_pipeline(
        source_uri=str(video_path),
        output_video=str(out_video),
        output_log=str(out_log),
        train_state_val="APPROACHING",
        show_window=False
    )

    assert out_video.exists()
    assert out_log.exists()

    with open(out_log, "r", encoding="utf-8") as f:
        log_data = json.load(f)
        assert "events" in log_data
        assert log_data["frames_processed"] == 35


def test_replay_jump_triggers_api_and_buzzer(tmp_path):
    """Integration test: Replays demo_clips/demo_jump.mp4 through pipeline connected to LiveHub,
    asserting DELIBERATE_JUMP event reaches /api/events on Flask dashboard and triggers the buzzer.
    """
    from metro_safety.demo.hub import LiveHub
    from metro_safety.escalation.buzzer import HardwareBuzzer
    from metro_safety.demo.dashboard import create_dashboard_app
    from metro_safety.tools.replay import replay_clip

    video_clip = "demo_clips/demo_jump.mp4"
    assert Path(video_clip).exists(), f"Missing required test clip: {video_clip}"

    hub = LiveHub()
    # Audio disabled to prevent host beeping during automated test suite
    buzzer = HardwareBuzzer(audio_enabled=False)
    hub.set_hardware_components(buzzer=buzzer)

    flask_app = create_dashboard_app(hub)
    client = flask_app.test_client()

    audit_log = tmp_path / "replay_test_audit.json"

    # Replay first 30 frames (deliberate jump occurs at frames 20-23)
    replay_clip(
        video_path=video_clip,
        show_window=False,
        output_log=str(audit_log),
        hub=hub,
        buzzer=buzzer,
        max_frames=30
    )

    # 1. Verify buzzer state
    status = buzzer.get_status()
    assert status["active"] is True
    assert status["pattern"] == "L3"

    # 2. Verify Flask /api/events endpoint
    resp = client.get("/api/events")
    assert resp.status_code == 200
    data = resp.get_json()

    events = data.get("events", [])
    assert len(events) > 0

    jump_events = [ev for ev in events if ev.get("event_type") == "DELIBERATE_JUMP"]
    assert len(jump_events) > 0, f"Expected DELIBERATE_JUMP event in /api/events, got: {events}"

    first_jump = jump_events[0]
    assert first_jump["severity"] == "L3"
    assert "kinematics" in first_jump

    # 3. Verify Telemetry exposes buzzer active status
    assert data["telemetry"]["buzzer"]["active"] is True
    assert data["telemetry"]["buzzer"]["pattern"] == "L3"

    # 4. Verify Operator Acknowledge silences the buzzer
    ack_resp = client.post("/api/acknowledge")
    assert ack_resp.status_code == 200
    assert buzzer.get_status()["active"] is False

    buzzer.stop()


