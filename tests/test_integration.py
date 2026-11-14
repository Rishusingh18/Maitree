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

