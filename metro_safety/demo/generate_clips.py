"""Pre-generation script to create standard demo video clips:
1. demo_fall.mp4 (Accidental Fall incident)
2. demo_jump.mp4 (Deliberate Jump incident)
3. demo_distress.mp4 (Silent Distress Gesture incident)
"""
from pathlib import Path
from metro_safety.simulation.generator import ScenarioGenerator

def generate_demo_clips(output_dir: Path | str = "demo_clips"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = ScenarioGenerator()

    clips = [
        ("demo_fall.mp4", "ACCIDENTAL_FALL", "SIGNAL_FOR_HELP"),
        ("demo_jump.mp4", "DELIBERATE_JUMP", "SIGNAL_FOR_HELP"),
        ("demo_distress.mp4", "DISTRESS_GESTURE", "WRISTS_CROSSED")
    ]

    for fname, scenario, variant in clips:
        clip_path = out_dir / fname
        print(f"Generating {fname} ({scenario})...")
        traj = gen.generate_person_trajectory(scenario, num_frames=80, gesture_variant=variant)
        gen.render_scenario_video(traj, clip_path)
        print(f"Saved: {clip_path}")

if __name__ == "__main__":
    generate_demo_clips()
