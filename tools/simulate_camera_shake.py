"""Synthetic Camera Shake Simulation and Drift Ground-Truth Verification.
Generates deterministic synthetic camera wobble on a video sequence,
saves ground truth (frame, dx, dy, dtheta) to CSV,
and verifies tracked drift against ground truth with RMS error bounds.
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def generate_camera_shake(
    input_video_path: str,
    output_video_path: str,
    ground_truth_csv_path: str,
    max_frames: int = 150,
    seed: int = 42,
    max_dx: float = 14.0,
    max_dy: float = 10.0,
    max_dtheta_deg: float = 1.8
) -> Dict[str, Any]:
    """Generate deterministic wobbled video clip and save ground-truth CSV."""
    np.random.seed(seed)
    cap = cv2.VideoCapture(input_video_path)

    use_synthetic = False
    if not cap.isOpened():
        use_synthetic = True
        w, h = 640, 480
        fps = 25.0
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_p = Path(output_video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    csv_p = Path(ground_truth_csv_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

    records = []
    frame_idx = 0

    # Smooth sinusoidal + Perlin-like random walk wobble
    phase_x = np.random.uniform(0, 2 * math.pi)
    phase_y = np.random.uniform(0, 2 * math.pi)
    phase_r = np.random.uniform(0, 2 * math.pi)

    # Base synthetic pattern if video not opened
    base_frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw realistic platform textured lines
    cv2.rectangle(base_frame, (0, 0), (w, h), (40, 45, 50), -1)
    for y in range(0, h, 40):
        cv2.line(base_frame, (0, y), (w, y), (65, 70, 75), 1)
    for x in range(0, w, 50):
        cv2.line(base_frame, (x, 0), (x, h), (65, 70, 75), 1)
    # Visual landmarks / posters
    cv2.rectangle(base_frame, (50, 40), (220, 150), (120, 160, 200), -1)
    cv2.rectangle(base_frame, (350, 40), (580, 180), (180, 140, 120), -1)
    cv2.putText(base_frame, "DMRC STATION OCC", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(base_frame, "PLATFORM 2 SAFETY", (370, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    try:
        while frame_idx < max_frames:
            if use_synthetic:
                frame = base_frame.copy()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            # Compute wobble
            t = float(frame_idx) / fps
            dx = max_dx * math.sin(2.0 * math.pi * 0.8 * t + phase_x) + np.random.normal(0, 0.5)
            dy = max_dy * math.sin(2.0 * math.pi * 1.1 * t + phase_y) + np.random.normal(0, 0.4)
            dtheta = max_dtheta_deg * math.sin(2.0 * math.pi * 0.5 * t + phase_r)

            # Apply affine warp (rotation around center + translation)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, dtheta, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy

            warped = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            writer.write(warped)

            records.append({
                "frame": frame_idx,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "dtheta_deg": round(dtheta, 4),
                "displacement_px": round(math.hypot(dx, dy), 4)
            })
            frame_idx += 1
    finally:
        if cap.isOpened():
            cap.release()
        writer.release()

    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "dx", "dy", "dtheta_deg", "displacement_px"])
        writer_csv.writeheader()
        writer_csv.writerows(records)

    return {
        "frames": frame_idx,
        "video": str(out_p),
        "csv": str(csv_p),
        "mean_displacement": float(np.mean([r["displacement_px"] for r in records])) if records else 0.0
    }


def verify_drift_against_ground_truth(
    estimated_drifts: List[float],
    ground_truth_csv_path: str,
    max_rms_threshold_px: float = 3.5
) -> Tuple[bool, float, float]:
    """Verify tracked drift estimates against wobble ground truth.
    Returns (passed, rms_error, max_error).
    """
    gt_drifts = []
    with open(ground_truth_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_drifts.append(float(row["displacement_px"]))

    min_len = min(len(estimated_drifts), len(gt_drifts))
    if min_len == 0:
        return False, 999.0, 999.0

    est = np.array(estimated_drifts[:min_len])
    gt = np.array(gt_drifts[:min_len])

    errors = est - gt
    rms_error = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(np.abs(errors)))

    passed = (rms_error <= max_rms_threshold_px)
    return passed, rms_error, max_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate camera wobble and output ground-truth CSV.")
    parser.add_argument("--input", type=str, default="demo_clips/demo_normal.mp4", help="Input video")
    parser.add_argument("--output", type=str, default="data/recordings/shaken_test_clip.mp4", help="Output shaken clip")
    parser.add_argument("--csv", type=str, default="data/recordings/camera_wobble_gt.csv", help="Ground truth CSV")
    parser.add_argument("--frames", type=int, default=100, help="Frames to generate")
    args = parser.parse_args()

    res = generate_camera_shake(args.input, args.output, args.csv, max_frames=args.frames)
    print(f"Generated {res['frames']} frames shaken video: {res['video']}")
    print(f"Ground truth saved to: {res['csv']}")
