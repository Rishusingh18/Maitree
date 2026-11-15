"""Synthetic Camera Shake Simulation and Drift Ground-Truth Verification.
Generates deterministic synthetic camera wobble on a video sequence,
saves ground truth (frame, dx, dy, dtheta) to CSV,
and verifies tracked drift against ground truth with RMS error bounds.
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
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
    if not writer.isOpened():
        if cap.isOpened():
            cap.release()
        raise IOError(f"Failed to open VideoWriter for output path: {out_p}")

    records = []
    frame_idx = 0

    baseline_polys = {
        "platform": [[0.1 * w, 0.2 * h], [0.47 * w, 0.2 * h], [0.47 * w, 0.9 * h], [0.1 * w, 0.9 * h]],
        "yellow_band": [[0.47 * w, 0.2 * h], [0.53 * w, 0.2 * h], [0.53 * w, 0.9 * h], [0.47 * w, 0.9 * h]],
        "track_bed": [[0.53 * w, 0.2 * h], [0.9 * w, 0.2 * h], [0.9 * w, 0.9 * h], [0.53 * w, 0.9 * h]]
    }

    poly_pts_list = []
    for poly in baseline_polys.values():
        for pt in poly:
            poly_pts_list.append([float(pt[0]), float(pt[1])])
    ref_poly_pts = np.array(poly_pts_list, dtype=np.float32).reshape(-1, 1, 2)
    ref_first_pts = None

    # Base synthetic pattern across all 4 quadrants if video not opened
    base_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(50 + 120 * (y / float(h)))
        base_frame[y, :] = [val - 10, val, val + 10]

    for y in range(60, h, 35):
        cv2.line(base_frame, (0, y), (w, y), (40, 45, 50), 2)
    for x in range(0, w, 45):
        cv2.line(base_frame, (x, 60), (x, h), (40, 45, 50), 2)

    # Distinct high-contrast landmarks covering all 4 quadrants
    cv2.rectangle(base_frame, (50, 80), (190, 220), (30, 80, 220), -1)    # Top-Left
    cv2.rectangle(base_frame, (420, 90), (580, 240), (220, 160, 40), -1)  # Top-Right
    cv2.rectangle(base_frame, (220, 250), (380, 410), (50, 180, 70), -1)  # Center
    cv2.rectangle(base_frame, (50, 270), (180, 420), (80, 140, 200), -1)   # Bottom-Left
    cv2.rectangle(base_frame, (440, 310), (590, 440), (140, 60, 180), -1) # Bottom-Right

    cv2.putText(base_frame, "DMRC METRO OCC", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(base_frame, "PLATFORM 2 SOUTH", (430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(base_frame, "ENTRY GATE A", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(base_frame, "TRACK 2 HAZARD", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.circle(base_frame, (300, 330), 40, (255, 255, 255), -1)
    cv2.circle(base_frame, (300, 330), 25, (0, 0, 0), -1)
    cv2.circle(base_frame, (520, 400), 18, (255, 200, 0), -1)

    # Wobble starts smoothly from stationary reference at t=0
    phase_x = 0.0
    phase_y = 0.0
    phase_r = 0.0

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

            pts_warped = cv2.transform(ref_poly_pts, M).reshape(-1, 2)
            if ref_first_pts is None:
                ref_first_pts = pts_warped.copy()
            mean_vertex_disp = float(np.mean(np.linalg.norm(pts_warped - ref_first_pts, axis=1)))

            records.append({
                "frame": frame_idx,
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "dtheta_deg": round(dtheta, 4),
                "displacement_px": round(mean_vertex_disp, 4)
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
    estimated_drifts: Union[List[float], Dict[int, float], List[Tuple[int, float]]],
    ground_truth_csv_path: str,
    max_rms_threshold_px: float = 3.5,
    eval_window: Optional[Tuple[int, int]] = None
) -> Tuple[bool, float, float]:
    """Verify tracked drift estimates against wobble ground truth.
    Supports either frame-aligned dense lists (length must match) or sparse mappings by frame index.
    Returns (passed, rms_error, max_error).
    """
    gt_map = {}
    with open(ground_truth_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            gt_map[f_idx] = float(row["displacement_px"])

    if not gt_map:
        return False, 999.0, 999.0

    pairs = []
    if isinstance(estimated_drifts, dict):
        for f_idx, est_d in estimated_drifts.items():
            if eval_window and not (eval_window[0] <= f_idx <= eval_window[1]):
                continue
            if f_idx in gt_map:
                pairs.append((float(est_d), gt_map[f_idx]))
    elif isinstance(estimated_drifts, list) and estimated_drifts and isinstance(estimated_drifts[0], (tuple, list)):
        for f_idx, est_d in estimated_drifts:
            if eval_window and not (eval_window[0] <= f_idx <= eval_window[1]):
                continue
            if f_idx in gt_map:
                pairs.append((float(est_d), gt_map[f_idx]))
    else:
        # Dense list
        if eval_window:
            start_f, end_f = eval_window
            if len(estimated_drifts) > end_f:
                for f_idx in range(start_f, end_f + 1):
                    if f_idx in gt_map:
                        pairs.append((float(estimated_drifts[f_idx]), gt_map[f_idx]))
            else:
                window_len = end_f - start_f + 1
                if len(estimated_drifts) != window_len:
                    raise ValueError(f"Dense estimated_drifts length ({len(estimated_drifts)}) must exactly match eval_window length {window_len} (frames {start_f}..{end_f})")
                for offset, f_idx in enumerate(range(start_f, end_f + 1)):
                    if f_idx in gt_map:
                        pairs.append((float(estimated_drifts[offset]), gt_map[f_idx]))
        else:
            if len(estimated_drifts) != len(gt_map):
                raise ValueError(
                    f"Mismatched sequence lengths: estimated_drifts has {len(estimated_drifts)} items, "
                    f"ground truth has {len(gt_map)} frames. Use {{frame_idx: drift}} mapping for sparse evaluation."
                )
            for f_idx in range(len(estimated_drifts)):
                if f_idx in gt_map:
                    pairs.append((float(estimated_drifts[f_idx]), gt_map[f_idx]))

    if not pairs:
        return False, 999.0, 999.0

    est = np.array([p[0] for p in pairs])
    gt = np.array([p[1] for p in pairs])

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
