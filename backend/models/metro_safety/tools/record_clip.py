"""Tool to record raw footage from live camera stream (phone IP webcam, webcam, or RTSP).
Saves pristine video clips to disk for offline kinematic threshold tuning and replay.

Usage:
  python -m metro_safety.tools.record_clip --seconds 20
  python -m metro_safety.tools.record_clip --seconds 15 --out data/recordings/jump_test_1.mp4
"""
import time
import argparse
from pathlib import Path
import cv2

from metro_safety.config import load_config
from metro_safety.perception.frame_source import create_frame_source

def record_clip(
    seconds: float = 20.0,
    output_path: str = "data/recordings/live_clip.mp4",
    source_uri: str = None,
    config_path: str = None,
    show_window: bool = True
):
    cfg = load_config(config_path)
    resolved_source = source_uri or cfg["system"].get("source", "http://192.168.1.32:8080/video")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to live camera source: {resolved_source}")
    frame_source = create_frame_source(resolved_source)

    print(f"Recording {seconds:.1f} seconds of raw video to: {out_file}")
    print("Press 'q' in the preview window to stop early.")

    writer = None
    win_name = f"Recording Raw Clip ({seconds:.0f}s) - Press 'q' to stop"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t_start = None
    frames_recorded = 0
    fps_estimate = 25.0

    try:
        while True:
            ret, frame, _ = frame_source.read()
            if not ret or frame is None:
                if t_start and (time.time() - t_start) >= seconds:
                    print("Reached recording duration limit during camera outage.")
                    break
                if show_window:
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord('q'):
                        print("User requested early stop during camera outage.")
                        break
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]

            # Lazy initialize writer on first valid frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_file), fourcc, fps_estimate, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {out_file}")
                t_start = time.time()
                print(f"Capturing at {w}x{h} resolution...")

            writer.write(frame)
            frames_recorded += 1

            elapsed = time.time() - t_start
            remaining = max(0.0, seconds - elapsed)

            if show_window:
                display = frame.copy()
                # On-screen HUD
                cv2.circle(display, (25, 30), 10, (0, 0, 255), -1)  # Red recording dot
                cv2.putText(display, f"REC [{frames_recorded} frames] | REMAINING: {remaining:.1f}s", (45, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, f"Saving to: {out_file.name}", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow(win_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested early stop.")
                    break

            if elapsed >= seconds:
                print(f"Completed {seconds:.1f}s recording target.")
                break

    finally:
        if writer is not None:
            writer.release()
        frame_source.release()
        if show_window:
            cv2.destroyAllWindows()

    actual_fps = frames_recorded / elapsed if t_start and elapsed > 0 else 0
    print(f"\n[DONE] Saved raw clip: {out_file}")
    print(f"Total Frames: {frames_recorded} | Duration: {elapsed:.2f}s | Effective FPS: {actual_fps:.1f}")
    return str(out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record raw video clip from live CCTV camera")
    parser.add_argument("--seconds", type=float, default=20.0, help="Duration to record in seconds")
    parser.add_argument("--out", type=str, default="data/recordings/live_clip.mp4", help="Output MP4 file path")
    parser.add_argument("--source", type=str, default=None, help="Camera source URI")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--no-window", action="store_true", help="Run headlessly without GUI window")
    args = parser.parse_args()

    record_clip(
        seconds=args.seconds,
        output_path=args.out,
        source_uri=args.source,
        config_path=args.config,
        show_window=not args.no_window
    )

