"""Interactive tool to calibrate and draw platform zones:
- Can capture directly from live camera stream (HTTP IP Webcam, RTSP, Webcam index, or video file).
- Displays live preview and lets user press SPACE to capture reference frame for zone calibration.
- User clicks points to outline 'platform', 'yellow_band', 'track_bed', and 'danger_zone'.
- Saves polygons directly into config.yaml (preserving remaining config sections).
"""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import time

from metro_safety.perception.frame_source import create_frame_source
from metro_safety.config import load_config, DEFAULT_CONFIG_PATH

def calibrate_zones(
    source_uri: str = "http://192.168.1.32:8080/video",
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    captured_image_path: str = "data/calibrated_frame.jpg"
):
    cfg_file = Path(config_path)
    cfg = load_config(cfg_file)

    print(f"Connecting to source: {source_uri}...")
    source = create_frame_source(source_uri)

    reference_frame = None

    # Step 1: Live Preview & Capture
    preview_win = "Zone Calibration - Press SPACE to capture frame, or 'q' to cancel"
    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)

    print("\n--- Live Preview Active ---")
    print("Point the camera at your simulated platform edge.")
    print("Press SPACE when satisfied with the framing to begin drawing zones.")
    print("Press 'q' to abort.")

    while True:
        ret, frame, _ = source.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        display = frame.copy()
        h, w = display.shape[:2]
        # Overlay instructions
        cv2.putText(display, f"LIVE STREAM: {source.source_name} ({w}x{h})", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press [SPACE] to capture frame for polygon drawing", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(preview_win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord(' '):
            reference_frame = frame.copy()
            break
        elif key == ord('q'):
            print("Calibration aborted by user.")
            cv2.destroyAllWindows()
            source.release()
            return

    cv2.destroyWindow(preview_win)
    source.release()

    if reference_frame is None:
        print("No frame captured.")
        return

    # Save reference frame
    ref_path = Path(captured_image_path)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref_path), reference_frame)
    print(f"Reference frame saved to: {ref_path}")

    # Step 2: Polygon Drawing
    zones_to_define = ["platform", "yellow_band", "track_bed", "danger_zone"]
    current_zone_idx = 0
    saved_zones = {}
    current_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([int(x), int(y)])

    draw_win = "Draw Zones: Click vertices, 'c' to close zone, 'r' reset, 's' save, 'q' quit"
    cv2.namedWindow(draw_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(draw_win, mouse_callback)

    colors = {
        "platform": (0, 200, 0),
        "yellow_band": (0, 230, 255),
        "track_bed": (0, 0, 220),
        "danger_zone": (0, 140, 255)
    }

    print("\n--- Zone Annotation Mode ---")
    print("Click around each area in order:")
    print(" 1. PLATFORM (safe waiting area)")
    print(" 2. YELLOW_BAND (safety boundary line/tactile tiles)")
    print(" 3. TRACK_BED (track area below platform)")
    print(" 4. DANGER_ZONE (combined edge + tracks)")
    print("\nKey commands:")
    print(" - 'c': Complete & close current zone polygon (requires >= 3 points)")
    print(" - 'r': Reset current zone points")
    print(" - 's': Save polygons directly into config.yaml")
    print(" - 'q': Quit without saving")

    while True:
        display = reference_frame.copy()

        # Draw already saved zones
        for name, pts in saved_zones.items():
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            col = colors.get(name, (200, 200, 200))
            cv2.polylines(display, [pts_arr], isClosed=True, color=col, thickness=2)
            cv2.putText(display, name.upper(), (pts[0][0], pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # Draw active zone points
        curr_name = zones_to_define[current_zone_idx] if current_zone_idx < len(zones_to_define) else "ALL ZONES DEFINED! Press 's' to save"
        curr_col = colors.get(curr_name, (0, 255, 255))
        cv2.putText(display, f"ACTIVE: {curr_name.upper()}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, curr_col, 2)

        for pt in current_points:
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
        if len(current_points) > 1:
            cv2.polylines(display, [np.array(current_points, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

        cv2.imshow(draw_win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('c') and len(current_points) >= 3:
            if current_zone_idx < len(zones_to_define):
                z_name = zones_to_define[current_zone_idx]
                saved_zones[z_name] = current_points
                print(f"-> Completed polygon for: {z_name} ({len(current_points)} vertices)")
                current_zone_idx += 1
                current_points = []
                if current_zone_idx >= len(zones_to_define):
                    print("\nAll 4 zones defined! Press 's' to save to config.yaml.")
        elif key == ord('r'):
            current_points = []
            print("Reset current polygon points.")
        elif key == ord('s'):
            if len(saved_zones) < 3:
                print("[Warning] Please define at least platform, yellow_band, and track_bed before saving.")
                continue

            # Update config file
            cfg["zones"] = saved_zones
            # Also update input resolution to match calibrated camera
            h, w = reference_frame.shape[:2]
            cfg["system"]["input_resolution"] = [int(w), int(h)]
            cfg["system"]["source"] = source_uri

            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            print(f"\n[SUCCESS] Calibrated zones successfully saved into: {cfg_file}")
            print(f"Resolution set to: {w}x{h}")
            break
        elif key == ord('q'):
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live CCTV Zone Calibration Tool")
    parser.add_argument("--source", type=str, default="http://192.168.1.32:8080/video",
                        help="Live camera stream URL, webcam index (0), or video file")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Config YAML path to update")
    parser.add_argument("--capture-path", type=str, default="data/calibrated_frame.jpg",
                        help="Where to save captured reference image")
    args = parser.parse_args()

    calibrate_zones(args.source, args.config, args.capture_path)

