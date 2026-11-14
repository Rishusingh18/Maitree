"""Interactive tool to calibrate and draw platform zones:
- Captures from live camera stream or video file.
- Validates polygons (detects bowtie self-intersections, < 3 points, tiny areas).
- Hotkey 'f' provides automatic repair by sorting points angularly around centroid.
- Supports 'ignore' zone type where detections are discarded.
- Keeps a backup of config.yaml before saving changes.
"""
import shutil
import argparse
import json
from pathlib import Path
import yaml
import cv2
import numpy as np
import time

from metro_safety.perception.frame_source import create_frame_source
from metro_safety.config import load_config, DEFAULT_CONFIG_PATH
from metro_safety.zones.geometry import (
    validate_polygon,
    reorder_polygon_around_centroid,
    is_polygon_self_intersecting
)


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

    # Step 2: Polygon Drawing with Validation
    zones_to_define = ["platform", "yellow_band", "track_bed", "danger_zone", "ignore"]
    current_zone_idx = 0
    saved_zones = {}
    current_points = []
    status_bar_msg = "[c] Close  [f] Auto-Fix Bowtie  [u] Undo  [r] Reset Zone  [s] Save  [q] Quit"
    status_color = (220, 220, 220)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([int(x), int(y)])

    draw_win = "Draw Zones: [c] close | [f] auto-fix | [u] undo | [s] save | [q] quit"
    cv2.namedWindow(draw_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(draw_win, mouse_callback)

    colors = {
        "platform": (0, 200, 0),
        "yellow_band": (0, 230, 255),
        "track_bed": (0, 0, 220),
        "danger_zone": (0, 140, 255),
        "ignore": (160, 100, 160)
    }

    print("\n--- Zone Annotation Mode with Validation ---")
    print("Click around each area in order:")
    print(" 1. PLATFORM (safe waiting area)")
    print(" 2. YELLOW_BAND (safety boundary line/tactile tiles)")
    print(" 3. TRACK_BED (track area below platform)")
    print(" 4. DANGER_ZONE (combined edge + tracks)")
    print(" 5. IGNORE (optional seating/vending area where detections are discarded)")
    print("\nHotkeys:")
    print(" - 'c'           : Validate & close current zone polygon (rejects bowties/<3 pts/tiny areas)")
    print(" - 'f'           : Auto-fix self-intersecting polygon by reordering points around centroid")
    print(" - 'u' / Backspace: Undo last vertex or reopen previously closed zone")
    print(" - 's'           : Validate all zones and save to config.yaml (keeps config.yaml.bak)")
    print(" - 'q'           : Quit\n")

    while True:
        display = reference_frame.copy()
        h, w = display.shape[:2]

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
        cv2.putText(display, f"ACTIVE: {curr_name.upper()} ({len(current_points)} pts)",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, curr_col, 2)

        # Bottom help banner
        cv2.rectangle(display, (0, h - 35), (w, h), (30, 30, 30), -1)
        cv2.putText(display, status_bar_msg, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1)

        for pt in current_points:
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)
        if len(current_points) > 1:
            poly_col = (255, 255, 0)
            if len(current_points) >= 3:
                self_int, _ = is_polygon_self_intersecting(current_points)
                if self_int:
                    poly_col = (0, 0, 255)  # Red alert if bowtie!
            cv2.polylines(display, [np.array(current_points, dtype=np.int32)], isClosed=False, color=poly_col, thickness=2)

        cv2.imshow(draw_win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('c'):
            if current_zone_idx >= len(zones_to_define):
                continue
            z_name = zones_to_define[current_zone_idx]
            valid, err_msg = validate_polygon(current_points, name=z_name)
            if not valid:
                status_bar_msg = f"[REJECTED] {err_msg}! Press 'f' to auto-fix around centroid."
                status_color = (0, 0, 255)
                print(f"[Validation Failed] Zone '{z_name}': {err_msg}. Press 'f' to auto-fix.")
            else:
                saved_zones[z_name] = current_points
                print(f"-> Accepted valid polygon for: {z_name} ({len(current_points)} vertices)")
                current_zone_idx += 1
                current_points = []
                status_bar_msg = f"Saved '{z_name}'. Next: {zones_to_define[current_zone_idx] if current_zone_idx < len(zones_to_define) else 'All Defined'}"
                status_color = (0, 255, 0)

        elif key == ord('f'):
            # Automatic repair by sorting points around centroid
            if len(current_points) >= 3:
                fixed_pts = reorder_polygon_around_centroid(current_points)
                valid, err_msg = validate_polygon(fixed_pts, name=curr_name)
                current_points = [[int(p[0]), int(p[1])] for p in fixed_pts]
                if valid:
                    status_bar_msg = "[AUTO-FIX APPLIED] Reordered points around centroid. Press 'c' to accept."
                    status_color = (0, 255, 0)
                    print(f"[Auto-Fix] Successfully repaired bowtie polygon around centroid.")
                else:
                    status_bar_msg = f"[Auto-Fix Warning] {err_msg}"
                    status_color = (0, 140, 255)

        elif key == ord('u') or key == ord('z') or key == 8:  # 8 is Backspace
            if len(current_points) > 0:
                removed = current_points.pop()
                status_bar_msg = f"Removed last vertex: {removed}"
                status_color = (200, 200, 200)
            elif current_zone_idx > 0:
                current_zone_idx -= 1
                undone_zone = zones_to_define[current_zone_idx]
                if undone_zone in saved_zones:
                    current_points = saved_zones.pop(undone_zone)
                status_bar_msg = f"Re-opened zone '{undone_zone}'. Press 'u' to drop points or 'f' to repair."
                status_color = (0, 255, 255)

        elif key == ord('r'):
            current_points = []
            status_bar_msg = "Reset vertices for active polygon."
            status_color = (255, 255, 0)

        elif key == ord('R'):
            saved_zones.clear()
            current_points = []
            current_zone_idx = 0
            status_bar_msg = "Reset all zones."
            status_color = (255, 255, 0)

        elif key == ord('s'):
            if len(saved_zones) < 3:
                status_bar_msg = "Please define at least platform, yellow_band, and track_bed before saving."
                status_color = (0, 0, 255)
                continue

            # Validate all zones before writing to config
            all_valid = True
            for zn, zpts in saved_zones.items():
                v, m = validate_polygon(zpts, name=zn)
                if not v:
                    all_valid = False
                    status_bar_msg = f"Cannot save: zone '{zn}' is invalid ({m})"
                    status_color = (0, 0, 255)
                    break

            if not all_valid:
                continue

            # Backup old config
            bak_file = cfg_file.with_suffix(".yaml.bak")
            try:
                shutil.copyfile(cfg_file, bak_file)
                print(f"[Backup] Saved backup of old config to: {bak_file}")
            except Exception as e:
                print(f"[Backup Warning] Could not create backup: {e}")

            # Save reference frame image and reference polygons
            ref_img_path = Path("data/reference_frame.jpg")
            ref_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ref_img_path), reference_frame)

            ref_poly_path = Path("data/reference_polygons.json")
            with open(ref_poly_path, "w", encoding="utf-8") as rf:
                json.dump({
                    "zones": saved_zones,
                    "resolution": [int(w), int(h)],
                    "timestamp": time.time(),
                    "source": source_uri
                }, rf, indent=2)

            # Update config file
            cfg["zones"] = saved_zones
            h, w = reference_frame.shape[:2]
            cfg["system"]["input_resolution"] = [int(w), int(h)]
            cfg["system"]["source"] = source_uri
            cfg["reference_frame"] = "data/reference_frame.jpg"

            with open(cfg_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            print(f"\n[SUCCESS] Calibrated zones & reference frame saved into: {cfg_file}")
            print(f"Reference frame image saved to: {ref_img_path}")
            print(f"Reference polygons saved to: {ref_poly_path}")
            print(f"Resolution set to: {w}x{h}")
            break

        elif key == ord('q'):
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live CCTV Zone Calibration & Validation Tool")
    parser.add_argument("--source", type=str, default="http://192.168.1.32:8080/video",
                        help="Live camera stream URL, webcam index (0), or video file")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="Config YAML path to update")
    parser.add_argument("--capture-path", type=str, default="data/calibrated_frame.jpg",
                        help="Where to save captured reference image")
    args = parser.parse_args()

    calibrate_zones(args.source, args.config, args.capture_path)
