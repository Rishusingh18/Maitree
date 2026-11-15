"""Interactive OpenCV tool to draw, validate, and save CCTV zone polygons.
Features:
- Validates polygons: detects self-intersecting bowties, < 3 points, tiny areas
- Hotkey 'f' offers automatic repair by reordering points angularly around centroid
- Visual preview of validated zones including optional 'ignore' zones
"""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np

from metro_safety.zones.geometry import (
    validate_polygon,
    reorder_polygon_around_centroid,
    is_polygon_self_intersecting
)


def run_annotator(image_path: str, output_yaml: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image at '{image_path}'. Creating 640x480 canvas.")
        image = np.zeros((480, 640, 3), dtype=np.uint8)

    zones_to_define = ["platform", "yellow_band", "track_bed", "danger_zone", "ignore"]
    current_zone_idx = 0
    saved_zones = {}
    current_points = []
    status_msg = "Click vertices to draw polygon. Press 'c' to close, 'f' to auto-fix bowtie."
    status_color = (0, 255, 255)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points, status_msg, status_color
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            status_msg = f"Added vertex #{len(current_points)} at ({x}, {y})"
            status_color = (255, 255, 255)

    window_name = "Zone Annotator - [c] close | [f] auto-fix | [u] undo | [s] save | [q] quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== DMRC Platform Zone Annotator Tool ===")
    print("Click mouse to add polygon vertices.")
    print("Press 'c' to validate & complete current zone.")
    print("Press 'f' to auto-fix self-intersecting (bowtie) polygon around its centroid.")
    print("Press 'u' or Backspace to undo last vertex.")
    print("Press 'r' to reset current zone vertices.")
    print("Press 's' to validate all zones and save YAML.")
    print("Press 'q' to quit without saving.\n")

    zone_colors = {
        "platform": (0, 200, 0),
        "yellow_band": (0, 230, 255),
        "track_bed": (0, 0, 220),
        "danger_zone": (0, 100, 255),
        "ignore": (160, 100, 160)
    }

    while True:
        display = image.copy()

        # Draw already saved zones
        for name, pts in saved_zones.items():
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            col = zone_colors.get(name, (0, 255, 0))
            cv2.polylines(display, [pts_arr], isClosed=True, color=col, thickness=2)
            cv2.putText(display, name.upper(), (pts[0][0], pts[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # Active zone header
        curr_name = zones_to_define[current_zone_idx] if current_zone_idx < len(zones_to_define) else "All Defined"
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(display, f"Active Zone: {curr_name} ({current_zone_idx+1}/{len(zones_to_define)})",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Draw currently placed points & tentative polygon
        for pt in current_points:
            cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)
        if len(current_points) > 1:
            poly_color = (255, 120, 0)
            if len(current_points) >= 3:
                self_int, _ = is_polygon_self_intersecting(current_points)
                if self_int:
                    poly_color = (0, 0, 255)  # Red warning if bowtie!
            cv2.polylines(display, [np.array(current_points, dtype=np.int32)],
                          isClosed=False, color=poly_color, thickness=2)

        # Status footer bar
        cv2.rectangle(display, (0, display.shape[0] - 30), (display.shape[1], display.shape[0]), (20, 20, 20), -1)
        cv2.putText(display, status_msg, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('c'):
            # Close and validate
            if current_zone_idx >= len(zones_to_define):
                status_msg = "All standard zones already defined. Press 's' to save."
                status_color = (0, 255, 255)
                continue

            valid, err_msg = validate_polygon(current_points, name=curr_name)
            if not valid:
                status_msg = f"[REJECTED] {err_msg}! Press 'f' to auto-fix."
                status_color = (0, 0, 255)
                print(f"\n[Warning] Cannot close zone '{curr_name}': {err_msg}")
                print("Tip: Press 'f' to auto-reorder points around their centroid, or 'u' to undo vertex.\n")
            else:
                saved_zones[curr_name] = current_points
                print(f"[Zone Annotator] Saved valid zone: {curr_name} ({len(current_points)} vertices)")
                current_zone_idx += 1
                current_points = []
                status_msg = f"Zone '{curr_name}' accepted. Next: {zones_to_define[current_zone_idx] if current_zone_idx < len(zones_to_define) else 'Done'}"
                status_color = (0, 255, 0)

        elif key == ord('f'):
            # Offer automatic fix reordering points around centroid
            if len(current_points) >= 3:
                fixed_pts = reorder_polygon_around_centroid(current_points)
                valid, err_msg = validate_polygon(fixed_pts, name=curr_name)
                current_points = [(int(round(p[0])), int(round(p[1]))) for p in fixed_pts]
                if valid:
                    status_msg = "[AUTO-FIX APPLIED] Reordered around centroid. Press 'c' to accept."
                    status_color = (0, 255, 0)
                    print(f"[Auto-Fix] Successfully repaired bowtie polygon around centroid.")
                else:
                    status_msg = f"[AUTO-FIX WARNING] {err_msg}"
                    status_color = (0, 140, 255)
            else:
                status_msg = "Need at least 3 points to auto-fix polygon."
                status_color = (0, 140, 255)

        elif key == ord('u') or key == ord('z') or key == 8:  # Backspace
            if len(current_points) > 0:
                removed = current_points.pop()
                status_msg = f"Removed vertex ({removed[0]}, {removed[1]})"
                status_color = (200, 200, 200)
            elif current_zone_idx > 0:
                current_zone_idx -= 1
                undone_zone = zones_to_define[current_zone_idx]
                if undone_zone in saved_zones:
                    current_points = saved_zones.pop(undone_zone)
                status_msg = f"Re-opened zone '{undone_zone}' for editing."
                status_color = (0, 255, 255)

        elif key == ord('r'):
            current_points = []
            status_msg = f"Reset vertices for current polygon '{curr_name}'."
            status_color = (255, 255, 0)

        elif key == ord('s'):
            # Validate all zones before saving
            all_valid = True
            for zname, zpts in saved_zones.items():
                v, msg = validate_polygon(zpts, name=zname)
                if not v:
                    all_valid = False
                    status_msg = f"Cannot save: zone '{zname}' is invalid ({msg})"
                    status_color = (0, 0, 255)
                    print(f"[Save Blocked] {status_msg}")
                    break

            if all_valid:
                out_path = Path(output_yaml)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    yaml.dump({"zones": saved_zones}, f, sort_keys=False)
                print(f"[Zone Annotator] All zones validated and saved to {output_yaml}")
                break

        elif key == ord('q'):
            print("[Zone Annotator] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate and validate camera zones")
    parser.add_argument("--image", type=str, default="", help="Path to reference CCTV frame")
    parser.add_argument("--out", type=str, default="config_zones.yaml", help="Output YAML path")
    args = parser.parse_args()
    run_annotator(args.image, args.out)
