"""Interactive OpenCV tool to draw and save CCTV zone polygons."""
import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np

def run_annotator(image_path: str, output_yaml: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image at {image_path}. Creating blank canvas.")
        image = np.zeros((480, 640, 3), dtype=np.uint8)

    zones_to_define = ["platform", "yellow_band", "track_bed", "danger_zone"]
    current_zone_idx = 0
    saved_zones = {}
    current_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])

    window_name = "Zone Annotator - Click points, 'c' to close zone, 's' to save, 'q' to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("=== Zone Annotator Tool ===")
    print("Click to add polygon vertices.")
    print("Press 'c' to complete current zone polygon.")
    print("Press 'r' to reset current zone.")
    print("Press 's' to save YAML and finish.")
    print("Press 'q' to quit.")

    while True:
        display = image.copy()
        # Draw completed zones
        for name, pts in saved_zones.items():
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(display, name, (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw current points
        curr_name = zones_to_define[current_zone_idx] if current_zone_idx < len(zones_to_define) else "All Defined"
        cv2.putText(display, f"Active Zone: {curr_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for pt in current_points:
            cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)
        if len(current_points) > 1:
            cv2.polylines(display, [np.array(current_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('c') and len(current_points) >= 3:
            if current_zone_idx < len(zones_to_define):
                saved_zones[zones_to_define[current_zone_idx]] = current_points
                print(f"Saved zone: {zones_to_define[current_zone_idx]} with {len(current_points)} points.")
                current_zone_idx += 1
                current_points = []
        elif key == ord('u') or key == ord('z') or key == 8:  # 8 is Backspace
            if len(current_points) > 0:
                removed = current_points.pop()
                print(f"<- Removed last vertex: {removed}")
            elif current_zone_idx > 0:
                current_zone_idx -= 1
                undone_zone = zones_to_define[current_zone_idx]
                if undone_zone in saved_zones:
                    current_points = saved_zones.pop(undone_zone)
                print(f"<- Undid closed zone '{undone_zone}'. Re-opened for editing.")
        elif key == ord('r'):
            current_points = []
            print("Reset vertices for current polygon.")
        elif key == ord('R'):
            saved_zones.clear()
            current_points = []
            current_zone_idx = 0
            print("Reset all zones.")
        elif key == ord('s'):
            out_path = Path(output_yaml)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump({"zones": saved_zones}, f)
            print(f"Zones saved to {output_yaml}")
            break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate camera zones")
    parser.add_argument("--image", type=str, default="", help="Path to reference CCTV frame")
    parser.add_argument("--out", type=str, default="config_zones.yaml", help="Output YAML path")
    args = parser.parse_args()
    run_annotator(args.image, args.out)

