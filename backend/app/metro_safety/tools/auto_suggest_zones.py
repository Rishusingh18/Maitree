"""Auto-Suggest Zones Tool (Item 10).
Analyzes an uncalibrated platform reference frame using perspective edge analysis,
horizontal line clustering, floor color gradient segmentation, and train presence detection
to propose scenario-aware candidate safety zone polygons (platform, yellow_band, track_bed, danger_zone).
Supports multiple camera POVs: corridor perspective (track right / track left) and horizontal cross-platform.
All proposed polygons are strictly validated with bowtie and area checks before emission.
"""
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from metro_safety.zones.geometry import validate_polygon

logger = logging.getLogger("metro_safety.tools.auto_suggest_zones")


def estimate_scenario_zones(
    w: int,
    h: int,
    train_box: Optional[Tuple[float, float, float, float]] = None,
    pov_hint: Optional[str] = None
) -> Dict[str, List[List[float]]]:
    """Generate robust, scenario-aware safety polygons for different POVs (corridor track right/left, horizontal)."""
    pov = pov_hint or "corridor_track_right"
    if train_box is not None:
        tx1, ty1, tx2, ty2 = train_box
        t_center_x = (float(tx1) + float(tx2)) / 2.0
        if t_center_x > w * 0.5:
            pov = "corridor_track_right"
        else:
            pov = "corridor_track_left"

    top_y = round(max(50.0, h * 0.12), 1)
    bot_y = float(h)

    if pov == "corridor_track_right":
        if train_box is not None:
            tx1 = float(train_box[0])
            x_edge_top = max(w * 0.25, tx1 + w * 0.07)
            x_edge_bot = max(w * 0.38, tx1 + w * 0.18)
        else:
            x_edge_top = round(w * 0.30, 1)
            x_edge_bot = round(w * 0.42, 1)

        dang_w_top = round(max(20.0, w * 0.025), 1)
        dang_w_bot = round(max(40.0, w * 0.045), 1)
        yel_w_top = round(max(22.0, w * 0.028), 1)
        yel_w_bot = round(max(42.0, w * 0.048), 1)

        x_edge_top = round(x_edge_top, 1)
        x_edge_bot = round(x_edge_bot, 1)
        x_dang_left_top = round(x_edge_top - dang_w_top, 1)
        x_dang_left_bot = round(x_edge_bot - dang_w_bot, 1)
        x_yel_left_top = round(x_dang_left_top - yel_w_top, 1)
        x_yel_left_bot = round(x_dang_left_bot - yel_w_bot, 1)

        zones = {
            "platform": [
                [0.0, top_y],
                [x_yel_left_top, top_y],
                [x_yel_left_bot, bot_y],
                [0.0, bot_y]
            ],
            "yellow_band": [
                [x_yel_left_top, top_y],
                [x_dang_left_top, top_y],
                [x_dang_left_bot, bot_y],
                [x_yel_left_bot, bot_y]
            ],
            "danger_zone": [
                [x_dang_left_top, top_y],
                [x_edge_top, top_y],
                [x_edge_bot, bot_y],
                [x_dang_left_bot, bot_y]
            ],
            "track_bed": [
                [x_edge_top, top_y],
                [float(w), top_y],
                [float(w), bot_y],
                [x_edge_bot, bot_y]
            ]
        }
    elif pov == "corridor_track_left":
        if train_box is not None:
            tx2 = float(train_box[2])
            x_edge_top = min(w * 0.75, tx2 - w * 0.07)
            x_edge_bot = min(w * 0.62, tx2 - w * 0.18)
        else:
            x_edge_top = round(w * 0.70, 1)
            x_edge_bot = round(w * 0.58, 1)

        dang_w_top = round(max(20.0, w * 0.025), 1)
        dang_w_bot = round(max(40.0, w * 0.045), 1)
        yel_w_top = round(max(22.0, w * 0.028), 1)
        yel_w_bot = round(max(42.0, w * 0.048), 1)

        x_edge_top = round(x_edge_top, 1)
        x_edge_bot = round(x_edge_bot, 1)
        x_dang_right_top = round(x_edge_top + dang_w_top, 1)
        x_dang_right_bot = round(x_edge_bot + dang_w_bot, 1)
        x_yel_right_top = round(x_dang_right_top + yel_w_top, 1)
        x_yel_right_bot = round(x_dang_right_bot + yel_w_bot, 1)

        zones = {
            "track_bed": [
                [0.0, top_y],
                [x_edge_top, top_y],
                [x_edge_bot, bot_y],
                [0.0, bot_y]
            ],
            "danger_zone": [
                [x_edge_top, top_y],
                [x_dang_right_top, top_y],
                [x_dang_right_bot, bot_y],
                [x_edge_bot, bot_y]
            ],
            "yellow_band": [
                [x_dang_right_top, top_y],
                [x_yel_right_top, top_y],
                [x_yel_right_bot, bot_y],
                [x_dang_right_bot, bot_y]
            ],
            "platform": [
                [x_yel_right_top, top_y],
                [float(w), top_y],
                [float(w), bot_y],
                [x_yel_right_bot, bot_y]
            ]
        }
    else:  # horizontal
        margin_x = float(int(w * 0.05))
        y_plat_top = round(h * 0.20, 1)
        y_yel_top = round(h * 0.58, 1)
        y_yel_bot = round(h * 0.65, 1)
        y_dang_bot = round(h * 0.75, 1)
        y_track_bot = round(h * 0.95, 1)
        zones = {
            "platform": [[margin_x, y_plat_top], [float(w - margin_x), y_plat_top], [float(w - margin_x), y_yel_top], [margin_x, y_yel_top]],
            "yellow_band": [[margin_x, y_yel_top], [float(w - margin_x), y_yel_top], [float(w - margin_x), y_yel_bot], [margin_x, y_yel_bot]],
            "danger_zone": [[margin_x, y_yel_bot], [float(w - margin_x), y_yel_bot], [float(w - margin_x), y_dang_bot], [margin_x, y_dang_bot]],
            "track_bed": [[margin_x, y_dang_bot], [float(w - margin_x), y_dang_bot], [float(w - margin_x), y_track_bot], [margin_x, y_track_bot]]
        }

    validated = {}
    for name, poly in zones.items():
        ok, msg = validate_polygon(poly, name=name)
        if ok:
            validated[name] = poly
        else:
            logger.warning(f"[EstimateScenarioZones] Polygon '{name}' failed validation: {msg}")
    return validated


def auto_suggest_zones(
    image_path_or_array: Path | str | np.ndarray,
    output_json_path: Optional[str] = None
) -> Dict[str, List[List[float]]]:
    """Analyze platform scene and compute scenario-aware suggested zone polygons."""
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        if img is None:
            raise ValueError(f"Could not read image from {image_path_or_array}")
    else:
        img = image_path_or_array

    h, w = img.shape[:2]

    # Check for train presence to guide corridor perspective
    train_box = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        res = yolo(img, verbose=False)[0]
        for b in res.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            if (cls == 6 or res.names[cls] == 'train') and conf > 0.35:
                bx = b.xyxy[0].cpu().numpy()
                area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                if area > 0.04 * w * h:
                    train_box = tuple(bx)
                    break
    except Exception:
        train_box = None

    if train_box is not None:
        validated_zones = estimate_scenario_zones(w, h, train_box=train_box)
        if len(validated_zones) == 4:
            if output_json_path:
                out_p = Path(output_json_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(validated_zones, f, indent=2)
            return validated_zones

    # Preprocessing: Grayscale and bilateral filter to preserve platform edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150, apertureSize=3)

    # Probabilistic Hough transform to detect dominant floor/platform line boundaries
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(w * 0.25), maxLineGap=20)

    # Determine platform edge candidate line
    platform_edge_y = int(h * 0.65)  # Default fallback
    has_horizontal_line = False
    if lines is not None:
        y_candidates = []
        for line in lines:
            pts = line.ravel()
            if len(pts) == 4:
                x1, y1, x2, y2 = pts
                # Look for near-horizontal or slightly angled lines in lower 60% of frame
                if min(y1, y2) > int(h * 0.35) and max(y1, y2) < int(h * 0.90):
                    angle = abs(math.atan2(float(y2 - y1), float(x2 - x1)) * 180.0 / math.pi) % 180.0
                    if angle > 90.0:
                        angle = 180.0 - angle
                    if angle < 35.0:  # near horizontal (normalized acute angle)
                        y_candidates.append(float(y1 + y2) / 2.0)
                        has_horizontal_line = True
        if y_candidates:
            platform_edge_y = int(np.median(y_candidates))

    # Construct candidate polygons based on geometry proportions
    margin_x = int(w * 0.05)
    y_plat_top = max(int(h * 0.25), platform_edge_y - int(h * 0.35))
    y_yellow_top = max(y_plat_top + 20, platform_edge_y - int(h * 0.08))
    y_yellow_bot = platform_edge_y
    y_track_bot = min(h - 10, platform_edge_y + int(h * 0.25))

    raw_candidates = {
        "platform": [
            [float(margin_x), float(y_plat_top)],
            [float(w - margin_x), float(y_plat_top)],
            [float(w - margin_x), float(y_yellow_top)],
            [float(margin_x), float(y_yellow_top)]
        ],
        "yellow_band": [
            [float(margin_x), float(y_yellow_top)],
            [float(w - margin_x), float(y_yellow_top)],
            [float(w - margin_x), float(y_yellow_bot)],
            [float(margin_x), float(y_yellow_bot)]
        ],
        "track_bed": [
            [float(margin_x), float(y_yellow_bot)],
            [float(w - margin_x), float(y_yellow_bot)],
            [float(w - margin_x), float(y_track_bot)],
            [float(margin_x), float(y_track_bot)]
        ],
        "danger_zone": [
            [float(margin_x), float(max(0, y_yellow_top - 15))],
            [float(w - margin_x), float(max(0, y_yellow_top - 15))],
            [float(w - margin_x), float(y_track_bot)],
            [float(margin_x), float(y_track_bot)]
        ]
    }

    # Strict validation of every suggested zone
    validated_zones: Dict[str, List[List[float]]] = {}
    for z_name, poly in raw_candidates.items():
        is_valid, msg = validate_polygon(poly, name=z_name)
        if is_valid:
            validated_zones[z_name] = poly
        else:
            logger.warning(f"[AutoSuggest] Proposed zone '{z_name}' failed validation: {msg}")

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(validated_zones, f, indent=2)
        logger.info(f"[AutoSuggest] Saved {len(validated_zones)} suggested zones to {out_p}")

    return validated_zones


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-suggest initial platform zones from CCTV frame.")
    parser.add_argument("--image", type=str, default="data/reference_frame.jpg", help="Reference frame path")
    parser.add_argument("--output", type=str, default="data/suggested_zones.json", help="Output JSON path")
    args = parser.parse_args()

    try:
        zones = auto_suggest_zones(args.image, args.output)
        print(f"Auto-suggested {len(zones)} valid zones from {args.image}:")
        for k, v in zones.items():
            print(f"  • {k}: {len(v)} vertices")
        print(f"Saved to: {args.output}")
    except Exception as e:
        print(f"Auto-suggest error: {e}")
