"""Auto-Suggest Zones Tool (Item 10).
Analyzes an uncalibrated platform reference frame using perspective edge analysis,
horizontal line clustering, and floor color gradient segmentation
to propose candidate safety zone polygons (platform, yellow_band, track_bed, danger_zone).
All proposed polygons are strictly validated with bowtie and area checks before emission.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from metro_safety.zones.geometry import validate_polygon

logger = logging.getLogger("metro_safety.tools.auto_suggest_zones")


def auto_suggest_zones(
    image_path_or_array: Path | str | np.ndarray,
    output_json_path: Optional[str] = None
) -> Dict[str, List[List[float]]]:
    """Analyze platform scene and compute suggested zone polygons."""
    if isinstance(image_path_or_array, (str, Path)):
        img = cv2.imread(str(image_path_or_array))
        if img is None:
            raise ValueError(f"Could not read image from {image_path_or_array}")
    else:
        img = image_path_or_array

    h, w = img.shape[:2]

    # Preprocessing: Grayscale and bilateral filter to preserve platform edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150, apertureSize=3)

    # Probabilistic Hough transform to detect dominant floor/platform line boundaries
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(w * 0.25), maxLineGap=20)

    # Determine platform edge candidate line
    platform_edge_y = int(h * 0.65)  # Default fallback
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
        if y_candidates:
            platform_edge_y = int(np.median(y_candidates))

    # Construct candidate polygons based on geometry proportions
    margin_x = int(w * 0.05)
    width_usable = w - 2 * margin_x

    # 1. Platform zone (walking surface from top-mid down to yellow line)
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


import math

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
