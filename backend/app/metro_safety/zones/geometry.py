"""Zone geometry definitions, validation, and point-in-polygon classifiers.
Includes:
- Robust geometric validation (detecting self-intersecting bowties, < 3 points, tiny areas)
- Automatic repair reordering points around centroid
- Strict zone semantics: platform, yellow_band, track_bed, danger_zone, ignore, and outside
"""
import math
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger("metro_safety.zones.geometry")


def do_segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Return True if line segment (p1, p2) and (p3, p4) properly cross each other."""
    if (p1[0] == p3[0] and p1[1] == p3[1]) or (p1[0] == p4[0] and p1[1] == p4[1]) or \
       (p2[0] == p3[0] and p2[1] == p3[1]) or (p2[0] == p4[0] and p2[1] == p4[1]):
        return False

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def is_polygon_self_intersecting(pts: List[List[float]]) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Check if any non-adjacent edges in the polygon intersect (bowtie / self-intersecting)."""
    n = len(pts)
    if n < 4:
        return False, None

    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if do_segments_intersect(p1, p2, p3, p4):
                return True, (i, j)
    return False, None


def reorder_polygon_around_centroid(pts: List[List[float]]) -> List[List[float]]:
    """Automatic repair: Sort vertices angularly around their centroid to eliminate self-intersections."""
    if len(pts) < 3:
        return [list(p) for p in pts]

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    # Sort in counter-clockwise order by polar angle from centroid
    sorted_pts = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [[float(p[0]), float(p[1])] for p in sorted_pts]


def validate_polygon(pts: List[List[float]], min_area: float = 100.0, name: str = "zone") -> Tuple[bool, Optional[str]]:
    """Validate polygon geometry:
    1. >= 3 points
    2. Non-self-intersecting edges (checked before area since self-intersecting shapes can cancel contour area)
    3. Non-tiny area
    """
    if len(pts) < 3:
        return False, f"Polygon '{name}' has fewer than 3 vertices ({len(pts)} points)"

    self_int, pair = is_polygon_self_intersecting(pts)
    if self_int:
        return False, f"Polygon '{name}' is self-intersecting (edges {pair[0]} and {pair[1]} cross; bowtie topology)"

    area = abs(cv2.contourArea(np.array(pts, dtype=np.float32)))
    if area < min_area:
        return False, f"Polygon '{name}' area is too small ({area:.1f} < {min_area:.1f} px^2)"

    return True, None


class Zone:
    """Represents a validated geometric polygon zone on a CCTV frame."""

    def __init__(self, name: str, polygon: List[Tuple[float, float]], strict_validate: bool = True):
        self.name = name
        poly_list = [[float(pt[0]), float(pt[1])] for pt in polygon]

        if strict_validate:
            valid, err_msg = validate_polygon(poly_list, name=name)
            if not valid:
                logger.error(f"[Zone Error] Zone '{name}' polygon is invalid: {err_msg}")
                raise ValueError(f"Zone '{name}' polygon is invalid: {err_msg}")

        self.polygon = np.array(polygon, dtype=np.int32)
        if len(self.polygon.shape) == 2:
            self.polygon = self.polygon.reshape((-1, 1, 2))

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a 2D point (x, y) lies inside or on the boundary of the polygon."""
        pt = (float(point[0]), float(point[1]))
        dist = cv2.pointPolygonTest(self.polygon, pt, measureDist=False)
        return dist >= 0


class ZoneManager:
    """Manages platform zones and determines which zone a person/object belongs to.
    Strictly differentiates monitored platform zones from 'ignore' and 'outside' areas.
    """

    # Precedence order when checking zones (from innermost danger outwards)
    CHECK_ORDER = ["ignore", "track_bed", "yellow_band", "danger_zone", "platform"]

    def __init__(self, zone_config: Dict[str, List[List[float]]], auto_fix: bool = False):
        self.zones: Dict[str, Zone] = {}
        self.invalid_zones: Dict[str, str] = {}
        self.zone_version: int = 1
        self.baseline_zones: Dict[str, List[List[float]]] = {k: [list(pt) for pt in v] for k, v in zone_config.items()}

        for name, poly in zone_config.items():
            if not poly or len(poly) < 3:
                logger.warning(f"[ZoneManager] Skipping empty/undersized zone '{name}'")
                continue

            valid, err_msg = validate_polygon(poly, name=name)
            if not valid:
                logger.error(f"[ZoneManager] Saved zone '{name}' is INVALID: {err_msg}")
                if auto_fix:
                    logger.warning(f"[ZoneManager] Auto-repairing self-intersecting zone '{name}' around centroid...")
                    fixed_poly = reorder_polygon_around_centroid(poly)
                    repaired_valid, rep_err = validate_polygon(fixed_poly, name=name)
                    if not repaired_valid:
                        self.invalid_zones[name] = f"Auto-repair failed: {rep_err}"
                        raise ValueError(f"Cannot initialize ZoneManager: repaired zone '{name}' is invalid ({rep_err}). Refusing to use invalid zone silently.")
                    self.zones[name] = Zone(name, [tuple(pt) for pt in fixed_poly], strict_validate=True)
                else:
                    self.invalid_zones[name] = err_msg
                    raise ValueError(f"Cannot initialize ZoneManager: saved zone '{name}' is invalid ({err_msg}). Refusing to use invalid zone silently.")
            else:
                self.zones[name] = Zone(name, [tuple(pt) for pt in poly], strict_validate=True)

    def get_zone_for_point(self, point: Tuple[float, float]) -> str:
        """Return the highest priority zone containing the point, or 'outside'.
        If point is in 'ignore' zone, returns 'ignore'.
        If point is not inside any defined zone, returns 'outside'.
        """
        for name in self.CHECK_ORDER:
            if name in self.zones and self.zones[name].contains_point(point):
                return name

        # Check any custom zones
        for name, zone in self.zones.items():
            if name not in self.CHECK_ORDER and zone.contains_point(point):
                return name

        # Strict: Persons not in any monitored platform/track zone are classified as 'outside'
        return "outside"

    def get_polygons(self) -> Dict[str, List[List[float]]]:
        """Return dict of current active polygon vertices."""
        return {name: zone.polygon.reshape(-1, 2).tolist() for name, zone in self.zones.items()}

    def update_zones(self, new_zone_config: Dict[str, List[List[float]]], version: Optional[int] = None) -> bool:
        """Atomically update all zone polygons after validation."""
        temp_zones: Dict[str, Zone] = {}
        for name, poly in new_zone_config.items():
            if not poly or len(poly) < 3:
                logger.error(f"[ZoneManager] Proposed zone '{name}' has fewer than 3 vertices. Rejection atomic.")
                return False
            valid, err_msg = validate_polygon(poly, name=name)
            if not valid:
                logger.error(f"[ZoneManager] Proposed zone '{name}' failed validation ({err_msg}). Rejection atomic.")
                return False
            temp_zones[name] = Zone(name, [tuple(pt) for pt in poly], strict_validate=True)

        # Atomic swap
        self.zones = temp_zones
        if version is not None:
            self.zone_version = version
        else:
            self.zone_version += 1
        return True

    def is_in_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        """Check if point is inside a specific zone."""
        if zone_name in self.zones:
            return self.zones[zone_name].contains_point(point)
        return False

    def is_point_monitored(self, point: Tuple[float, float]) -> bool:
        """Return True if point is inside an active monitored safety zone (not 'ignore' or 'outside')."""
        zone = self.get_zone_for_point(point)
        return zone not in ["outside", "ignore"]

    def draw_zones(
        self,
        frame: np.ndarray,
        alpha: float = 0.35,
        unverified: bool = False,
        draw_baseline: bool = False,
        drift_px: float = 0.0,
        status_badge: str = ""
    ) -> np.ndarray:
        """Draw transparent zone overlays onto the frame.
        Optionally renders baseline faint outlines, unverified styling, and drift telemetry.
        """
        overlay = frame.copy()
        colors = {
            "platform": (0, 200, 0),       # Green
            "yellow_band": (0, 230, 255),   # Yellow
            "track_bed": (0, 0, 220),       # Red
            "danger_zone": (0, 100, 255),   # Orange
            "ignore": (160, 100, 160)       # Muted Purple/Grey for Ignore/Seating
        }

        # Draw baseline faint outline if requested
        if draw_baseline:
            for b_name, b_poly in self.baseline_zones.items():
                if len(b_poly) >= 3:
                    pts_arr = np.array(b_poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts_arr], isClosed=True, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)

        for name, zone in self.zones.items():
            base_color = colors.get(name, (200, 200, 200))
            if unverified:
                # Distinct amber/orange tint when unverified
                color = (0, 165, 255)
            else:
                color = base_color

            cv2.fillPoly(overlay, [zone.polygon], color)
            line_thickness = 1 if unverified else 2
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
            # Add label
            moments = cv2.moments(zone.polygon)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                lbl = f"{name.upper()} [UNVERIFIED]" if unverified else name.upper()
                cv2.putText(frame, lbl, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw alignment status badge and drift vector arrow if requested
        if status_badge:
            h, w = result.shape[:2]
            badge_bg = (20, 20, 180) if unverified else (20, 100, 20)
            cv2.rectangle(result, (w - 380, h - 38), (w - 10, h - 10), badge_bg, -1)
            cv2.rectangle(result, (w - 380, h - 38), (w - 10, h - 10), (200, 200, 200), 1)
            cv2.putText(result, status_badge, (w - 370, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            if drift_px > 0.5:
                # Small drift vector arrow
                arrow_start = (w - 395, h - 24)
                arrow_end = (w - 415, h - 24)
                cv2.arrowedLine(result, arrow_start, arrow_end, (0, 200, 255), 2, tipLength=0.35)

        return result
