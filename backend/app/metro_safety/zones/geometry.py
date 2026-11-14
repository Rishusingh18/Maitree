"""Zone geometry definitions and point-in-polygon classifiers."""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

class Zone:
    """Represents a geometric polygon zone on a CCTV frame."""

    def __init__(self, name: str, polygon: List[Tuple[float, float]]):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        if len(self.polygon.shape) == 2:
            self.polygon = self.polygon.reshape((-1, 1, 2))

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a 2D point (x, y) lies inside or on the boundary of the polygon."""
        pt = (float(point[0]), float(point[1]))
        dist = cv2.pointPolygonTest(self.polygon, pt, measureDist=False)
        return dist >= 0

class ZoneManager:
    """Manages platform zones and determines which zone a person/object belongs to."""

    # Precedence order when checking zones (from innermost danger outwards)
    CHECK_ORDER = ["track_bed", "yellow_band", "danger_zone", "platform"]

    def __init__(self, zone_config: Dict[str, List[List[float]]]):
        self.zones: Dict[str, Zone] = {}
        for name, poly in zone_config.items():
            self.zones[name] = Zone(name, [tuple(pt) for pt in poly])

    def get_zone_for_point(self, point: Tuple[float, float]) -> str:
        """Return the highest priority zone containing the point, or 'outside'."""
        for name in self.CHECK_ORDER:
            if name in self.zones and self.zones[name].contains_point(point):
                return name
        # If not in the priority order, check remaining
        for name, zone in self.zones.items():
            if name not in self.CHECK_ORDER and zone.contains_point(point):
                return name
        return "outside"

    def is_in_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        """Check if point is inside a specific zone."""
        if zone_name in self.zones:
            return self.zones[zone_name].contains_point(point)
        return False

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """Draw transparent zone overlays onto the frame."""
        overlay = frame.copy()
        colors = {
            "platform": (0, 200, 0),       # Green
            "yellow_band": (0, 230, 255),   # Yellow
            "track_bed": (0, 0, 220),       # Red
            "danger_zone": (0, 100, 255)    # Orange
        }

        for name, zone in self.zones.items():
            color = colors.get(name, (200, 200, 200))
            cv2.fillPoly(overlay, [zone.polygon], color)
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=color, thickness=2)
            # Add label
            moments = cv2.moments(zone.polygon)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                cv2.putText(frame, name.upper(), (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

