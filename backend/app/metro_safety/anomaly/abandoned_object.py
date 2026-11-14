"""Abandoned Object (Unattended Luggage / Bag) Detector.

Detects stationary unattended bags, backpacks, and suitcases in monitored zones,
measuring owner separation distance and unattended dwell times.
Runs deep learning object detection at a decimated frame interval to conserve CPU.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import cv2
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager


# COCO Object Class IDs:
# 24: backpack, 26: handbag, 28: suitcase
BAG_CLASS_MAP = {
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}


@dataclass
class TrackedObject:
    """Represents a tracked luggage or bag item."""
    object_id: int
    object_class: str
    box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    first_seen_ts: float
    last_seen_ts: float
    center: Tuple[float, float]
    zone: str = "platform"
    is_unattended: bool = False
    unattended_start_ts: float = 0.0
    unattended_duration: float = 0.0
    owner_seen_near: bool = False
    last_owner_id: Optional[int] = None
    velocity: float = 0.0
    positions: List[Tuple[float, float, float]] = field(default_factory=list) # (x, y, ts)
    snapshot_path: Optional[str] = None
    alerted: bool = False


class AbandonedObjectDetector:
    """Detects and tracks luggage objects, flagging unattended items."""

    def __init__(
        self,
        detect_interval_frames: int = 10,
        unattended_radius_px: float = 130.0,
        unattended_sec: float = 15.0,
        alert_startup_objects: bool = False,
        confidence_thresh: float = 0.20,
        snapshot_dir: str = "data/snapshots",
        yolo_model_name: str = "yolov8n.pt",
        object_detector: Optional[Any] = None
    ):
        self.detect_interval_frames = max(1, detect_interval_frames)
        self.unattended_radius_px = unattended_radius_px
        self.unattended_sec = unattended_sec
        self.alert_startup_objects = alert_startup_objects
        self.confidence_thresh = confidence_thresh
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_model_name = yolo_model_name

        self._obj_detector = object_detector
        self._obj_counter = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.cached_raw_detections: List[Dict[str, Any]] = []
        self._last_inference_frame: int = -1
        self._last_inference_ts: float = 0.0

        if self._obj_detector is None:
            self._init_detector()

    @property
    def yolo_model(self):
        return self._obj_detector

    @yolo_model.setter
    def yolo_model(self, model):
        self._obj_detector = model

    @property
    def object_detector(self):
        return self._obj_detector

    @object_detector.setter
    def object_detector(self, model):
        self._obj_detector = model

    def _init_detector(self):
        try:
            from ultralytics import YOLO
            self._obj_detector = YOLO(self.yolo_model_name)
        except Exception:
            self._obj_detector = None

    def _next_object_id(self) -> int:
        self._obj_counter += 1
        return self._obj_counter

    def detect_objects_in_frame(self, frame: Optional[np.ndarray], frame_idx: int, timestamp: Optional[float] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """Run bag detection at decimated interval.
        Returns:
            (detections, is_fresh)
        """
        now = timestamp if timestamp is not None else time.time()
        # Bound cached detection reuse: expire after 2.5s
        if self._last_inference_ts > 0 and (now - self._last_inference_ts) > 2.5:
            self.cached_raw_detections = []

        should_run = (self._last_inference_frame < 0) or ((frame_idx - self._last_inference_frame) >= self.detect_interval_frames)

        if not should_run:
            return list(self.cached_raw_detections), False

        self._last_inference_frame = frame_idx
        self._last_inference_ts = now

        if self._obj_detector is None or frame is None or frame.size == 0:
            return list(self.cached_raw_detections), False

        try:
            results = self._obj_detector(
                frame,
                imgsz=320,
                conf=self.confidence_thresh,
                classes=list(BAG_CLASS_MAP.keys()),
                verbose=False
            )
            raw = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    for i in range(len(xyxy)):
                        cid = int(cls_ids[i])
                        cname = BAG_CLASS_MAP.get(cid, "bag")
                        raw.append({
                            "box": xyxy[i].copy(),
                            "conf": float(confs[i]),
                            "class": cname
                        })
            self.cached_raw_detections = raw
            return raw, True
        except Exception:
            return list(self.cached_raw_detections), False

    def update_objects(
        self,
        raw_detections: List[Dict[str, Any]],
        timestamp: float,
        zone_manager: ZoneManager,
        is_fresh: bool = True
    ):
        """Associate detections with existing tracked objects."""
        matched_obj_ids = set()

        for det in raw_detections:
            bx = det["box"]
            cx = float((bx[0] + bx[2]) / 2.0)
            cy = float((bx[1] + bx[3]) / 2.0)

            best_id = None
            min_dist = float("inf")

            for oid, tobj in self.tracked_objects.items():
                if oid in matched_obj_ids:
                    continue
                dt = max(0.01, timestamp - tobj.last_seen_ts)
                # Dynamic matching radius accounting for elapsed time and velocity
                match_radius = max(60.0, min(200.0, 60.0 + tobj.velocity * dt * 1.5))
                d = math.hypot(cx - tobj.center[0], cy - tobj.center[1])
                if d < match_radius and d < min_dist:
                    min_dist = d
                    best_id = oid

            zone = zone_manager.get_zone_for_point((cx, cy))

            if best_id is not None:
                # Update existing bag
                tobj = self.tracked_objects[best_id]
                dt = max(0.01, timestamp - tobj.last_seen_ts)
                dist = math.hypot(cx - tobj.center[0], cy - tobj.center[1])
                if is_fresh:
                    tobj.velocity = round(dist / dt, 2)
                    tobj.last_seen_ts = timestamp
                    tobj.positions.append((cx, cy, timestamp))
                    if len(tobj.positions) > 30:
                        tobj.positions.pop(0)
                tobj.box = bx.copy()
                tobj.center = (cx, cy)
                tobj.confidence = det["conf"]
                tobj.zone = zone
                matched_obj_ids.add(best_id)
            else:
                # New bag
                new_id = self._next_object_id()
                self.tracked_objects[new_id] = TrackedObject(
                    object_id=new_id,
                    object_class=det["class"],
                    box=bx.copy(),
                    confidence=det["conf"],
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                    center=(cx, cy),
                    zone=zone,
                    positions=[(cx, cy, timestamp)]
                )
                matched_obj_ids.add(new_id)

        # Prune dead bags after 5 seconds of non-detection
        dead_ids = [oid for oid, obj in self.tracked_objects.items() if (timestamp - obj.last_seen_ts) > 5.0]
        for did in dead_ids:
            del self.tracked_objects[did]

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        frame_idx: int,
        timestamp: float,
        tracked_persons: List[Detection],
        zone_manager: ZoneManager,
        train_state: str = "ABSENT",
        synthetic_objects: Optional[List[Dict[str, Any]]] = None
    ) -> List[SafetyEvent]:
        """Evaluate unattended bags against tracked persons."""
        events: List[SafetyEvent] = []

        if synthetic_objects is not None:
            raw_objs = synthetic_objects
            is_fresh = True
        else:
            raw_objs, is_fresh = self.detect_objects_in_frame(frame, frame_idx, timestamp)

        self.update_objects(raw_objs, timestamp, zone_manager, is_fresh=is_fresh)

        # Person positions
        person_anchors = []
        for p in tracked_persons:
            pid = p.track_id or 0
            foot = p.foot_anchor
            torso = p.torso_center
            bx = p.box
            person_anchors.append({
                "track_id": pid,
                "foot": foot,
                "torso": torso,
                "box": bx
            })

        for oid, obj in self.tracked_objects.items():
            if obj.zone in ["outside", "UNKNOWN"]:
                continue

            # Calculate min distance to any person
            min_dist = 9999.0
            nearest_person_id = None

            for p in person_anchors:
                # Check distance from bag center to person foot, torso, or bbox
                d_foot = math.hypot(obj.center[0] - p["foot"][0], obj.center[1] - p["foot"][1])
                d_torso = math.hypot(obj.center[0] - p["torso"][0], obj.center[1] - p["torso"][1])
                d = min(d_foot, d_torso)
                if d < min_dist:
                    min_dist = d
                    nearest_person_id = p["track_id"]

            is_near_person = min_dist <= self.unattended_radius_px

            if is_near_person:
                # Person is nearby: bag is attended
                obj.owner_seen_near = True
                obj.last_owner_id = nearest_person_id
                obj.is_unattended = False
                obj.unattended_start_ts = timestamp
                obj.unattended_duration = 0.0
                obj.alerted = False
            else:
                # No person nearby: bag is unattended
                if not obj.owner_seen_near and not self.alert_startup_objects:
                    # Ignore bags present at startup that never had an owner near them
                    continue

                # Stationary dwell check: require velocity <= 15.0 px/s (movement pauses/resets dwell)
                if obj.velocity > 15.0:
                    obj.unattended_start_ts = timestamp
                    obj.unattended_duration = 0.0
                    obj.is_unattended = False
                    continue

                if not obj.is_unattended:
                    obj.is_unattended = True
                    obj.unattended_start_ts = timestamp

                obj.unattended_duration = round(timestamp - obj.unattended_start_ts, 1)

                if obj.unattended_duration >= self.unattended_sec:
                    # Capture snapshot crop if frame is available
                    if frame is not None and frame.size > 0 and not obj.snapshot_path:
                        try:
                            h, w = frame.shape[:2]
                            bx = obj.box.astype(int)
                            pad = 25
                            x1 = max(0, bx[0] - pad)
                            y1 = max(0, bx[1] - pad)
                            x2 = min(w, bx[2] + pad)
                            y2 = min(h, bx[3] + pad)
                            crop = frame[y1:y2, x1:x2]
                            crop_fname = f"abandoned_{obj.object_id}_{int(timestamp)}.jpg"
                            crop_full = self.snapshot_dir / crop_fname
                            cv2.imwrite(str(crop_full), crop)
                            obj.snapshot_path = str(crop_full).replace("\\", "/")
                        except Exception:
                            obj.snapshot_path = None

                    ev = SafetyEvent(
                        event_type=EventType.ABANDONED_OBJECT,
                        confidence=min(0.96, 0.85 + 0.05 * min(2.0, obj.unattended_duration / 60.0)),
                        track_id=obj.object_id,
                        timestamp=timestamp,
                        zone=obj.zone,
                        train_state=train_state,
                        evidence_frames=[frame_idx],
                        metadata={
                            "duration_sec": obj.unattended_duration,
                            "object_id": obj.object_id,
                            "object_class": obj.object_class,
                            "min_person_dist_px": round(min_dist, 1),
                            "crop_path": obj.snapshot_path,
                            "last_owner_id": obj.last_owner_id,
                            "severity_code": "L2",
                            "action": f"OPERATOR REVIEW: Unattended {obj.object_class} in {obj.zone} for {obj.unattended_duration}s. Review required."
                        }
                    )
                    ev.is_anomaly = True
                    ev.crop_path = obj.snapshot_path
                    obj.alerted = True
                    events.append(ev)

        return events
