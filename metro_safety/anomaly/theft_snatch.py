"""Heuristic Theft / Bag-Snatch Prototype Detector.

Tracks bag ownership associations (sustained proximity / wrist contact),
and identifies potential snatch patterns:
1. Owner is established via proximity (< 60px) over multiple frames (e.g. >= 5).
2. A different person's wrist approaches the bag (< 50px).
3. The bag rapidly accelerates away (> 12 px/frame).
4. The established owner does not follow within 1.5s (divergent velocity / increasing distance).
5. Hand-over suppression rule: If both persons stay close (< 80px) and move in similar
   velocity vectors, suppress alert (consensual hand-off / companions walking together).

Ground Rules:
- Confidence is strictly capped at 0.70.
- Labeled "POSSIBLE_SNATCH - operator review", NEVER an accusation or confirmed theft.
- Severity capped at L2 (dashboard alert + overlooker).
- Never triggers simulated ETS, never sounds platform L3 siren.
- Anonymous track IDs only.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.anomaly.abandoned_object import TrackedObject


@dataclass
class OwnerLink:
    """Represents an association between a bag object and a person track."""
    track_id: int
    proximity_frames: int = 0
    first_associated_ts: float = 0.0
    last_associated_ts: float = 0.0
    confirmed_owner: bool = False


@dataclass
class SnatchCandidate:
    """Tracks a potential snatch interaction under evaluation."""
    object_id: int
    suspect_track_id: int
    victim_track_id: int
    approach_ts: float
    initial_victim_pos: Tuple[float, float]
    initial_bag_pos: Tuple[float, float]
    suspect_wrist_pos: Tuple[float, float]
    acceleration_detected: bool = False
    alerted: bool = False


class TheftSnatchDetector:
    """Heuristic detector for potential bag snatching / theft events."""

    def __init__(
        self,
        owner_proximity_px: float = 65.0,
        owner_min_frames: int = 4,
        suspect_wrist_proximity_px: float = 55.0,
        snatch_accel_threshold_px: float = 10.0,
        owner_follow_window_sec: float = 1.5,
        handover_max_dist_px: float = 85.0,
        handover_angle_cos_thresh: float = 0.60,  # cos(theta) > 0.6 => angle < ~53 deg
        max_confidence: float = 0.70
    ):
        self.owner_prox_px = owner_proximity_px
        self.owner_min_frames = owner_min_frames
        self.suspect_wrist_prox_px = suspect_wrist_proximity_px
        self.snatch_accel_px = snatch_accel_threshold_px
        self.owner_follow_sec = owner_follow_window_sec
        self.handover_max_dist = handover_max_dist_px
        self.handover_cos_thresh = handover_angle_cos_thresh
        self.max_confidence = min(0.70, max_confidence)

        # object_id -> Dict[track_id, OwnerLink]
        self.bag_owner_links: Dict[int, Dict[int, OwnerLink]] = {}
        # object_id -> confirmed owner track_id
        self.confirmed_owners: Dict[int, int] = {}
        # object_id -> SnatchCandidate
        self.candidates: Dict[int, SnatchCandidate] = {}
        # object history: object_id -> list of (center_x, center_y, ts, speed)
        self.object_velocity_history: Dict[int, List[Tuple[float, float, float, float]]] = {}

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _get_wrists(detection: Detection) -> List[Tuple[float, float]]:
        """Extract left and right wrist keypoints if available."""
        wrists = []
        if detection.keypoints is not None and len(detection.keypoints) >= 11:
            kps = detection.keypoints
            # COCO indices: 9 = left_wrist, 10 = right_wrist
            for idx in (9, 10):
                if idx < len(kps):
                    kp = kps[idx]
                    conf = kp[2] if len(kp) > 2 else 1.0
                    if conf > 0.30:
                        wrists.append((float(kp[0]), float(kp[1])))
        return wrists

    def update(
        self,
        tracked_bags: List[TrackedObject],
        person_detections: List[Detection],
        timestamp: Optional[float] = None,
        camera_motion: bool = False,
        motion_reason: str = "",
        zones_unverified: bool = False,
        zone_version: int = 1
    ) -> List[SafetyEvent]:
        """Update bag-owner associations and evaluate snatch conditions.

        Args:
            tracked_bags: List of luggage objects tracked by AbandonedObjectDetector.
            person_detections: Detected persons with track IDs and optional pose keypoints.
            timestamp: Wall-clock or video timestamp in seconds.
            camera_motion: Flag indicating active camera motion / vibration.
            motion_reason: Description of camera motion.
            zones_unverified: Flag indicating unverified zone baseline.
            zone_version: Zone configuration version.

        Returns:
            List of SafetyEvent instances for POSSIBLE_SNATCH.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[SafetyEvent] = []

        active_bag_ids = {bag.object_id for bag in tracked_bags}
        # Clean up state for departed bags
        for bid in list(self.bag_owner_links.keys()):
            if bid not in active_bag_ids:
                del self.bag_owner_links[bid]
        for bid in list(self.confirmed_owners.keys()):
            if bid not in active_bag_ids:
                del self.confirmed_owners[bid]
        for bid in list(self.object_velocity_history.keys()):
            if bid not in active_bag_ids:
                del self.object_velocity_history[bid]
        for bid in list(self.candidates.keys()):
            if bid not in active_bag_ids:
                del self.candidates[bid]

        person_map: Dict[int, Detection] = {d.track_id: d for d in person_detections if d.track_id is not None}
        person_centers: Dict[int, Tuple[float, float]] = {
            t_id: (float((d.box[0] + d.box[2]) / 2.0), float((d.box[1] + d.box[3]) / 2.0))
            for t_id, d in person_map.items()
        }

        # 1. Update bag velocity history
        for bag in tracked_bags:
            bid = bag.object_id
            bx, by = bag.center
            if bid not in self.object_velocity_history:
                self.object_velocity_history[bid] = []
            
            hist = self.object_velocity_history[bid]
            instant_speed = 0.0
            if hist:
                last_x, last_y, last_t, _ = hist[-1]
                dt = max(0.001, now - last_t)
                instant_speed = math.hypot(bx - last_x, by - last_y) / (dt * 30.0) # approx px/frame at 30fps
            
            hist.append((bx, by, now, instant_speed))
            if len(hist) > 30:
                hist.pop(0)

        # 2. Update owner associations for each bag
        for bag in tracked_bags:
            bid = bag.object_id
            bx, by = bag.center
            if bid not in self.bag_owner_links:
                self.bag_owner_links[bid] = {}

            links = self.bag_owner_links[bid]

            # Find persons near bag
            for pid, pcenter in person_centers.items():
                pdist = self._dist((bx, by), pcenter)
                pdet = person_map[pid]
                wrists = self._get_wrists(pdet)
                wrist_near = any(self._dist(w, (bx, by)) < self.owner_prox_px * 0.85 for w in wrists)

                if pdist < self.owner_prox_px or wrist_near:
                    if pid not in links:
                        links[pid] = OwnerLink(
                            track_id=pid,
                            proximity_frames=1,
                            first_associated_ts=now,
                            last_associated_ts=now
                        )
                    else:
                        link = links[pid]
                        link.proximity_frames += 1
                        link.last_associated_ts = now
                        if link.proximity_frames >= self.owner_min_frames:
                            link.confirmed_owner = True
                            if bid not in self.confirmed_owners:
                                self.confirmed_owners[bid] = pid
                else:
                    if pid in links and not links[pid].confirmed_owner:
                        links[pid].proximity_frames = 0

        # 3. Detect snatch attempts
        for bag in tracked_bags:
            bid = bag.object_id
            bx, by = bag.center
            owner_id = self.confirmed_owners.get(bid)

            # Check existing candidate for this bag
            if bid in self.candidates:
                cand = self.candidates[bid]
                elapsed = now - cand.approach_ts
                if cand.alerted:
                    if elapsed > self.owner_follow_sec + 2.0:
                        del self.candidates[bid]
                    continue

                # Check bag acceleration
                hist = self.object_velocity_history.get(bid, [])
                recent_speeds = [h[3] for h in hist[-5:]] if hist else [0.0]
                max_recent_speed = max(recent_speeds) if recent_speeds else 0.0

                elapsed = now - cand.approach_ts

                # Check hand-over suppression rule:
                # If suspect and victim stay close (< handover_max_dist) and move together
                suspect_center = person_centers.get(cand.suspect_track_id)
                victim_center = person_centers.get(cand.victim_track_id)

                is_handover = False
                if suspect_center and victim_center:
                    pv_dist = self._dist(suspect_center, victim_center)
                    if pv_dist < self.handover_max_dist:
                        # Check directional coherence if available
                        s_det = person_map.get(cand.suspect_track_id)
                        v_det = person_map.get(cand.victim_track_id)
                        s_hist = getattr(s_det, "history", [])
                        v_hist = getattr(v_det, "history", [])
                        if len(s_hist) >= 2 and len(v_hist) >= 2:
                            s_v = (s_hist[-1][0] - s_hist[0][0], s_hist[-1][1] - s_hist[0][1])
                            v_v = (v_hist[-1][0] - v_hist[0][0], v_hist[-1][1] - v_hist[0][1])
                            dot = s_v[0] * v_v[0] + s_v[1] * v_v[1]
                            norm_s = math.hypot(s_v[0], s_v[1])
                            norm_v = math.hypot(v_v[0], v_v[1])
                            if norm_s > 2.0 and norm_v > 2.0:
                                cos_sim = dot / (norm_s * norm_v)
                                if cos_sim > self.handover_cos_thresh:
                                    is_handover = True

                if is_handover:
                    # Suppress alert: consensual handover or companions
                    del self.candidates[bid]
                    # Update owner to the new person
                    self.confirmed_owners[bid] = cand.suspect_track_id
                    continue

                # Check if victim follows within owner_follow_sec
                victim_followed = False
                bag_displacement = self._dist((bx, by), cand.initial_bag_pos)
                if victim_center:
                    victim_disp = self._dist(victim_center, cand.initial_victim_pos)
                    dist_to_bag = self._dist(victim_center, (bx, by))
                    init_dist = self._dist(cand.initial_victim_pos, cand.initial_bag_pos)

                    # If victim remains largely stationary while bag moves away, victim did not follow
                    if victim_disp < 20.0 and bag_displacement > 35.0:
                        victim_followed = False
                    elif dist_to_bag < max(self.owner_prox_px * 1.2, init_dist + 15.0):
                        victim_followed = True

                if max_recent_speed >= self.snatch_accel_px:
                    cand.acceleration_detected = True

                # Snatch conditions met if:
                # 1. Bag accelerated away OR moved significantly away from initial bag pos
                # 2. Window elapsed >= 0.4s and <= owner_follow_sec
                # 3. Victim did NOT follow
                if (cand.acceleration_detected or bag_displacement > 40.0) and not victim_followed:
                    if elapsed >= min(self.owner_follow_sec, 0.5):
                        # Candidate confirmed as POSSIBLE_SNATCH
                        cand.alerted = True

                        raw_conf = 0.65
                        if cand.acceleration_detected:
                            raw_conf += 0.05
                        conf = min(self.max_confidence, raw_conf)

                        is_low_trust = camera_motion or zones_unverified
                        event_metadata: Dict[str, Any] = {
                            "anomaly_type": "POSSIBLE_SNATCH",
                            "severity_cap": "L2",
                            "disclaimer": "possible, operator review",
                            "object_id": bid,
                            "object_class": bag.object_class,
                            "suspect_track_id": cand.suspect_track_id,
                            "victim_track_id": cand.victim_track_id,
                            "bag_displacement_px": round(bag_displacement, 1),
                            "acceleration_detected": cand.acceleration_detected,
                            "low_trust": is_low_trust,
                            "low_trust_reason": motion_reason if camera_motion else ("unverified_zones" if zones_unverified else "")
                        }

                        ev = SafetyEvent(
                            event_type=EventType.POSSIBLE_SNATCH,
                            confidence=conf,
                            track_id=cand.suspect_track_id,
                            timestamp=now,
                            zone=bag.zone,
                            train_state="ABSENT",
                            metadata=event_metadata,
                            camera_motion=camera_motion,
                            motion_reason=motion_reason,
                            zones_unverified=zones_unverified,
                            zone_version=zone_version
                        )
                        events.append(ev)

                # Timeout stale candidate
                if elapsed > self.owner_follow_sec + 2.0:
                    del self.candidates[bid]

            else:
                # Check for new snatch attempt on this bag
                if owner_id is not None:
                    victim_pos = person_centers.get(owner_id)
                    if victim_pos is None:
                        continue

                    # Look for another person (suspect) whose wrist approaches bag
                    for pid, pcenter in person_centers.items():
                        if pid == owner_id:
                            continue

                        pdet = person_map[pid]
                        wrists = self._get_wrists(pdet)
                        # Check wrist distance or tight body distance
                        wrist_dists = [self._dist(w, (bx, by)) for w in wrists]
                        min_wrist_dist = min(wrist_dists) if wrist_dists else self._dist(pcenter, (bx, by))

                        if min_wrist_dist < self.suspect_wrist_prox_px:
                            # Suspect wrist approached bag owned by owner_id!
                            self.candidates[bid] = SnatchCandidate(
                                object_id=bid,
                                suspect_track_id=pid,
                                victim_track_id=owner_id,
                                approach_ts=now,
                                initial_victim_pos=victim_pos,
                                initial_bag_pos=(bx, by),
                                suspect_wrist_pos=wrists[0] if wrists else pcenter
                            )
                            break

        return events
