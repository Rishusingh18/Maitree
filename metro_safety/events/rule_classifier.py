"""Per-track State Machine with Temporal Smoothing and Hysteresis.
Detects Line Crossing, Track Intrusion, Accidental Fall, and Deliberate Jump.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState
from metro_safety.events.models import SafetyEvent, EventType

@dataclass
class TrackObservation:
    frame_idx: int
    timestamp: float
    foot_point: Tuple[float, float]
    torso_point: Tuple[float, float]
    torso_angle: float
    zone: str
    box: np.ndarray

class TrackStateMachine:
    """State machine maintaining temporal window and kinematics for a single person track."""

    def __init__(
        self,
        track_id: int,
        line_crossing_thresh: int = 4,
        track_intrusion_thresh: int = 3,
        fall_drop_vel_thresh: float = 14.0,
        fall_angle_thresh: float = 40.0,
        jump_approach_frames: int = 4,
        jump_vel_thresh: float = 12.0
    ):
        self.track_id = track_id
        self.line_cross_thresh = line_crossing_thresh
        self.track_intrus_thresh = track_intrusion_thresh
        self.fall_vel_thresh = fall_drop_vel_thresh
        self.fall_angle_thresh = fall_angle_thresh
        self.jump_approach_frames = jump_approach_frames
        self.jump_vel_thresh = jump_vel_thresh

        self.observations: List[TrackObservation] = []
        self.current_event: EventType = EventType.NORMAL
        self.confirmed_fall: bool = False
        self.confirmed_jump: bool = False

        # Counters for hysteresis
        self.yellow_consecutive = 0
        self.track_consecutive = 0

    @property
    def latest_kinematics(self) -> Dict[str, Any]:
        """Return latest zone, vertical velocity, and torso angle for debug overlay."""
        if not self.observations:
            return {"zone": "UNKNOWN", "vy": 0.0, "angle": 90.0}
        latest = self.observations[-1]
        vy = 0.0
        if len(self.observations) >= 2:
            vy = latest.torso_point[1] - self.observations[-2].torso_point[1]
        return {
            "zone": latest.zone,
            "vy": round(vy, 1),
            "angle": round(latest.torso_angle, 1)
        }

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        detection: Detection,
        zone_manager: ZoneManager,
        train_state: TrainState
    ) -> Optional[SafetyEvent]:
        """Update track state with new observation and evaluate event logic."""
        foot = detection.foot_anchor
        torso = detection.torso_center
        angle = detection.torso_angle
        zone = zone_manager.get_zone_for_point(foot)

        obs = TrackObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            foot_point=foot,
            torso_point=torso,
            torso_angle=angle,
            zone=zone,
            box=detection.box
        )
        self.observations.append(obs)
        if len(self.observations) > 45:
            self.observations.pop(0)

        # Update zone hysteresis counters
        if zone == "yellow_band":
            self.yellow_consecutive += 1
        else:
            self.yellow_consecutive = max(0, self.yellow_consecutive - 1)

        if zone in ["track_bed", "danger_zone"]:
            self.track_consecutive += 1
        else:
            self.track_consecutive = max(0, self.track_consecutive - 1)

        # Evaluate Fall vs Jump vs Intrusion vs Crossing
        detected_event = self._classify_kinematics(train_state)
        if detected_event:
            return detected_event
        return None

    def _classify_kinematics(self, train_state: TrainState) -> Optional[SafetyEvent]:
        if len(self.observations) < 3:
            return None

        latest = self.observations[-1]
        prev = self.observations[-2]
        oldest = self.observations[0]

        # 1. Check if person is on track / entering track
        if self.track_consecutive >= self.track_intrus_thresh or latest.zone == "track_bed":
            # If already classified as fall or jump, preserve it
            if self.confirmed_fall:
                return self._make_event(EventType.ACCIDENTAL_FALL, 0.95, latest, train_state)
            if self.confirmed_jump:
                return self._make_event(EventType.DELIBERATE_JUMP, 0.92, latest, train_state)

            # Compute vertical drop velocity of torso/center-of-mass across recent frames
            drop_deltas = []
            for i in range(1, min(6, len(self.observations))):
                dy = self.observations[-i].torso_point[1] - self.observations[-i-1].torso_point[1]
                drop_deltas.append(dy)
            max_drop_vel = max(drop_deltas) if drop_deltas else 0.0

            # Compute horizontal edge approach velocity prior to entering track
            # Look at trajectory prior to track entry
            prior_obs = [o for o in self.observations[:-2]]
            approach_vy = 0.0
            if len(prior_obs) >= 3:
                approach_vy = (prior_obs[-1].foot_point[1] - prior_obs[0].foot_point[1]) / float(len(prior_obs))

            # Distinguish ACCIDENTAL_FALL vs DELIBERATE_JUMP
            # Fall: sudden high downward velocity, loss of upright posture (torso angle < 40 deg), no deliberate approach
            if (max_drop_vel >= self.fall_vel_thresh or latest.torso_angle <= self.fall_angle_thresh) and approach_vy < 6.0:
                self.confirmed_fall = True
                return self._make_event(
                    EventType.ACCIDENTAL_FALL,
                    confidence=0.94,
                    obs=latest,
                    train_state=train_state,
                    meta={"drop_vel": round(max_drop_vel, 2), "torso_angle": round(latest.torso_angle, 1)}
                )

            # Jump: deliberate forward/downward velocity into track while keeping controlled leap/posture
            elif approach_vy >= 4.0 or (max_drop_vel >= 10.0 and latest.torso_angle > self.fall_angle_thresh):
                self.confirmed_jump = True
                return self._make_event(
                    EventType.DELIBERATE_JUMP,
                    confidence=0.91,
                    obs=latest,
                    train_state=train_state,
                    meta={"approach_vel": round(approach_vy, 2), "leap_vel": round(max_drop_vel, 2)}
                )

            # Default upright track presence: TRACK_INTRUSION
            else:
                return self._make_event(
                    EventType.TRACK_INTRUSION,
                    confidence=0.90,
                    obs=latest,
                    train_state=train_state,
                    meta={"track_consecutive": self.track_consecutive}
                )

        # 2. Check Line Crossing (in yellow band for N consecutive frames)
        if self.yellow_consecutive >= self.line_cross_thresh:
            return self._make_event(
                EventType.LINE_CROSSING,
                confidence=0.88,
                obs=latest,
                train_state=train_state,
                meta={"yellow_consecutive": self.yellow_consecutive}
            )

        return None

    def _make_event(
        self,
        event_type: EventType,
        confidence: float,
        obs: TrackObservation,
        train_state: TrainState,
        meta: Optional[Dict] = None
    ) -> SafetyEvent:
        frames = [o.frame_idx for o in self.observations[-5:]]
        return SafetyEvent(
            event_type=event_type,
            confidence=confidence,
            track_id=self.track_id,
            timestamp=obs.timestamp,
            zone=obs.zone,
            train_state=train_state.value,
            evidence_frames=frames,
            metadata=meta or {}
        )


class RuleBasedEventClassifier:
    """Manages multi-track state machines across all detected persons."""

    def __init__(
        self,
        line_crossing_thresh: int = 4,
        track_intrusion_thresh: int = 3,
        fall_drop_vel_thresh: float = 14.0,
        fall_angle_thresh: float = 40.0,
        jump_vel_thresh: float = 12.0
    ):
        self.line_thresh = line_crossing_thresh
        self.track_thresh = track_intrusion_thresh
        self.fall_vel_thresh = fall_drop_vel_thresh
        self.fall_angle_thresh = fall_angle_thresh
        self.jump_vel_thresh = jump_vel_thresh
        self.track_machines: Dict[int, TrackStateMachine] = {}

    def get_track_kinematics(self, track_id: int) -> Dict[str, Any]:
        """Fetch latest kinematics for debug display."""
        if track_id in self.track_machines:
            return self.track_machines[track_id].latest_kinematics
        return {"zone": "UNKNOWN", "vy": 0.0, "angle": 90.0}

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_detections: List[Detection],
        zone_manager: ZoneManager,
        train_state: TrainState
    ) -> List[SafetyEvent]:
        """Update each active track machine and collect all triggered safety events."""
        events: List[SafetyEvent] = []
        active_ids = set()

        for det in tracked_detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            active_ids.add(tid)

            if tid not in self.track_machines:
                self.track_machines[tid] = TrackStateMachine(
                    track_id=tid,
                    line_crossing_thresh=self.line_thresh,
                    track_intrusion_thresh=self.track_thresh,
                    fall_drop_vel_thresh=self.fall_vel_thresh,
                    fall_angle_thresh=self.fall_angle_thresh,
                    jump_vel_thresh=self.jump_vel_thresh
                )

            ev = self.track_machines[tid].update(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detection=det,
                zone_manager=zone_manager,
                train_state=train_state
            )
            if ev:
                events.append(ev)

        # Prune inactive tracks
        for dead_id in list(self.track_machines.keys()):
            if dead_id not in active_ids:
                if len(self.track_machines[dead_id].observations) == 0:
                    del self.track_machines[dead_id]

        return events

