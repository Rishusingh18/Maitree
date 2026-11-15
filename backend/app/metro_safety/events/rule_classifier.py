"""Per-track State Machine with Temporal Smoothing and Hysteresis.
Detects Line Crossing, Track Intrusion, Accidental Fall, and Deliberate Jump.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np

from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState
from metro_safety.events.models import SafetyEvent, EventType

logger = logging.getLogger("metro_safety.events.rule_classifier")

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
        jump_vel_thresh: float = 12.0,
        min_event_frames: int = 4,
        min_drop_displacement: float = 10.0,
        min_jump_displacement: float = 8.0
    ):
        self.track_id = track_id
        self.line_cross_thresh = line_crossing_thresh
        self.track_intrus_thresh = track_intrusion_thresh
        self.fall_vel_thresh = fall_drop_vel_thresh
        self.fall_angle_thresh = fall_angle_thresh
        self.jump_approach_frames = jump_approach_frames
        self.jump_vel_thresh = jump_vel_thresh
        self.min_event_frames = min_event_frames
        self.min_drop_displacement = min_drop_displacement
        self.min_jump_displacement = min_jump_displacement

        self.observations: List[TrackObservation] = []
        self.current_event: EventType = EventType.NORMAL
        self.confirmed_fall: bool = False
        self.confirmed_jump: bool = False
        self.fall_motion_induced: bool = False
        self.jump_motion_induced: bool = False

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
        train_state: TrainState,
        geofence_active: bool = True
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

        # Update zone hysteresis counters - reset when zone is exited to bound hysteresis
        if zone == "yellow_band":
            self.yellow_consecutive += 1
        else:
            self.yellow_consecutive = 0

        if zone in ["track_bed", "danger_zone"]:
            self.track_consecutive += 1
        else:
            self.track_consecutive = 0

        # Evaluate Fall vs Jump vs Intrusion vs Crossing
        detected_event = self._classify_kinematics(train_state, geofence_active=geofence_active)
        if detected_event:
            return detected_event
        return None

    def _classify_kinematics(self, train_state: TrainState, geofence_active: bool = True) -> Optional[SafetyEvent]:
        if len(self.observations) < 2:
            return None

        latest = self.observations[-1]
        prev = self.observations[-2]
        oldest = self.observations[0]

        # Check if train is stopped / docked at the platform
        is_train_docked = (
            not geofence_active
            or train_state in (TrainState.STOPPED_DOORS_OPEN, TrainState.PRESENT)
        )

        # 1. Check if person is on track / entering track
        if self.track_consecutive >= self.track_intrus_thresh or latest.zone in ["track_bed", "danger_zone"]:
            # If train is docked at platform and person is in danger clearance zone (train doorway),
            # upright passengers boarding or standing in doorway are NOT jumpers or intruders.
            if is_train_docked and latest.zone == "danger_zone":
                # Only severe accidental fall into the gap (severe vertical drop + collapsed posture) is flagged
                w_gap_len = min(len(self.observations), 6)
                cum_gap_disp = latest.torso_point[1] - self.observations[-w_gap_len].torso_point[1]
                if (
                    len(self.observations) >= self.min_event_frames
                    and latest.torso_angle <= 30.0
                    and cum_gap_disp >= 25.0
                ):
                    self.confirmed_fall = True
                    return self._make_event(
                        EventType.ACCIDENTAL_FALL,
                        confidence=0.88,
                        obs=latest,
                        train_state=train_state,
                        meta={"gap_fall": True}
                    )
                # Upright passenger boarding/alighting in doorway: safe, do not trigger false intrusion or jump
                return None

            # If already classified as fall or jump, preserve it (capped at 0.95/0.92)
            if self.confirmed_fall:
                return self._make_event(EventType.ACCIDENTAL_FALL, 0.95, latest, train_state)
            if self.confirmed_jump:
                return self._make_event(EventType.DELIBERATE_JUMP, 0.92, latest, train_state)

            # Compute vertical drop velocity of torso/center-of-mass across recent frames, accounting for frame gaps
            drop_deltas = []
            for i in range(1, min(6, len(self.observations))):
                gap = self.observations[-i].frame_idx - self.observations[-i-1].frame_idx
                # Preserve fall evidence across stride-four detector sampling (gap <= 5)
                if gap > 5:
                    drop_deltas.append(0.0)
                else:
                    dy = (self.observations[-i].torso_point[1] - self.observations[-i-1].torso_point[1]) / max(1, gap)
                    drop_deltas.append(dy)
            max_drop_vel = max(drop_deltas) if drop_deltas else 0.0
            positive_drops = sum(1 for dy in drop_deltas if dy > 0.5)

            # Compute cumulative vertical drop displacement across recent window
            w_disp_len = min(len(self.observations), 6)
            cum_vert_disp = latest.torso_point[1] - self.observations[-w_disp_len].torso_point[1]

            # Compute horizontal edge approach velocity prior to entering track
            prior_obs = [o for o in self.observations[:-2]]
            approach_vy = 0.0
            if len(prior_obs) >= 3:
                approach_vy = (prior_obs[-1].foot_point[1] - prior_obs[0].foot_point[1]) / float(len(prior_obs))

            # False-alert gates: A jump or fall requires sustained motion & minimum displacement
            is_sustained_duration = len(self.observations) >= self.min_event_frames

            # Fall: sudden high downward velocity, loss of upright posture (torso angle < 40 deg),
            # sustained motion and minimum displacement without deliberate sprint approach
            if (
                is_sustained_duration
                and cum_vert_disp >= self.min_drop_displacement
                and (max_drop_vel >= self.fall_vel_thresh or latest.torso_angle <= self.fall_angle_thresh)
                and approach_vy < 6.0
                and positive_drops >= 2
            ):
                self.confirmed_fall = True
                calibrated_conf = min(0.98, max(0.85, 0.88 + 0.08 * min(1.0, cum_vert_disp / 40.0)))
                return self._make_event(
                    EventType.ACCIDENTAL_FALL,
                    confidence=calibrated_conf,
                    obs=latest,
                    train_state=train_state,
                    meta={
                        "drop_vel": round(max_drop_vel, 2),
                        "torso_angle": round(latest.torso_angle, 1),
                        "cum_disp": round(cum_vert_disp, 1)
                    }
                )

            # Jump: deliberate forward/downward velocity into track while keeping controlled leap/posture
            elif (
                is_sustained_duration
                and (cum_vert_disp >= self.min_jump_displacement or approach_vy >= 4.0)
                and (approach_vy >= 4.0 or (max_drop_vel >= 10.0 and latest.torso_angle > self.fall_angle_thresh))
            ):
                self.confirmed_jump = True
                calibrated_conf = min(0.97, max(0.84, 0.86 + 0.08 * min(1.0, max_drop_vel / 20.0)))
                return self._make_event(
                    EventType.DELIBERATE_JUMP,
                    confidence=calibrated_conf,
                    obs=latest,
                    train_state=train_state,
                    meta={
                        "approach_vel": round(approach_vy, 2),
                        "leap_vel": round(max_drop_vel, 2),
                        "cum_disp": round(cum_vert_disp, 1)
                    }
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

        # 2. Check Line Crossing (in yellow band for N consecutive frames, gated by geofence status)
        if geofence_active and self.yellow_consecutive >= self.line_cross_thresh:
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
        # Never report exactly 100% (1.0); cap at 0.99
        calibrated_conf = min(0.99, max(0.50, float(confidence)))
        kine = self.latest_kinematics
        logger.debug(
            f"[SafetyEvent] Track #{self.track_id:02d} | Event: {event_type.value} | "
            f"Zone: {obs.zone} | vy: {kine['vy']:+05.1f} px/f | angle: {kine['angle']:04.1f} deg | "
            f"Conf: {calibrated_conf:.2f}"
        )
        return SafetyEvent(
            event_type=event_type,
            confidence=calibrated_conf,
            track_id=self.track_id,
            timestamp=obs.timestamp,
            zone=obs.zone,
            train_state=train_state.value if hasattr(train_state, "value") else str(train_state),
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
        jump_vel_thresh: float = 12.0,
        min_event_frames: int = 4,
        min_drop_displacement: float = 10.0,
        min_jump_displacement: float = 8.0
    ):
        self.line_thresh = line_crossing_thresh
        self.track_thresh = track_intrusion_thresh
        self.fall_vel_thresh = fall_drop_vel_thresh
        self.fall_angle_thresh = fall_angle_thresh
        self.jump_vel_thresh = jump_vel_thresh
        self.min_event_frames = min_event_frames
        self.min_drop_displacement = min_drop_displacement
        self.min_jump_displacement = min_jump_displacement
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
        train_state: TrainState,
        camera_motion: bool = False,
        motion_reason: str = "",
        geofence_active: Optional[bool] = None,
        train_state_provider: Optional[Any] = None,
        platform_id: Optional[str] = None
    ) -> List[SafetyEvent]:
        """Update each active track machine and collect all triggered safety events."""
        events: List[SafetyEvent] = []
        active_ids = set()

        # Resolve geofence active status (fail-safe is active)
        eff_geofence_active = True
        if geofence_active is not None:
            eff_geofence_active = bool(geofence_active)
        elif train_state_provider is not None and hasattr(train_state_provider, "is_geofence_active"):
            try:
                eff_geofence_active = train_state_provider.is_geofence_active(
                    platform_id=platform_id,
                    timestamp=timestamp
                )
            except Exception as e:
                logger.warning(f"Error checking is_geofence_active on provider: {e}")
                eff_geofence_active = True
        else:
            st_val = train_state.value.upper() if hasattr(train_state, "value") else str(train_state).upper()
            if st_val in ("STOPPED_DOORS_OPEN", "PRESENT"):
                eff_geofence_active = False
            else:
                eff_geofence_active = True

        for det in tracked_detections:
            if det.track_id is None:
                continue

            # Strict zone semantics: ignore detections outside monitored zones or in 'ignore' zone
            foot = det.foot_anchor
            zone = zone_manager.get_zone_for_point(foot)
            if zone in ["outside", "ignore"]:
                logger.debug(f"[RuleClassifier Suppressed] Track #{det.track_id} at {foot} is in '{zone}' zone. Ignoring.")
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
                    jump_vel_thresh=self.jump_vel_thresh,
                    min_event_frames=self.min_event_frames,
                    min_drop_displacement=self.min_drop_displacement,
                    min_jump_displacement=self.min_jump_displacement
                )

            if not camera_motion:
                if self.track_machines[tid].fall_motion_induced:
                    self.track_machines[tid].confirmed_fall = False
                    self.track_machines[tid].fall_motion_induced = False
                    self.track_machines[tid].positive_drop_frames = 0
                    self.track_machines[tid].drop_deltas = []
                    # Discard contaminated prior observations before kinematics resumes
                    self.track_machines[tid].observations = self.track_machines[tid].observations[-1:]
                if self.track_machines[tid].jump_motion_induced:
                    self.track_machines[tid].confirmed_jump = False
                    self.track_machines[tid].jump_motion_induced = False
                    self.track_machines[tid].positive_jump_frames = 0
                    self.track_machines[tid].jump_deltas = []
                    # Discard contaminated prior observations before kinematics resumes
                    self.track_machines[tid].observations = self.track_machines[tid].observations[-1:]

            pre_fall = self.track_machines[tid].confirmed_fall
            pre_jump = self.track_machines[tid].confirmed_jump

            ev = self.track_machines[tid].update(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detection=det,
                zone_manager=zone_manager,
                train_state=train_state,
                geofence_active=eff_geofence_active
            )
            if ev:
                ev.camera_motion = bool(camera_motion)
                ev.motion_reason = str(motion_reason)
                if camera_motion:
                    if self.track_machines[tid].confirmed_fall and not pre_fall:
                        self.track_machines[tid].fall_motion_induced = True
                    if self.track_machines[tid].confirmed_jump and not pre_jump:
                        self.track_machines[tid].jump_motion_induced = True
                events.append(ev)

        # Prune inactive tracks after a bounded recovery interval (30 frames / ~1 second)
        max_recovery_frames = 30
        for dead_id in list(self.track_machines.keys()):
            if dead_id not in active_ids:
                obs_list = self.track_machines[dead_id].observations
                last_f = obs_list[-1].frame_idx if obs_list else 0
                if not obs_list or (frame_idx - last_f) > max_recovery_frames:
                    del self.track_machines[dead_id]

        return events

