"""Person Down (Unresponsive Prone State) Anomaly Detector.

Detects passengers who are prone and unresponsive on the ground over sustained
time windows, differentiating medical emergencies / collapse from temporary
sitting, bending, crouching, or fast fall transitions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager


@dataclass
class ProneTrackState:
    """Tracks temporal posture and prone timers for a single person track."""
    track_id: int
    first_prone_ts: float = 0.0
    last_seen_ts: float = 0.0
    is_currently_prone: bool = False
    prone_duration: float = 0.0
    last_position: Tuple[float, float] = (0.0, 0.0)
    torso_angle: float = 90.0
    aspect_ratio: float = 0.5
    motion_energy: float = 0.0
    recent_torso_pts: List[Tuple[float, float, float]] = field(default_factory=list) # (x, y, ts)
    recent_kpts: List[np.ndarray] = field(default_factory=list)
    linked_fall_id: Optional[str] = None
    l1_alerted: bool = False
    l2_alerted: bool = False


class PersonDownDetector:
    """Evaluates tracked persons for sustained unresponsive prone states."""

    def __init__(
        self,
        prone_t1_sec: float = 10.0,
        prone_t2_sec: float = 30.0,
        horizontal_torso_angle_deg: float = 35.0,
        max_head_hip_y_diff_px: float = 45.0,
        min_aspect_ratio: float = 1.15,
        max_motion_energy: float = 4.0,
        track_reassoc_window_sec: float = 3.0,
        track_reassoc_dist_px: float = 70.0,
        ignore_bench_lying: bool = True
    ):
        self.prone_t1_sec = prone_t1_sec
        self.prone_t2_sec = prone_t2_sec
        self.horizontal_torso_angle_deg = horizontal_torso_angle_deg
        self.max_head_hip_y_diff_px = max_head_hip_y_diff_px
        self.min_aspect_ratio = min_aspect_ratio
        self.max_motion_energy = max_motion_energy
        self.track_reassoc_window_sec = track_reassoc_window_sec
        self.track_reassoc_dist_px = track_reassoc_dist_px
        self.ignore_bench_lying = ignore_bench_lying

        self.track_states: Dict[int, ProneTrackState] = {}
        # Dormant tracks for ID churn re-association: (last_pos, first_prone_ts, last_seen_ts, linked_fall_id)
        self.dormant_states: List[Dict[str, Any]] = []

    def _is_pose_prone(self, det: Detection) -> Tuple[bool, float, float, float, float]:
        """Analyze keypoints and bbox geometry for horizontal prone posture.
        Returns:
            (is_prone, torso_angle_from_horizontal, aspect_ratio, head_hip_dy, motion_energy)
        """
        bx = det.box
        w = max(1.0, float(bx[2] - bx[0]))
        h = max(1.0, float(bx[3] - bx[1]))
        aspect = w / h

        kpts = det.keypoints
        if kpts is None or len(kpts) < 13:
            # Fallback to bbox aspect ratio only
            is_prone = aspect >= self.min_aspect_ratio
            return is_prone, (15.0 if is_prone else 80.0), aspect, 20.0, 0.0

        # COCO Joints:
        # 0: nose, 5: l_shldr, 6: r_shldr, 11: l_hip, 12: r_hip, 15: l_ankle, 16: r_ankle
        shldr_valid = kpts[5][2] > 0.25 and kpts[6][2] > 0.25
        hip_valid = kpts[11][2] > 0.25 and kpts[12][2] > 0.25

        if shldr_valid and hip_valid:
            shldr_mid = (kpts[5][:2] + kpts[6][:2]) / 2.0
            hip_mid = (kpts[11][:2] + kpts[12][:2]) / 2.0
        elif shldr_valid:
            shldr_mid = (kpts[5][:2] + kpts[6][:2]) / 2.0
            hip_mid = np.array([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])
        elif hip_valid:
            hip_mid = (kpts[11][:2] + kpts[12][:2]) / 2.0
            shldr_mid = np.array([(bx[0] + bx[2]) / 2.0, bx[1] + 15.0])
        else:
            shldr_mid = np.array([(bx[0] + bx[2]) / 2.0, bx[1] + 10.0])
            hip_mid = np.array([(bx[0] + bx[2]) / 2.0, bx[3] - 10.0])

        dx = float(shldr_mid[0] - hip_mid[0])
        dy = float(shldr_mid[1] - hip_mid[1])

        # Angle from horizontal axis in degrees
        angle_from_horizontal = abs(math.degrees(math.atan2(abs(dy), abs(dx))))

        # Head-to-hip vertical difference
        nose_y = float(kpts[0][1]) if kpts[0][2] > 0.25 else float(shldr_mid[1])
        head_hip_dy = abs(nose_y - hip_mid[1])

        # Prone posture criteria:
        # 1. Torso angle is close to horizontal (< horizontal_torso_angle_deg, e.g. < 35 deg)
        # 2. Or head-to-hip height difference is low (< max_head_hip_y_diff_px)
        # 3. AND aspect ratio is wide (width/height >= min_aspect_ratio)
        is_prone = (
            (angle_from_horizontal <= self.horizontal_torso_angle_deg or head_hip_dy <= self.max_head_hip_y_diff_px)
            and (aspect >= self.min_aspect_ratio or angle_from_horizontal <= 25.0)
        )

        return is_prone, angle_from_horizontal, aspect, head_hip_dy, 0.0

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_dets: List[Detection],
        zone_manager: ZoneManager,
        train_state: str = "ABSENT",
        active_falls_map: Optional[Dict[int, str]] = None
    ) -> List[SafetyEvent]:
        """Evaluate tracked persons for sustained unresponsive prone states."""
        events: List[SafetyEvent] = []
        active_falls = active_falls_map or {}
        incoming_tids = {det.track_id for det in tracked_dets if det.track_id}

        # Pre-retire missing tracks into dormant_states so newly assigned track IDs
        # in this exact same frame can re-associate immediately
        dead_tids = [t for t in self.track_states if t not in incoming_tids]
        for dead_id in dead_tids:
            st = self.track_states[dead_id]
            if st.is_currently_prone and (timestamp - st.last_seen_ts) <= self.track_reassoc_window_sec:
                self.dormant_states.append({
                    "last_pos": st.last_position,
                    "first_prone_ts": st.first_prone_ts,
                    "last_seen_ts": st.last_seen_ts,
                    "linked_fall_id": st.linked_fall_id,
                    "l1_alerted": st.l1_alerted,
                    "l2_alerted": st.l2_alerted,
                    "recent_torso_pts": list(st.recent_torso_pts)
                })
            del self.track_states[dead_id]

        # Clean old dormant states
        self.dormant_states = [d for d in self.dormant_states if (timestamp - d["last_seen_ts"]) <= self.track_reassoc_window_sec]

        for det in tracked_dets:
            tid = det.track_id or 0
            if tid == 0:
                continue

            foot = det.foot_anchor
            zone = zone_manager.get_zone_for_point(foot)

            # Check bench/seating ignore zone
            if zone == "ignore" and self.ignore_bench_lying:
                if tid in self.track_states:
                    st = self.track_states[tid]
                    st.is_currently_prone = False
                    st.first_prone_ts = timestamp
                    st.prone_duration = 0.0
                    st.l1_alerted = False
                    st.l2_alerted = False
                continue

            # Check track state or re-associate churned track
            if tid not in self.track_states:
                # Attempt spatial-temporal re-association with recently dormant prone tracks
                # Only associate if candidate is posture-compatible (prone), and pick closest
                reassoc = None
                is_cand_prone, _, _, _, _ = self._is_pose_prone(det)
                if is_cand_prone:
                    best_i = None
                    best_dist = float("inf")
                    for i, d in enumerate(self.dormant_states):
                        if (timestamp - d["last_seen_ts"]) <= self.track_reassoc_window_sec:
                            dist = math.hypot(foot[0] - d["last_pos"][0], foot[1] - d["last_pos"][1])
                            if dist <= self.track_reassoc_dist_px and dist < best_dist:
                                best_dist = dist
                                best_i = i

                    if best_i is not None:
                        reassoc = self.dormant_states.pop(best_i)

                if reassoc:
                    self.track_states[tid] = ProneTrackState(
                        track_id=tid,
                        first_prone_ts=reassoc["first_prone_ts"],
                        last_seen_ts=timestamp,
                        is_currently_prone=True,
                        prone_duration=timestamp - reassoc["first_prone_ts"],
                        last_position=foot,
                        linked_fall_id=reassoc["linked_fall_id"],
                        l1_alerted=reassoc["l1_alerted"],
                        l2_alerted=reassoc["l2_alerted"],
                        recent_torso_pts=list(reassoc.get("recent_torso_pts", []))
                    )
                else:
                    linked_f = active_falls.get(tid)
                    self.track_states[tid] = ProneTrackState(
                        track_id=tid,
                        first_prone_ts=timestamp,
                        last_seen_ts=timestamp,
                        last_position=foot,
                        linked_fall_id=linked_f
                    )

            state = self.track_states[tid]
            state.last_seen_ts = timestamp
            state.last_position = foot

            # Link with fall if fall occurred on this track
            if tid in active_falls and not state.linked_fall_id:
                state.linked_fall_id = active_falls[tid]

            is_prone, angle, aspect, head_hip_dy, _ = self._is_pose_prone(det)
            state.torso_angle = round(angle, 1)
            state.aspect_ratio = round(aspect, 2)

            # Track torso kinematics for motion energy (stillness check)
            torso_pt = det.torso_center
            state.recent_torso_pts.append((torso_pt[0], torso_pt[1], timestamp))
            if len(state.recent_torso_pts) > 30:
                state.recent_torso_pts.pop(0)

            # Compute motion velocity over recent 1.5 seconds summing consecutive path displacements with jitter deadband
            recent_window = [p for p in state.recent_torso_pts if (timestamp - p[2]) <= 1.5]
            if len(recent_window) >= 2:
                dt = max(0.1, recent_window[-1][2] - recent_window[0][2])
                path_dist = 0.0
                for k in range(len(recent_window) - 1):
                    d_step = math.hypot(recent_window[k + 1][0] - recent_window[k][0], recent_window[k + 1][1] - recent_window[k][1])
                    if d_step > 1.2:  # deadband for sub-pixel / 1px detection jitter
                        path_dist += d_step
                state.motion_energy = round(path_dist / dt, 2)
            else:
                state.motion_energy = 0.0

            if det.keypoints is not None:
                state.recent_kpts.append(det.keypoints)
                if len(state.recent_kpts) > 24:
                    state.recent_kpts.pop(0)

            # Person is classified prone if posture is horizontal AND keypoints are largely stationary
            is_unresponsive_prone = is_prone and (state.motion_energy <= self.max_motion_energy)

            if is_unresponsive_prone:
                if not state.is_currently_prone:
                    state.is_currently_prone = True
                    state.first_prone_ts = timestamp

                state.prone_duration = round(timestamp - state.first_prone_ts, 1)

                # Check escalation timers: T1 -> L1, T2 -> L2
                if state.prone_duration >= self.prone_t2_sec:
                    if not state.l2_alerted:
                        # L2 Welfare Alert
                        ev = SafetyEvent(
                            event_type=EventType.PERSON_DOWN,
                            confidence=min(0.96, 0.85 + 0.05 * min(2.0, state.prone_duration / 60.0)),
                            track_id=tid,
                            timestamp=timestamp,
                            zone=zone,
                            train_state=train_state,
                            evidence_frames=[frame_idx],
                            metadata={
                                "duration_sec": state.prone_duration,
                                "torso_angle": state.torso_angle,
                                "aspect_ratio": state.aspect_ratio,
                                "motion_energy": state.motion_energy,
                                "severity_cap": "L2",
                                "severity_code": "L2",
                                "disclaimer": "operator welfare review",
                                "linked_fall_id": state.linked_fall_id,
                                "welfare_alert": "Possible medical emergency - person unresponsive"
                            }
                        )
                        ev.is_anomaly = True
                        ev.linked_fall_id = state.linked_fall_id
                        state.l2_alerted = True
                        events.append(ev)

                elif state.prone_duration >= self.prone_t1_sec:
                    if not state.l1_alerted:
                        # L1 Dashboard Notice
                        ev = SafetyEvent(
                            event_type=EventType.PERSON_DOWN,
                            confidence=0.85,
                            track_id=tid,
                            timestamp=timestamp,
                            zone=zone,
                            train_state=train_state,
                            evidence_frames=[frame_idx],
                            metadata={
                                "duration_sec": state.prone_duration,
                                "torso_angle": state.torso_angle,
                                "aspect_ratio": state.aspect_ratio,
                                "motion_energy": state.motion_energy,
                                "severity_cap": "L1",
                                "severity_code": "L1",
                                "disclaimer": "operator welfare review",
                                "linked_fall_id": state.linked_fall_id,
                                "welfare_alert": "Person motionless on floor - monitoring"
                            }
                        )
                        ev.is_anomaly = True
                        ev.linked_fall_id = state.linked_fall_id
                        state.l1_alerted = True
                        events.append(ev)

            else:
                # Person has stood up or resumed active motion: reset prone state
                state.is_currently_prone = False
                state.first_prone_ts = timestamp
                state.prone_duration = 0.0
                state.l1_alerted = False
                state.l2_alerted = False

        return events

