"""Silent distress gesture detector module.
Supports:
1. Signal for Help (open palm raised -> thumb tuck -> fingers fold)
2. Long-range CCTV fallback: Wrists crossed overhead (both wrists crossed above head level)
With anti-false-positive temporal persistence and waving rejection logic.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from metro_safety.perception.base import Detection
from metro_safety.events.models import SafetyEvent, EventType
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState

@dataclass
class GestureObservation:
    frame_idx: int
    timestamp: float
    wrist_l: Tuple[float, float, float]  # x, y, conf
    wrist_r: Tuple[float, float, float]
    nose: Tuple[float, float, float]
    sh_l: Tuple[float, float, float]
    sh_r: Tuple[float, float, float]
    signal_for_help_stage: int            # 0: none, 1: palm open, 2: thumb/fingers folded
    wrists_crossed: bool
    gesture_variant: str = "WRISTS_CROSSED_OVERHEAD"

def _ccw(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> bool:
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

def _segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))

class TrackGestureStateMachine:
    """State machine tracking distress gestures per person track."""

    def __init__(
        self,
        track_id: int,
        detection_window: int = 20,
        min_persistence: int = 2,
        cooldown_frames: int = 20,
        use_macro_fallback: bool = True
    ):
        self.track_id = track_id
        self.window = detection_window
        self.min_persistence = min_persistence
        self.cooldown = cooldown_frames
        self.use_macro = use_macro_fallback

        self.history: List[GestureObservation] = []
        self.crossed_persist_count = 0
        self.signal_persist_count = 0
        self.last_trigger_frame = -999
        self.detected_cross_variant: str = "WRISTS_CROSSED_OVERHEAD"

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        detection: Detection,
        zone_manager: ZoneManager,
        train_state: TrainState
    ) -> Optional[SafetyEvent]:
        """Evaluate keypoints for distress gestures."""
        if frame_idx - self.last_trigger_frame < self.cooldown:
            return None

        kpts = detection.keypoints
        if kpts is None or len(kpts) < 11:
            return None

        # Keypoints extraction
        # 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
        nose = tuple(kpts[0])
        sh_l = tuple(kpts[5])
        sh_r = tuple(kpts[6])
        el_l = tuple(kpts[7]) if len(kpts) > 7 else (0.0, 0.0, 0.0)
        el_r = tuple(kpts[8]) if len(kpts) > 8 else (0.0, 0.0, 0.0)
        w_l = tuple(kpts[9])
        w_r = tuple(kpts[10])

        sh_y = (sh_l[1] + sh_r[1]) / 2.0
        sh_width = max(10.0, abs(sh_r[0] - sh_l[0]))
        sh_dx = sh_r[0] - sh_l[0]
        wrist_dx = w_r[0] - w_l[0]

        # Head level reference: robust to nose occlusion when hands cover the face/hair
        head_y = nose[1] if (len(nose) > 2 and nose[2] > 0.25) else (sh_y - sh_width * 0.45)

        # --- 1. Macro Gesture: Wrists crossed / Arms crossed ---
        wrists_crossed = False
        detected_variant = "WRISTS_CROSSED_OVERHEAD"

        if w_l[2] > 0.20 and w_r[2] > 0.20 and sh_l[2] > 0.20 and sh_r[2] > 0.20:
            # Elevation check:
            # 1. Overhead cross: wrists above shoulders and near or above head level
            is_overhead = (w_l[1] <= head_y + 15) and (w_r[1] <= head_y + 15) and (w_l[1] < sh_y - 10) and (w_r[1] < sh_y - 10)
            # 2. Chest/torso cross: wrists in front of chest/neck
            is_chest = (not is_overhead) and (w_l[1] < sh_y + sh_width * 1.1) and (w_r[1] < sh_y + sh_width * 1.1) and (w_l[1] > head_y - 40) and (w_r[1] > head_y - 40)

            if is_overhead or is_chest:
                # Geometric crossing verification (humanly cross vs artificial non-cross)
                # A. 2D arm segment intersection
                arm_intersect = _segments_intersect((sh_l[0], sh_l[1]), (w_l[0], w_l[1]), (sh_r[0], sh_r[1]), (w_r[0], w_r[1]))
                forearm_intersect = False
                if el_l[2] > 0.20 and el_r[2] > 0.20:
                    forearm_intersect = _segments_intersect((el_l[0], el_l[1]), (w_l[0], w_l[1]), (el_r[0], el_r[1]), (w_r[0], w_r[1]))

                # B. Wrist inversion relative to shoulders
                is_inverted = (sh_dx * wrist_dx <= 0)

                # C. Wrist proximity & convergence
                wrist_dist = float(np.hypot(w_l[0] - w_r[0], w_l[1] - w_r[1]))
                wrists_touching = (wrist_dist <= sh_width * 0.48)

                # Check whether arms converge towards torso center vs parallel hands
                sh_center_x = (sh_l[0] + sh_r[0]) / 2.0
                wrist_center_x = (w_l[0] + w_r[0]) / 2.0
                centered = abs(wrist_center_x - sh_center_x) < (sh_width * 0.85)

                # Reject artificial parallel hands: wrists on natural sides without intersecting and far apart (> 0.48 * sh_width)
                is_parallel = (sh_dx * wrist_dx > 0) and not arm_intersect and not forearm_intersect and (wrist_dist > sh_width * 0.48)

                if not is_parallel:
                    if arm_intersect or forearm_intersect or (is_inverted and wrist_dist < sh_width * 0.95) or (wrists_touching and centered):
                        wrists_crossed = True
                        detected_variant = "WRISTS_CROSSED_OVERHEAD" if is_overhead else "ARMS_CROSSED_CHEST"
                        self.detected_cross_variant = detected_variant

        # --- 2. Micro Gesture: Signal for Help (Raised Hand sequence) ---
        # Note: 17-point body pose keypoints cannot observe individual thumb-tuck / finger-fold
        # sequences required for Signal for Help. To avoid false positives from simple raised hands,
        # stage is set to 0 unless full hand keypoints can confirm the sequence.
        r_raised = (w_r[2] > 0.4 and w_r[1] < sh_y - 15)
        l_raised = (w_l[2] > 0.4 and w_l[1] < sh_y - 15)
        hand_raised = r_raised or l_raised

        obs = GestureObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            wrist_l=w_l,
            wrist_r=w_r,
            nose=nose,
            sh_l=sh_l,
            sh_r=sh_r,
            signal_for_help_stage=0,  # Disabled for 17-point body pose without finger keypoints
            wrists_crossed=wrists_crossed,
            gesture_variant=detected_variant
        )
        self.history.append(obs)
        if len(self.history) > self.window:
            self.history.pop(0)

        # Anti-False-Positive: Waving Rejection
        # Waving is an alternating wave of a SINGLE raised hand, not both arms crossed in an X.
        is_waving = False
        def _check_wrist_waving(wrist_selector):
            xs = [wrist_selector(o)[0] for o in self.history if wrist_selector(o)[2] > 0.3]
            if len(xs) >= 8:
                x_diffs = np.diff(xs)
                sign_changes = np.sum(np.diff(np.sign(x_diffs)) != 0)
                min_span = max(35.0, sh_width * 0.35)
                if sign_changes >= 4 and (np.max(xs) - np.min(xs)) > min_span:
                    return True
            return False

        if r_raised and not wrists_crossed and _check_wrist_waving(lambda o: o.wrist_r):
            is_waving = True
        if l_raised and not wrists_crossed and _check_wrist_waving(lambda o: o.wrist_l):
            is_waving = True

        # Persistence counters: crossed wrists are verified geometrically
        if wrists_crossed:
            self.crossed_persist_count += 1
        else:
            self.crossed_persist_count = max(0, self.crossed_persist_count - 1)

        # Raised hand requires at least 8 observations and confirmed micro-gesture sequence evidence
        has_min_obs_for_waving = len(self.history) >= 8
        has_signal_sequence = any(o.signal_for_help_stage > 0 for o in self.history)
        if hand_raised and not is_waving and has_min_obs_for_waving and has_signal_sequence:
            self.signal_persist_count += 1
        else:
            self.signal_persist_count = max(0, self.signal_persist_count - 1)

        zone = zone_manager.get_zone_for_point(detection.foot_anchor)

        # Trigger check: Macro Wrists Crossed (preferred in long-range CCTV)
        if self.use_macro and self.crossed_persist_count >= self.min_persistence:
            self.last_trigger_frame = frame_idx
            self.crossed_persist_count = 0
            return SafetyEvent(
                event_type=EventType.DISTRESS_GESTURE,
                confidence=0.92,
                track_id=self.track_id,
                timestamp=timestamp,
                zone=zone,
                train_state=train_state.value,
                evidence_frames=[o.frame_idx for o in self.history[-self.min_persistence:]],
                metadata={
                    "gesture_variant": self.detected_cross_variant,
                    "severity": "HIGH",
                    "persistence_frames": self.min_persistence,
                    "waving_rejected": False
                }
            )

        # Trigger check: Signal for Help (Hand raised & held steady with confirmed sequence)
        if self.signal_persist_count >= (self.min_persistence + 2) and has_signal_sequence:
            self.last_trigger_frame = frame_idx
            self.signal_persist_count = 0
            return SafetyEvent(
                event_type=EventType.DISTRESS_GESTURE,
                confidence=0.89,
                track_id=self.track_id,
                timestamp=timestamp,
                zone=zone,
                train_state=train_state.value,
                evidence_frames=[o.frame_idx for o in self.history[-self.min_persistence:]],
                metadata={
                    "gesture_variant": "SIGNAL_FOR_HELP",
                    "persistence_frames": self.min_persistence + 2,
                    "waving_rejected": False
                }
            )

        return None


class DistressGestureDetector:
    """Manages distress gesture state across all tracked persons."""

    def __init__(
        self,
        detection_window: int = 20,
        min_persistence: int = 2,
        cooldown_frames: int = 20,
        use_macro_fallback: bool = True
    ):
        self.window = detection_window
        self.min_persistence = min_persistence
        self.cooldown = cooldown_frames
        self.use_macro = use_macro_fallback
        self.track_machines: Dict[int, TrackGestureStateMachine] = {}

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_detections: List[Detection],
        zone_manager: ZoneManager,
        train_state: TrainState
    ) -> List[SafetyEvent]:
        """Scan active tracks for distress gestures."""
        events: List[SafetyEvent] = []
        active_ids = set()

        for det in tracked_detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            active_ids.add(tid)

            if tid not in self.track_machines:
                self.track_machines[tid] = TrackGestureStateMachine(
                    track_id=tid,
                    detection_window=self.window,
                    min_persistence=self.min_persistence,
                    cooldown_frames=self.cooldown,
                    use_macro_fallback=self.use_macro
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

        # Cleanup
        for dead_id in list(self.track_machines.keys()):
            if dead_id not in active_ids:
                del self.track_machines[dead_id]

        return events

