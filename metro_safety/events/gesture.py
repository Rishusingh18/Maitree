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

class TrackGestureStateMachine:
    """State machine tracking distress gestures per person track."""

    def __init__(
        self,
        track_id: int,
        detection_window: int = 20,
        min_persistence: int = 4,
        cooldown_frames: int = 30,
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
        # 0: nose, 5: l_shoulder, 6: r_shoulder, 9: l_wrist, 10: r_wrist
        nose = tuple(kpts[0])
        sh_l = tuple(kpts[5])
        sh_r = tuple(kpts[6])
        w_l = tuple(kpts[9])
        w_r = tuple(kpts[10])

        sh_y = (sh_l[1] + sh_r[1]) / 2.0
        sh_width = max(10.0, abs(sh_r[0] - sh_l[0]))

        # --- 1. Macro Gesture: Wrists crossed overhead ---
        # Both wrists above nose level, and close horizontally (crossing)
        wrists_crossed = False
        if w_l[2] > 0.35 and w_r[2] > 0.35 and nose[2] > 0.35:
            both_above_head = (w_l[1] < nose[1]) and (w_r[1] < nose[1])
            wrists_close = abs(w_l[0] - w_r[0]) < (sh_width * 0.75)
            # Wrists crossed horizontally: L wrist on right side or close to center
            if both_above_head and wrists_close:
                wrists_crossed = True

        # --- 2. Micro Gesture: Signal for Help (Raised Hand sequence) ---
        # Hand raised above shoulder height, steady position (not waving left-right)
        hand_raised = False
        if (w_r[2] > 0.4 and w_r[1] < sh_y - 15) or (w_l[2] > 0.4 and w_l[1] < sh_y - 15):
            hand_raised = True

        obs = GestureObservation(
            frame_idx=frame_idx,
            timestamp=timestamp,
            wrist_l=w_l,
            wrist_r=w_r,
            nose=nose,
            sh_l=sh_l,
            sh_r=sh_r,
            signal_for_help_stage=2 if hand_raised else 0,
            wrists_crossed=wrists_crossed
        )
        self.history.append(obs)
        if len(self.history) > self.window:
            self.history.pop(0)

        # Anti-False-Positive: Waving Rejection
        # Normal waving has high horizontal velocity oscillations of wrist
        wrist_xs = [o.wrist_r[0] for o in self.history if o.wrist_r[2] > 0.3]
        is_waving = False
        if len(wrist_xs) >= 8:
            x_diffs = np.diff(wrist_xs)
            sign_changes = np.sum(np.diff(np.sign(x_diffs)) != 0)
            if sign_changes >= 4 and np.max(wrist_xs) - np.min(wrist_xs) > 25.0:
                is_waving = True  # Reject waving / hailing

        # Persistence counters
        if wrists_crossed and not is_waving:
            self.crossed_persist_count += 1
        else:
            self.crossed_persist_count = max(0, self.crossed_persist_count - 1)

        if hand_raised and not is_waving:
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
                    "gesture_variant": "WRISTS_CROSSED_OVERHEAD",
                    "persistence_frames": self.min_persistence,
                    "waving_rejected": False
                }
            )

        # Trigger check: Signal for Help (Hand raised & held steady)
        if self.signal_persist_count >= (self.min_persistence + 2):
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
        min_persistence: int = 4,
        cooldown_frames: int = 30,
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
