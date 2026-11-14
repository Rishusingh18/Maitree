"""Loitering / Dwell Anomaly Detector.

Detects passengers dwelling or loitering within a localized spatial radius
in platform or safety zones for prolonged durations without an arriving train.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager


@dataclass
class DwellTrackState:
    """Tracks position history and dwell timers for a single person track."""
    track_id: int
    anchor_position: Tuple[float, float]
    anchor_start_ts: float
    last_seen_ts: float
    dwell_duration: float = 0.0
    zone: str = "platform"
    positions: List[Tuple[float, float, float]] = field(default_factory=list) # (x, y, ts)
    l1_alerted: bool = False
    l2_alerted: bool = False
    l2_trusted_alerted: bool = False


class LoiteringDetector:
    """Evaluates tracked persons for extended dwell / loitering in monitored zones."""

    def __init__(
        self,
        radius_px: float = 70.0,
        dwell_t1_sec: float = 30.0,
        dwell_t2_sec: float = 60.0,
        train_suppression: bool = True
    ):
        self.radius_px = radius_px
        self.dwell_t1_sec = dwell_t1_sec
        self.dwell_t2_sec = dwell_t2_sec
        self.train_suppression = train_suppression

        self.track_states: Dict[int, DwellTrackState] = {}

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_dets: List[Detection],
        zone_manager: ZoneManager,
        train_state: str = "ABSENT",
        camera_motion: bool = False,
        zones_unverified: bool = False
    ) -> List[SafetyEvent]:
        """Evaluate dwell and loitering times for active tracks."""
        events: List[SafetyEvent] = []
        current_tids = set()

        # Check train suppression: if train is approaching or present/docked, loitering is suppressed
        effective_ts = str(train_state).upper()
        if self.train_suppression and effective_ts in ["APPROACHING", "PRESENT", "DOCKED", "ARRIVING", "STOPPED_DOORS_OPEN"]:
            # Pause/reset dwell timer during train suppression so post-departure does not immediately fire
            for tid, st in self.track_states.items():
                st.anchor_start_ts = timestamp
                st.dwell_duration = 0.0
                st.l1_alerted = False
                st.l2_alerted = False
                st.l2_trusted_alerted = False
            return events

        for det in tracked_dets:
            tid = det.track_id or 0
            if tid == 0:
                continue
            current_tids.add(tid)

            foot = det.foot_anchor
            zone = zone_manager.get_zone_for_point(foot)

            # Only monitor platform and buffer/danger zones; ignore seating or outside
            if zone in ["outside", "ignore", "UNKNOWN"]:
                if tid in self.track_states:
                    del self.track_states[tid]
                continue

            if tid not in self.track_states:
                self.track_states[tid] = DwellTrackState(
                    track_id=tid,
                    anchor_position=foot,
                    anchor_start_ts=timestamp,
                    last_seen_ts=timestamp,
                    zone=zone,
                    positions=[(foot[0], foot[1], timestamp)]
                )
                continue

            st = self.track_states[tid]
            st.last_seen_ts = timestamp
            st.zone = zone
            st.positions.append((foot[0], foot[1], timestamp))
            if len(st.positions) > 50:
                st.positions.pop(0)

            # Check displacement from anchor position
            dist_from_anchor = math.hypot(
                foot[0] - st.anchor_position[0],
                foot[1] - st.anchor_position[1]
            )

            if dist_from_anchor <= self.radius_px:
                # Within dwell radius: advance dwell timer
                st.dwell_duration = round(timestamp - st.anchor_start_ts, 1)

                is_low_trust = bool(camera_motion or zones_unverified)
                if st.dwell_duration >= self.dwell_t2_sec:
                    if not st.l2_alerted or (not getattr(st, "l2_trusted_alerted", False) and not is_low_trust):
                        # L2 Loitering Security Alert
                        ev = SafetyEvent(
                            event_type=EventType.LOITERING,
                            confidence=min(0.95, 0.82 + 0.05 * min(2.0, st.dwell_duration / 120.0)),
                            track_id=tid,
                            timestamp=timestamp,
                            zone=zone,
                            train_state=effective_ts,
                            evidence_frames=[frame_idx],
                            metadata={
                                "duration_sec": st.dwell_duration,
                                "dwell_radius_px": self.radius_px,
                                "displacement_px": round(dist_from_anchor, 1),
                                "severity_cap": "L2",
                                "severity_code": "L2",
                                "disclaimer": "operator welfare review",
                                "action": f"SECURITY ALERT: Loitering in {zone} for {st.dwell_duration}s without train. Operator review."
                            }
                        )
                        ev.is_anomaly = True
                        if not is_low_trust:
                            st.l2_trusted_alerted = True
                        st.l2_alerted = True
                        events.append(ev)

                elif st.dwell_duration >= self.dwell_t1_sec:
                    if not st.l1_alerted:
                        # L1 Loitering Notice
                        ev = SafetyEvent(
                            event_type=EventType.LOITERING,
                            confidence=0.80,
                            track_id=tid,
                            timestamp=timestamp,
                            zone=zone,
                            train_state=effective_ts,
                            evidence_frames=[frame_idx],
                            metadata={
                                "duration_sec": st.dwell_duration,
                                "dwell_radius_px": self.radius_px,
                                "displacement_px": round(dist_from_anchor, 1),
                                "severity_cap": "L1",
                                "severity_code": "L1",
                                "disclaimer": "operator welfare review",
                                "action": f"SECURITY NOTICE: Extended dwell in {zone} ({st.dwell_duration}s)."
                            }
                        )
                        ev.is_anomaly = True
                        st.l1_alerted = True
                        events.append(ev)

            else:
                # Person walked away: update anchor to new position and reset dwell timer
                st.anchor_position = foot
                st.anchor_start_ts = timestamp
                st.dwell_duration = 0.0
                st.l1_alerted = False
                st.l2_alerted = False
                st.l2_trusted_alerted = False

        # Prune inactive tracks
        dead_tids = [tid for tid in self.track_states if tid not in current_tids]
        for dead_id in dead_tids:
            del self.track_states[dead_id]

        return events

