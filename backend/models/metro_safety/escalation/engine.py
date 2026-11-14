"""Escalation Engine executing the Platform Safety Severity Ladder.
Ladder Rules:
- L1: Line crossing, train absent -> Local audible buzzer on platform
- L2: Person on track, train absent/far -> Buzzer + Control Room alert with snapshot/clip
- L3: Person on track with train approaching, OR confirmed fall/jump -> Alarm + ETS Trip recommend/trigger
- DISTRESS: Silent distress gesture -> Silent alert to Control Room & Security ONLY (no buzzer)
With auto-escalation timer and audit trail.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import time
import numpy as np

from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel
from metro_safety.escalation.adapters import (
    BasePlatformBuzzer,
    BaseControlRoomNotifier,
    BaseEmergencyTripSwitch,
    SimulatedPlatformBuzzer,
    SimulatedControlRoomNotifier,
    SimulatedEmergencyTripSwitch
)

@dataclass
class EscalationDecision:
    severity: SeverityLevel
    action_taken: str
    buzzer_active: bool
    control_room_alerted: bool
    ets_triggered: bool
    latency_ms: float
    timestamp: float
    event: SafetyEvent
    snapshot: Optional[Any] = None

class EscalationEngine:
    """Evaluates safety events against severity rules and hardware actuators."""

    def __init__(
        self,
        station_id: str = "DMRC_01",
        platform_id: str = "PL_01",
        buzzer: Optional[BasePlatformBuzzer] = None,
        notifier: Optional[BaseControlRoomNotifier] = None,
        trip_switch: Optional[BaseEmergencyTripSwitch] = None,
        auto_escalate_timer_sec: float = 60.0
    ):
        self.station_id = station_id
        self.platform_id = platform_id
        self.buzzer = buzzer or SimulatedPlatformBuzzer()
        self.notifier = notifier or SimulatedControlRoomNotifier()
        self.trip_switch = trip_switch or SimulatedEmergencyTripSwitch()
        self.auto_escalate_sec = auto_escalate_timer_sec

        self.active_intrusions: Dict[int, float] = {}  # track_id -> first detection timestamp
        self.active_audible_hazards: Dict[int, EventType] = {}  # track_id -> buzzer hazard event type
        self.decision_log: List[EscalationDecision] = []

    def _safe_send_alert(self, event: SafetyEvent, severity: SeverityLevel, action_msg: str):
        """Isolate notifier/audit failures so they never block ETS or buzzer actuation."""
        try:
            self.notifier.send_alert(event, severity, action_msg)
        except Exception as e:
            import logging
            logging.getLogger("EscalationEngine").error(f"Notifier send_alert failed: {e}")

    def evaluate(
        self,
        event: SafetyEvent,
        train_state: Optional[Union[str, Any]] = None,
        current_time: Optional[float] = None,
        snapshot: Optional[Any] = None,
        **kwargs
    ) -> EscalationDecision:
        """Process event, determine severity ladder action, trigger adapters, and measure latency."""
        if isinstance(train_state, np.ndarray):
            raise TypeError("Invalid type for 'train_state': received numpy.ndarray. Pass the frame as snapshot=<frame> explicitly.")
        if isinstance(current_time, np.ndarray):
            raise TypeError("Invalid type for 'current_time': received numpy.ndarray. Pass the frame as snapshot=<frame> explicitly.")

        # Handle positional swap where 2nd arg was current_time (e.g., evaluate(ev, 1700000.0))
        if isinstance(train_state, (int, float)) and current_time is None:
            current_time = float(train_state)
            train_state = None

        t_now = float(current_time) if current_time is not None else time.time()
        start_time = time.perf_counter()

        ev_type = event.event_type

        # Resolve train state string
        if train_state is not None:
            effective_train_state = train_state.value.upper() if hasattr(train_state, "value") else str(train_state).upper()
        elif event.train_state is not None:
            effective_train_state = event.train_state.value.upper() if hasattr(event.train_state, "value") else str(event.train_state).upper()
        else:
            effective_train_state = "ABSENT"

        train_state = effective_train_state

        severity = SeverityLevel.NORMAL
        action_msg = ""
        buzzer_active = False
        cr_alert = False
        ets_active = False

        # --- Severity Ladder Mapping ---

        # 1. Silent Distress Gesture
        if ev_type == EventType.DISTRESS_GESTURE:
            severity = SeverityLevel.DISTRESS
            action_msg = "Silent distress alarm dispatched to station security and control room."
            cr_alert = True
            buzzer_active = False  # STRICT REQUIREMENT: NO audible buzzer to protect victim
            self._safe_send_alert(event, severity, action_msg)

        # 2. Confirmed Fall or Jump -> Immediate Level 3 Emergency
        elif ev_type in [EventType.ACCIDENTAL_FALL, EventType.DELIBERATE_JUMP]:
            is_camera_moving = getattr(event, "camera_motion", False)
            if is_camera_moving:
                severity = SeverityLevel.L2
                action_msg = f"CAMERA MOVING: needs operator review - {getattr(event, 'motion_reason', 'Optical flow vibration')}"
                buzzer_active = True
                cr_alert = True
                ets_active = False
                if event.track_id in self.active_intrusions:
                    del self.active_intrusions[event.track_id]
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="PULSED")
                self._safe_send_alert(event, severity, action_msg)
            else:
                severity = SeverityLevel.L3
                action_msg = f"CRITICAL: {ev_type.value} on track! Actuating platform buzzer, alarm, and emergency stop recommendation."
                buzzer_active = True
                cr_alert = True
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="CONTINUOUS_HIGH")
                self._safe_send_alert(event, severity, action_msg)
                ets_active = self.trip_switch.trigger_stop(event, self.station_id, self.platform_id)

        # 3. Track Intrusion
        elif ev_type == EventType.TRACK_INTRUSION:
            # Track duration for auto-escalation
            if event.track_id not in self.active_intrusions:
                self.active_intrusions[event.track_id] = t_now
            elapsed = t_now - self.active_intrusions[event.track_id]

            # High severity if train approaching OR duration exceeds auto-escalate timer
            if train_state == "APPROACHING" or elapsed >= self.auto_escalate_sec:
                severity = SeverityLevel.L3
                action_msg = f"CRITICAL: Track intrusion with train approaching ({elapsed:.1f}s)! Triggering ETS."
                buzzer_active = True
                cr_alert = True
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="CONTINUOUS_HIGH")
                self._safe_send_alert(event, severity, action_msg)
                ets_active = self.trip_switch.trigger_stop(event, self.station_id, self.platform_id)
            else:
                # Level 2: Person on track, train absent or far
                severity = SeverityLevel.L2
                action_msg = f"WARNING: Person on track bed (train {train_state}). Sounding buzzer & alerting control room."
                buzzer_active = True
                cr_alert = True
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="PULSED")
                self._safe_send_alert(event, severity, action_msg)

        # 4. Yellow Line Crossing
        elif ev_type == EventType.LINE_CROSSING:
            if train_state == "APPROACHING":
                severity = SeverityLevel.L2
                action_msg = "Yellow line crossed while train is approaching! Warning buzzer + control room notification."
                buzzer_active = True
                cr_alert = True
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="PULSED_FAST")
                self._safe_send_alert(event, severity, action_msg)
            else:
                severity = SeverityLevel.L1
                action_msg = "Yellow line safety boundary crossed. Local platform warning buzzer active."
                buzzer_active = True
                self.active_audible_hazards[event.track_id] = ev_type
                self.buzzer.activate(self.platform_id, pattern="PULSED")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision = EscalationDecision(
            severity=severity,
            action_taken=action_msg,
            buzzer_active=buzzer_active,
            control_room_alerted=cr_alert,
            ets_triggered=ets_active,
            latency_ms=round(latency_ms, 2),
            timestamp=t_now,
            event=event,
            snapshot=snapshot
        )
        self.decision_log.append(decision)
        return decision

    def clear_track(self, track_id: int):
        """Called when a person clears back to the platform."""
        if track_id in self.active_intrusions:
            del self.active_intrusions[track_id]
        if track_id in self.active_audible_hazards:
            del self.active_audible_hazards[track_id]
        if not self.active_audible_hazards and not self.active_intrusions:
            self.buzzer.deactivate(self.platform_id)

    def check_auto_escalations(self, current_time: float, train_state: str = "ABSENT") -> List[EscalationDecision]:
        """Independently re-evaluates active intrusions at or after auto_escalate_sec."""
        decisions = []
        for track_id, start_ts in list(self.active_intrusions.items()):
            elapsed = current_time - start_ts
            if elapsed >= self.auto_escalate_sec:
                synthetic_ev = SafetyEvent(
                    event_type=EventType.TRACK_INTRUSION,
                    confidence=0.95,
                    track_id=track_id,
                    timestamp=current_time,
                    zone="track_bed",
                    train_state=train_state
                )
                dec = self.evaluate(synthetic_ev, current_time=current_time)
                decisions.append(dec)
        return decisions

