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
from typing import List, Dict, Optional, Tuple, Union
import time
import numpy as np

from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel
from metro_safety.perception.train_state import TrainState
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
    snapshot: Optional[np.ndarray] = None

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
        self.decision_log: List[EscalationDecision] = []

    def evaluate(
        self,
        event: SafetyEvent,
        train_state: Optional[Union[str, TrainState]] = None,
        current_time: Optional[float] = None,
        snapshot: Optional[np.ndarray] = None
    ) -> EscalationDecision:
        """Process event, determine severity ladder action, trigger adapters, and measure latency."""
        # 1. Strict runtime type checks to catch argument misplacements early
        if not isinstance(event, SafetyEvent):
            raise TypeError(f"Expected SafetyEvent for 'event', got {type(event).__name__}")

        if isinstance(train_state, np.ndarray):
            raise TypeError(
                f"Invalid type for 'train_state': received numpy.ndarray. "
                f"Did you pass a video frame as positional argument? "
                f"Pass the frame as snapshot=<frame> and train_state=<str/TrainState> explicitly."
            )

        if isinstance(current_time, np.ndarray):
            raise TypeError(
                f"Invalid type for 'current_time': received numpy.ndarray. "
                f"Did you pass a video frame as positional argument? "
                f"Pass the frame as snapshot=<frame> explicitly."
            )

        if current_time is not None and not isinstance(current_time, (int, float)):
            raise TypeError(f"Invalid type for 'current_time': expected float or None, got {type(current_time).__name__}")

        if snapshot is not None and not isinstance(snapshot, np.ndarray):
            raise TypeError(f"Invalid type for 'snapshot': expected numpy.ndarray or None, got {type(snapshot).__name__}")

        # 2. Resolve train_state
        effective_train_state: str = "ABSENT"
        if train_state is not None:
            if isinstance(train_state, str):
                effective_train_state = train_state.upper()
            elif hasattr(train_state, "value"):
                effective_train_state = str(train_state.value).upper()
            else:
                raise TypeError(
                    f"Invalid type for 'train_state': expected str, TrainState, or None, got {type(train_state).__name__}"
                )
        elif event.train_state is not None:
            if hasattr(event.train_state, "value"):
                effective_train_state = str(event.train_state.value).upper()
            else:
                effective_train_state = str(event.train_state).upper()

        t_now = float(current_time) if current_time is not None else time.time()
        start_time = time.perf_counter()

        ev_type = event.event_type
        train_state_str = effective_train_state

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
            self.notifier.send_alert(event, severity, action_msg)

        # 2. Confirmed Fall or Jump -> Immediate Level 3 Emergency
        elif ev_type in [EventType.ACCIDENTAL_FALL, EventType.DELIBERATE_JUMP]:
            severity = SeverityLevel.L3
            action_msg = f"CRITICAL: {ev_type.value} on track! Actuating platform buzzer, alarm, and emergency stop recommendation."
            buzzer_active = True
            cr_alert = True
            self.buzzer.activate(self.platform_id, pattern="CONTINUOUS_HIGH")
            self.notifier.send_alert(event, severity, action_msg)
            ets_active = self.trip_switch.trigger_stop(event, self.station_id, self.platform_id)

        # 3. Track Intrusion
        elif ev_type == EventType.TRACK_INTRUSION:
            # Track duration for auto-escalation
            if event.track_id not in self.active_intrusions:
                self.active_intrusions[event.track_id] = t_now
            elapsed = t_now - self.active_intrusions[event.track_id]

            # High severity if train approaching OR duration exceeds auto-escalate timer
            if train_state_str == "APPROACHING" or elapsed >= self.auto_escalate_sec:
                severity = SeverityLevel.L3
                action_msg = f"CRITICAL: Track intrusion with train approaching ({elapsed:.1f}s)! Triggering ETS."
                buzzer_active = True
                cr_alert = True
                self.buzzer.activate(self.platform_id, pattern="CONTINUOUS_HIGH")
                self.notifier.send_alert(event, severity, action_msg)
                ets_active = self.trip_switch.trigger_stop(event, self.station_id, self.platform_id)
            else:
                # Level 2: Person on track, train absent or far
                severity = SeverityLevel.L2
                action_msg = f"WARNING: Person on track bed (train {train_state_str}). Sounding buzzer & alerting control room."
                buzzer_active = True
                cr_alert = True
                self.buzzer.activate(self.platform_id, pattern="PULSED")
                self.notifier.send_alert(event, severity, action_msg)

        # 4. Yellow Line Crossing
        elif ev_type == EventType.LINE_CROSSING:
            if train_state_str == "APPROACHING":
                severity = SeverityLevel.L2
                action_msg = "Yellow line crossed while train is approaching! Warning buzzer + control room notification."
                buzzer_active = True
                cr_alert = True
                self.buzzer.activate(self.platform_id, pattern="PULSED_FAST")
                self.notifier.send_alert(event, severity, action_msg)
            else:
                severity = SeverityLevel.L1
                action_msg = "Yellow line safety boundary crossed. Local platform warning buzzer active."
                buzzer_active = True
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
        self.decision_log.append(decision)
        return decision

    def clear_track(self, track_id: int):
        """Called when a person clears back to the platform."""
        if track_id in self.active_intrusions:
            del self.active_intrusions[track_id]
        if not self.active_intrusions:
            self.buzzer.deactivate(self.platform_id)

