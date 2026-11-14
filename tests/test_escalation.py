"""Unit tests for Escalation Ladder, Hardware Adapters, and Interlocks."""
import pytest
from metro_safety.escalation.engine import EscalationEngine
from metro_safety.escalation.adapters import SimulatedEmergencyTripSwitch
from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel

def test_escalation_l1():
    eng = EscalationEngine()
    ev = SafetyEvent(EventType.LINE_CROSSING, 0.9, 1, 1.0, "yellow_band", "ABSENT")
    dec = eng.evaluate(ev)
    assert dec.severity == SeverityLevel.L1
    assert dec.buzzer_active is True
    assert dec.control_room_alerted is False
    assert dec.ets_triggered is False

def test_escalation_l2():
    eng = EscalationEngine()
    ev = SafetyEvent(EventType.TRACK_INTRUSION, 0.92, 2, 1.0, "track_bed", "ABSENT")
    dec = eng.evaluate(ev)
    assert dec.severity == SeverityLevel.L2
    assert dec.buzzer_active is True
    assert dec.control_room_alerted is True
    assert dec.ets_triggered is False

def test_escalation_l3_approaching_train():
    ets = SimulatedEmergencyTripSwitch(human_confirmation_required=False)
    eng = EscalationEngine(trip_switch=ets)
    ev = SafetyEvent(EventType.TRACK_INTRUSION, 0.95, 3, 1.0, "track_bed", "APPROACHING")
    dec = eng.evaluate(ev)
    assert dec.severity == SeverityLevel.L3
    assert dec.buzzer_active is True
    assert dec.control_room_alerted is True
    assert dec.ets_triggered is True

def test_silent_distress_gesture_ladder():
    eng = EscalationEngine()
    ev = SafetyEvent(EventType.DISTRESS_GESTURE, 0.94, 4, 1.0, "platform", "ABSENT")
    dec = eng.evaluate(ev)
    assert dec.severity == SeverityLevel.DISTRESS
    assert dec.buzzer_active is False  # Strict requirement: no audible buzzer
    assert dec.control_room_alerted is True

def test_ets_human_confirmation_interlock():
    ets = SimulatedEmergencyTripSwitch(human_confirmation_required=True)
    eng = EscalationEngine(trip_switch=ets)
    ev = SafetyEvent(EventType.ACCIDENTAL_FALL, 0.96, 5, 1.0, "track_bed", "ABSENT")
    dec = eng.evaluate(ev)
    assert dec.ets_triggered is False
    assert ets.is_tripped() is False
    assert ets.pending_confirmation is not None

    # Operator confirms
    res = ets.confirm_by_operator()
    assert res is True
    assert ets.is_tripped() is True


def test_escalation_explicit_train_state_and_snapshot():
    """Verify train_state passed explicitly overrides event.train_state, and snapshot is accepted."""
    import numpy as np
    from metro_safety.perception.train_state import TrainState

    eng = EscalationEngine()
    ev = SafetyEvent(EventType.TRACK_INTRUSION, 0.92, 10, 1.0, "track_bed", "ABSENT")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # When train_state="APPROACHING" is passed explicitly, should escalate to L3
    dec = eng.evaluate(ev, train_state=TrainState.APPROACHING, snapshot=dummy_frame)
    assert dec.severity == SeverityLevel.L3
    assert dec.snapshot is not None
    assert dec.snapshot.shape == (480, 640, 3)

    # String train_state also supported
    dec2 = eng.evaluate(ev, train_state="ABSENT", snapshot=dummy_frame)
    assert dec2.severity == SeverityLevel.L2


def test_escalation_invalid_argument_type_checks():
    """Runtime check: Passing an ndarray as train_state or current_time must raise a clear TypeError."""
    import numpy as np

    eng = EscalationEngine()
    ev = SafetyEvent(EventType.LINE_CROSSING, 0.90, 11, 1.0, "yellow_band", "ABSENT")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Invalid type for 'train_state'"):
        eng.evaluate(ev, dummy_frame)  # Positional frame passed as train_state!

    with pytest.raises(TypeError, match="Invalid type for 'current_time'"):
        eng.evaluate(ev, "ABSENT", dummy_frame)  # Positional frame passed as current_time!


