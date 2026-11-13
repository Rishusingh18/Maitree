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
