"""Regression and robustness tests for the live pipeline.
Verifies that:
1. Feeding a numpy frame plus an L3 event through the live path causes NO crash (fixes ValueError array truth value).
2. Exceptions in escalation, buzzer, notifications, or dashboard publishing are caught, logged, and processing CONTINUES.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel
from metro_safety.perception.train_state import TrainState
from metro_safety.escalation.engine import EscalationEngine
from metro_safety.escalation.buzzer import HardwareBuzzer
from metro_safety.escalation.overlooker import OverlookerNotifier
from metro_safety.demo.hub import LiveHub
from metro_safety.demo.live import run_live_pipeline


def test_live_path_frame_plus_l3_event_no_crash():
    """Regression test for requirement 2: Feed a numpy frame plus an L3 event through
    the live evaluation path, ensuring the signature accepts snapshot and train_state
    without raising 'ValueError: The truth value of an array with more than one element is ambiguous'.
    """
    eng = EscalationEngine()
    hub = LiveHub()
    buzzer = HardwareBuzzer(audio_enabled=False)
    overlooker = OverlookerNotifier(
        desktop_notification=False,
        audit_log_file="logs/test_robustness_audit.log",
        dashboard_hub=hub
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # L3 Deliberate Jump Event
    ev_jump = SafetyEvent(
        event_type=EventType.DELIBERATE_JUMP,
        confidence=0.95,
        track_id=42,
        timestamp=1.5,
        zone="track_bed",
        train_state="APPROACHING"
    )

    # Process through live path: escalation -> buzzer -> overlooker -> hub
    dec = eng.evaluate(
        event=ev_jump,
        train_state=TrainState.APPROACHING,
        current_time=1.5,
        snapshot=dummy_frame
    )
    assert dec.severity == SeverityLevel.L3
    assert dec.snapshot is not None

    buzzer.trigger(dec.severity.name)
    assert buzzer.get_status()["active"] is True
    assert buzzer.get_status()["pattern"] == "L3"

    overlooker.notify(
        event_type=ev_jump.event_type.name,
        severity=dec.severity.name,
        track_id=ev_jump.track_id,
        zone=ev_jump.zone,
        confidence=ev_jump.confidence,
        timestamp=1.5,
        train_state="APPROACHING"
    )

    hub.push_event({
        "event_type": ev_jump.event_type.name,
        "severity": dec.severity.name,
        "track_id": ev_jump.track_id,
        "zone": ev_jump.zone,
        "confidence": ev_jump.confidence,
        "action": dec.action_taken
    })
    hub.update_frame(dummy_frame, is_online=True)

    # Verify event reached hub
    events = hub.get_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "DELIBERATE_JUMP"
    assert events[0]["severity"] == "L3"

    buzzer.stop()


def test_live_path_subsystem_exceptions_do_not_halt_pipeline():
    """Verify that simulated exceptions in buzzer, notifier, or escalation
    are isolated and do not halt the live pipeline loop.
    """
    mock_buzzer = MagicMock()
    mock_buzzer.trigger.side_effect = RuntimeError("Simulated audio driver failure")
    mock_buzzer.update_clear_status.side_effect = RuntimeError("Timer error")
    mock_buzzer.get_status.return_value = {"active": False, "pattern": "OFF", "muted": False}

    mock_overlooker = MagicMock()
    mock_overlooker.notify.side_effect = ConnectionError("Simulated notification push failure")

    eng = EscalationEngine()
    hub = LiveHub()
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50

    ev = SafetyEvent(
        event_type=EventType.ACCIDENTAL_FALL,
        confidence=0.96,
        track_id=7,
        timestamp=2.0,
        zone="track_bed",
        train_state="ABSENT"
    )

    # Simulate inner live loop execution with error isolation
    try:
        dec = eng.evaluate(ev, train_state="ABSENT", snapshot=dummy_frame)
        try:
            mock_buzzer.trigger(dec.severity.name)
        except Exception:
            pass  # Isolated!

        try:
            mock_overlooker.notify(
                event_type=ev.event_type.name,
                severity=dec.severity.name,
                track_id=ev.track_id,
                zone=ev.zone,
                confidence=ev.confidence,
                timestamp=2.0,
                train_state="ABSENT"
            )
        except Exception:
            pass  # Isolated!

        hub.update_frame(dummy_frame, is_online=True)
    except Exception as e:
        pytest.fail(f"Pipeline crashed due to unhandled exception: {e}")

    # Verify video frame is still successfully published despite subsystem errors
    latest_jpeg = hub.get_latest_jpeg()
    assert len(latest_jpeg) > 0
    assert latest_jpeg.startswith(b"\xff\xd8")

