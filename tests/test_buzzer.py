"""Unit tests for HardwareBuzzer."""
import time
from unittest.mock import MagicMock
from metro_safety.escalation.buzzer import HardwareBuzzer

def test_buzzer_trigger_patterns():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, clear_timeout_sec=1.0, beep_fn=mock_beep)

    try:
        status = buzzer.get_status()
        assert not status["active"]
        assert status["pattern"] == "OFF"

        # Trigger L1
        buzzer.trigger("L1")
        time.sleep(0.05)
        status = buzzer.get_status()
        assert status["active"]
        assert status["pattern"] == "L1"

        # Trigger higher severity L3
        buzzer.trigger("L3")
        time.sleep(0.05)
        status = buzzer.get_status()
        assert status["active"]
        assert status["pattern"] == "L3"

        # Distress MUST NOT trigger buzzer (silent)
        buzzer.acknowledge()
        status = buzzer.get_status()
        assert not status["active"]

        buzzer.trigger("DISTRESS")
        status = buzzer.get_status()
        assert not status["active"]
        assert status["pattern"] == "OFF"

    finally:
        buzzer.stop()


def test_buzzer_acknowledge():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    try:
        buzzer.trigger("L2")
        assert buzzer.get_status()["active"]

        buzzer.acknowledge()
        assert not buzzer.get_status()["active"]
        assert buzzer.get_status()["pattern"] == "OFF"
    finally:
        buzzer.stop()


def test_buzzer_mute_toggle():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    try:
        assert not buzzer.get_status()["muted"]
        is_muted = buzzer.toggle_mute()
        assert is_muted
        assert buzzer.get_status()["muted"]

        # When muted, trigger does not invoke beep
        mock_beep.reset_mock()
        buzzer.trigger("L1")
        time.sleep(0.15)
        assert mock_beep.call_count == 0

        # Unmute
        buzzer.toggle_mute()
        assert not buzzer.get_status()["muted"]
    finally:
        buzzer.stop()


def test_buzzer_clear_timeout():
    mock_beep = MagicMock()
    # 0.2s clear timeout for fast test
    buzzer = HardwareBuzzer(audio_enabled=True, clear_timeout_sec=0.2, beep_fn=mock_beep)
    try:
        buzzer.trigger("L1")
        assert buzzer.get_status()["active"]

        # Keep intruding
        buzzer.update_clear_status(is_intruding=True)
        time.sleep(0.1)
        assert buzzer.get_status()["active"]

        # Cleared for > 0.2s
        time.sleep(0.25)
        buzzer.update_clear_status(is_intruding=False)
        assert not buzzer.get_status()["active"]
        assert buzzer.get_status()["pattern"] == "OFF"
    finally:
        buzzer.stop()
