"""Verify via real HTTP requests that Mute, Test Beep, Test Siren, ENGAGE ETS,
Reset, and Acknowledge each visibly change state in /api/telemetry or /api/events,
and assert all events are tagged with a unique session_id.
"""
import time
import socket
import threading
from unittest.mock import MagicMock
import pytest
import requests
from werkzeug.serving import make_server

from metro_safety.demo.hub import LiveHub
from metro_safety.demo.dashboard import create_dashboard_app
from metro_safety.escalation.buzzer import HardwareBuzzer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    hub = LiveHub(cctv_id="CCTV_TEST", station_id="STN_TEST", platform_id="PL_TEST")
    hub.set_hardware_components(buzzer=buzzer)

    app = create_dashboard_app(hub)
    port = get_free_port()
    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    # Wait for server to accept connections
    for _ in range(50):
        try:
            requests.get(f"{base_url}/api/telemetry", timeout=0.5)
            break
        except Exception:
            time.sleep(0.05)

    yield base_url, hub, buzzer

    server.shutdown()
    buzzer.stop()


def test_control_actions_and_session_id_verification(live_server):
    base_url, hub, buzzer = live_server

    # 1. Telemetry and session_id baseline
    r = requests.get(f"{base_url}/api/telemetry", timeout=2.0)
    assert r.status_code == 200
    tel = r.json()
    session_id = tel.get("session_id")
    assert session_id is not None and session_id.startswith("SES-")
    assert hub.session_id == session_id

    # 2. Test MUTE & UNMUTE
    r_mute = requests.post(f"{base_url}/api/buzzer/toggle_mute", timeout=2.0)
    assert r_mute.status_code == 200
    assert r_mute.json()["muted"] is True

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["buzzer"]["muted"] is True

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    ev_types = [e["event_type"] for e in r_ev["events"]]
    assert "BUZZER_MUTED" in ev_types

    # Unmute
    r_unmute = requests.post(f"{base_url}/api/buzzer/toggle_mute", timeout=2.0)
    assert r_unmute.status_code == 200
    assert r_unmute.json()["muted"] is False
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["buzzer"]["muted"] is False

    # 3. Test Beep (L2)
    r_beep = requests.post(f"{base_url}/api/buzzer/test", json={"pattern": "L2"}, timeout=2.0)
    assert r_beep.status_code == 200

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["buzzer"]["last_test_pattern"] == "L2"

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    assert any(e["event_type"] == "BUZZER_TEST_L2" for e in r_ev["events"])

    # 4. Test Siren (L3)
    r_siren = requests.post(f"{base_url}/api/buzzer/test", json={"pattern": "L3"}, timeout=2.0)
    assert r_siren.status_code == 200

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["buzzer"]["last_test_pattern"] == "L3"

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    assert any(e["event_type"] == "BUZZER_TEST_L3" for e in r_ev["events"])

    # 5. ENGAGE ETS (Simulated traction power cut)
    r_ets = requests.post(f"{base_url}/api/ets/trigger", timeout=2.0)
    assert r_ets.status_code == 200
    assert r_ets.json()["ets_status"] == "ENGAGED"

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["ets_status"] == "ENGAGED"

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    assert any(e["event_type"] == "ETS_MANUAL_TRIP" for e in r_ev["events"])

    # 6. Reset ETS
    r_rst = requests.post(f"{base_url}/api/ets/reset", timeout=2.0)
    assert r_rst.status_code == 200
    assert r_rst.json()["ets_status"] == "DISENGAGED"

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["ets_status"] == "DISENGAGED"

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    assert any(e["event_type"] == "ETS_RESET" for e in r_ev["events"])

    # 7. Acknowledge Alarm
    buzzer.trigger("L3")
    hub.set_overlooker_alert({"title": "Test Alert", "body": "Testing acknowledge"})
    r_ack = requests.post(f"{base_url}/api/acknowledge", timeout=2.0)
    assert r_ack.status_code == 200

    # Visible in /api/telemetry
    r_tel = requests.get(f"{base_url}/api/telemetry", timeout=2.0).json()
    assert r_tel["buzzer"]["acknowledged"] is True
    assert r_tel["active_alert"] is None

    # Visible in /api/events
    r_ev = requests.get(f"{base_url}/api/events", timeout=2.0).json()
    assert any(e["event_type"] == "ALARM_ACKNOWLEDGED" for e in r_ev["events"])

    # 8. Tag all events with session_id
    for ev in r_ev["events"]:
        assert "session_id" in ev
        assert ev["session_id"] == session_id

