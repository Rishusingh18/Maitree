"""Unit tests for Dashboard Flask endpoints and LiveHub."""
import pytest
from unittest.mock import MagicMock
from metro_safety.demo.hub import LiveHub
from metro_safety.demo.dashboard import create_dashboard_app
from metro_safety.escalation.buzzer import HardwareBuzzer

@pytest.fixture
def client_and_hub():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    hub = LiveHub(cctv_id="CCTV_TEST", station_id="STN_TEST", platform_id="PL_TEST")
    hub.set_hardware_components(buzzer=buzzer)

    app = create_dashboard_app(hub)
    app.config["TESTING"] = True
    client = app.test_client()

    yield client, hub, buzzer
    buzzer.stop()


def test_dashboard_index(client_and_hub):
    client, hub, _ = client_and_hub
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "/video_feed" in html
    assert "DMRC CENTRAL CONTROL ROOM" in html


def test_dashboard_telemetry_api(client_and_hub):
    client, hub, _ = client_and_hub
    hub.update_telemetry(fps=24.5, latency_ms=18.2, train_state="APPROACHING")

    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["live_fps"] == 24.5
    assert data["live_latency_ms"] == 18.2
    assert data["train_state"] == "APPROACHING"
    assert "synthetic benchmark" in data["synthetic_accuracy"]


def test_dashboard_events_push_and_api(client_and_hub):
    client, hub, _ = client_and_hub
    hub.push_event({
        "event_type": "DELIBERATE_JUMP",
        "severity": "L3",
        "track_id": 3,
        "zone": "track_bed",
        "confidence": 0.95,
        "action": "ETS Traction Cut"
    })

    resp = client.get("/api/events")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["events"]) == 1
    assert data["events"][0]["event_type"] == "DELIBERATE_JUMP"


def test_dashboard_acknowledge_and_buzzer(client_and_hub):
    client, hub, buzzer = client_and_hub
    # Trigger buzzer
    buzzer.trigger("L3")
    hub.set_overlooker_alert({"title": "L3 Jump", "body": "Alert body"})
    assert buzzer.get_status()["active"]
    assert hub.active_alert is not None

    # Call acknowledge endpoint
    resp = client.post("/api/acknowledge")
    assert resp.status_code == 200
    assert not buzzer.get_status()["active"]
    assert hub.active_alert is None


def test_dashboard_buzzer_mute_and_test(client_and_hub):
    client, hub, buzzer = client_and_hub
    resp = client.post("/api/buzzer/toggle_mute")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["muted"] is True
    assert buzzer.get_status()["muted"] is True

    resp_test = client.post("/api/buzzer/test", json={"pattern": "L1"})
    assert resp_test.status_code == 200


def test_dashboard_train_state_and_ets(client_and_hub):
    client, hub, _ = client_and_hub
    resp = client.post("/api/train_state", json={"state": "APPROACHING"})
    assert resp.status_code == 200
    assert hub.train_state == "APPROACHING"

    resp_ets = client.post("/api/ets/trigger")
    assert resp_ets.status_code == 200
    assert hub.ets_status == "ENGAGED"

    resp_reset = client.post("/api/ets/reset")
    assert resp_reset.status_code == 200
    assert hub.ets_status == "DISENGAGED"
