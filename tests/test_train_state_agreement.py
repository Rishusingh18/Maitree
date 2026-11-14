"""Automated tests ensuring 'Simulate Train Movement' panel, header badge, and telemetry always agree."""
import pytest
from unittest.mock import MagicMock
from metro_safety.demo.hub import LiveHub
from metro_safety.demo.dashboard import create_dashboard_app
from metro_safety.escalation.buzzer import HardwareBuzzer
from metro_safety.perception.train_state import SimulatedTrainStateProvider


@pytest.fixture
def test_setup():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    train_provider = SimulatedTrainStateProvider()
    hub = LiveHub(cctv_id="CCTV_TEST", station_id="STN_TEST", platform_id="PL_TEST")
    hub.set_hardware_components(buzzer=buzzer, train_state_provider=train_provider)

    app = create_dashboard_app(hub)
    app.config["TESTING"] = True
    client = app.test_client()

    yield client, hub, train_provider
    buzzer.stop()


def test_train_state_backend_telemetry_agreement(test_setup):
    """Assert /api/train_state updates telemetry and backend provider synchronously."""
    client, hub, train_provider = test_setup

    states = ["ABSENT", "APPROACHING", "PRESENT", "ABSENT"]
    for st in states:
        # POST train state change
        post_resp = client.post("/api/train_state", json={"state": st})
        assert post_resp.status_code == 200
        res = post_resp.get_json()
        assert res["status"] == "ok"
        assert res["train_state"] == st

        # Verify train provider matches
        assert train_provider.get_state().value == st

        # Verify /api/telemetry matches
        tel_resp = client.get("/api/telemetry")
        assert tel_resp.status_code == 200
        tel_data = tel_resp.get_json()
        assert tel_data["train_state"] == st


def test_dashboard_template_train_state_dom_elements(test_setup):
    """Assert dashboard HTML contains exact IDs and dynamic highlight logic matching backend train state."""
    client, hub, _ = test_setup
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Header badge & Telemetry value IDs
    assert 'id="train-state-badge"' in html
    assert 'id="tel-train"' in html

    # Controller panel button IDs
    assert 'id="btn-train-absent"' in html
    assert 'id="btn-train-approaching"' in html
    assert 'id="btn-train-present"' in html

    # Active class style definition
    assert '.btn-active-train' in html

    # Assert JavaScript logic strictly uses data.train_state from telemetry rather than click state
    assert "btnAbsent.classList.toggle('btn-active-train', state === 'ABSENT')" in html
    assert "btnApp.classList.toggle('btn-active-train', state === 'APPROACHING')" in html
    assert "btnPres.classList.toggle('btn-active-train', state === 'PRESENT')" in html

