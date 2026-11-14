"""Tests for frontend dead backend detection, banner, placeholder, and error toast handling."""
import pytest
from unittest.mock import MagicMock
from metro_safety.demo.hub import LiveHub
from metro_safety.demo.dashboard import create_dashboard_app
from metro_safety.escalation.buzzer import HardwareBuzzer


@pytest.fixture
def client():
    mock_beep = MagicMock()
    buzzer = HardwareBuzzer(audio_enabled=True, beep_fn=mock_beep)
    hub = LiveHub(cctv_id="CCTV_TEST", station_id="STN_TEST", platform_id="PL_TEST")
    hub.set_hardware_components(buzzer=buzzer)

    app = create_dashboard_app(hub)
    app.config["TESTING"] = True
    c = app.test_client()

    yield c
    buzzer.stop()


def test_dead_backend_dom_and_script_elements(client):
    """Verify HTML elements, banners, placeholders, and JavaScript dead-backend handlers."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # 1. Dead Backend Banner & Last Update Time element
    assert 'id="backend-dead-banner"' in html
    assert "BACKEND DISCONNECTED - data is stale" in html
    assert 'id="backend-dead-time"' in html

    # 2. No Signal Video Placeholder
    assert 'id="no-signal-placeholder"' in html
    assert "NO SIGNAL" in html
    assert "Backend server disconnected" in html

    # 3. Toast notifications container and styling
    assert 'id="toast-container"' in html
    assert ".toast {" in html
    assert "showErrorToast" in html

    # 4. Failed poll counter and automatic dead-state transition
    assert "consecutiveFailedPolls++" in html
    assert "consecutiveFailedPolls >= 3" in html
    assert "setBackendDeadState(true)" in html

    # 5. Recovery on poll success
    assert "consecutiveFailedPolls = 0" in html
    assert "setBackendDeadState(false)" in html

    # 6. Button disablement when dead, re-enablement on recovery
    assert "buttons.forEach(b => b.disabled = true)" in html
    assert "buttons.forEach(b => b.disabled = false)" in html

