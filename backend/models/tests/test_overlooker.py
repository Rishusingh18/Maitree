"""Unit tests for OverlookerNotifier."""
import time
import json
from pathlib import Path
from metro_safety.escalation.overlooker import OverlookerNotifier

def test_overlooker_severity_filtering(tmp_path):
    audit_log = tmp_path / "audit.log"
    notifier = OverlookerNotifier(
        cooldown_sec=1.0,
        desktop_notification=False,
        audit_log_file=str(audit_log)
    )

    # L1 should be suppressed (local platform buzzer only)
    res = notifier.notify(
        event_type="LINE_CROSSING",
        severity="L1",
        track_id=1,
        zone="yellow_band",
        confidence=0.9,
        timestamp=time.time()
    )
    assert res["status"] == "suppressed"

    # L2 should be dispatched
    res = notifier.notify(
        event_type="TRACK_INTRUSION",
        severity="L2",
        track_id=1,
        zone="track_bed",
        confidence=0.95,
        timestamp=time.time()
    )
    assert res["status"] == "dispatched"
    assert "TRACK_INTRUSION" in res["record"]["event_type"]


def test_overlooker_cooldown(tmp_path):
    audit_log = tmp_path / "audit.log"
    # Cooldown of 0.3s
    notifier = OverlookerNotifier(
        cooldown_sec=0.3,
        desktop_notification=False,
        audit_log_file=str(audit_log)
    )

    t0 = time.time()
    # First jump event
    res1 = notifier.notify(
        event_type="DELIBERATE_JUMP",
        severity="L3",
        track_id=1,
        zone="track_bed",
        confidence=0.98,
        timestamp=t0
    )
    assert res1["status"] == "dispatched"

    # Immediate second jump event (within 0.3s cooldown)
    res2 = notifier.notify(
        event_type="DELIBERATE_JUMP",
        severity="L3",
        track_id=1,
        zone="track_bed",
        confidence=0.99,
        timestamp=t0 + 0.05
    )
    assert res2["status"] == "suppressed"
    assert "Cooldown active" in res2["reason"]

    # Different event type (FALL) should NOT be blocked by JUMP's cooldown
    res_fall = notifier.notify(
        event_type="ACCIDENTAL_FALL",
        severity="L3",
        track_id=2,
        zone="track_bed",
        confidence=0.92,
        timestamp=t0 + 0.06
    )
    assert res_fall["status"] == "dispatched"

    # After cooldown expires for JUMP
    time.sleep(0.35)
    res3 = notifier.notify(
        event_type="DELIBERATE_JUMP",
        severity="L3",
        track_id=1,
        zone="track_bed",
        confidence=0.97,
        timestamp=time.time()
    )
    assert res3["status"] == "dispatched"


def test_overlooker_silent_distress_and_audit(tmp_path):
    audit_log = tmp_path / "audit.log"
    notifier = OverlookerNotifier(
        cooldown_sec=0.5,
        desktop_notification=False,
        audit_log_file=str(audit_log)
    )

    t = time.time()
    res = notifier.notify(
        event_type="DISTRESS_GESTURE",
        severity="DISTRESS",
        track_id=4,
        zone="platform",
        confidence=0.91,
        timestamp=t
    )
    assert res["status"] == "dispatched"
    assert res["record"]["severity"] == "DISTRESS"

    # Verify audit file contents
    assert audit_log.exists()
    with open(audit_log, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "DISTRESS_GESTURE"
        assert data["track_id"] == 4

