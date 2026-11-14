"""Comprehensive unit test suite for train-state gating of yellow-line geofence."""
import time
import pytest
import numpy as np

from metro_safety.perception.train_state import (
    TrainState,
    TrainStateRecord,
    TrainStateProvider,
    SimulatedProvider,
    VisionProvider,
    SimulatedTrainStateProvider,
    VisionTrainStateProvider,
    ScheduleFileTrainStateProvider,
)
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.perception.base import Detection
from metro_safety.events.models import EventType
from metro_safety.zones.geometry import ZoneManager


# =====================================================================
# Fixtures & Helpers
# =====================================================================

class MockZoneManager:
    """Mock zone manager returning specified zones."""
    def __init__(self, zone_map=None):
        self.zone_map = zone_map or {}
        self.zone_version = 1

    def get_zone_for_point(self, pt):
        return self.zone_map.get(pt, "platform")


def make_det(track_id: int, foot_pt: tuple) -> Detection:
    x, y = foot_pt
    box = np.array([x - 20, y - 80, x + 20, y], dtype=np.float32)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 2] = 0.9  # high confidence
    # 5, 6 shoulders
    kpts[5] = [x - 10, y - 60, 0.9]
    kpts[6] = [x + 10, y - 60, 0.9]
    # 11, 12 hips
    kpts[11] = [x - 8, y - 40, 0.9]
    kpts[12] = [x + 8, y - 40, 0.9]
    # 15, 16 ankles
    kpts[15] = [x - 5, y, 0.9]
    kpts[16] = [x + 5, y, 0.9]
    return Detection(
        box=box,
        confidence=0.95,
        keypoints=kpts,
        track_id=track_id
    )


# =====================================================================
# 1. TrainState Enum & String Parsing
# =====================================================================

def test_train_state_enum_members():
    """Assert all required enum states are present."""
    assert TrainState.ABSENT.value == "ABSENT"
    assert TrainState.APPROACHING.value == "APPROACHING"
    assert TrainState.STOPPED_DOORS_OPEN.value == "STOPPED_DOORS_OPEN"
    assert TrainState.DEPARTING.value == "DEPARTING"
    assert TrainState.PRESENT.value == "PRESENT"


def test_train_state_from_str_aliases():
    """Assert from_str handles lowercase, uppercase, and aliases."""
    assert TrainState.from_str("absent") == TrainState.ABSENT
    assert TrainState.from_str("APPROACHING") == TrainState.APPROACHING
    assert TrainState.from_str("stopped_doors_open") == TrainState.STOPPED_DOORS_OPEN
    assert TrainState.from_str("DEPARTING") == TrainState.DEPARTING

    # Aliases
    assert TrainState.from_str("STOPPED") == TrainState.STOPPED_DOORS_OPEN
    assert TrainState.from_str("DOORS_OPEN") == TrainState.STOPPED_DOORS_OPEN
    assert TrainState.from_str("BOARDING") == TrainState.STOPPED_DOORS_OPEN
    assert TrainState.from_str("DOCKED") == TrainState.STOPPED_DOORS_OPEN
    assert TrainState.from_str("PRESENT") == TrainState.PRESENT
    assert TrainState.from_str("LEAVING") == TrainState.DEPARTING
    assert TrainState.from_str("DEPART") == TrainState.DEPARTING
    assert TrainState.from_str("ARRIVING") == TrainState.APPROACHING
    assert TrainState.from_str("NONE") == TrainState.ABSENT

    with pytest.raises(ValueError):
        TrainState.from_str("FLYING_AWAY")


# =====================================================================
# 2. Transition Tests (All Transitions)
# =====================================================================

def test_geofence_active_in_absent_and_approaching():
    """Geofence is ACTIVE in ABSENT and APPROACHING."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0

    # 1. Initial state ABSENT -> ACTIVE
    provider.set_state(TrainState.ABSENT, timestamp=t0)
    assert provider.is_geofence_active(timestamp=t0) is True

    # 2. Transition ABSENT -> APPROACHING -> ACTIVE
    provider.set_state(TrainState.APPROACHING, timestamp=t0 + 5.0)
    assert provider.is_geofence_active(timestamp=t0 + 5.0) is True


def test_geofence_disabled_in_stopped_doors_open():
    """Geofence is DISABLED only in STOPPED_DOORS_OPEN."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0

    # APPROACHING -> STOPPED_DOORS_OPEN -> DISABLED
    provider.set_state(TrainState.APPROACHING, timestamp=t0)
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0 + 10.0)
    assert provider.is_geofence_active(timestamp=t0 + 10.0) is False
    assert provider.is_geofence_active(timestamp=t0 + 15.0) is False  # while stopped and fresh


def test_geofence_disabled_in_present_backward_compat():
    """PRESENT state behaves identically to STOPPED_DOORS_OPEN."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0
    provider.set_state(TrainState.PRESENT, timestamp=t0)
    assert provider.is_geofence_active(timestamp=t0) is False


def test_transition_stopped_to_departing_grace_period():
    """On transition STOPPED_DOORS_OPEN -> DEPARTING, geofence re-enables after grace period."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0

    # 1. Train stopped with doors open
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0)
    assert provider.is_geofence_active(timestamp=t0) is False

    # 2. Transition to DEPARTING at t = 1020.0
    t_dep = 1020.0
    provider.set_state(TrainState.DEPARTING, timestamp=t_dep)

    # At transition instant (dt = 0.0s): DISABLED
    assert provider.is_geofence_active(timestamp=t_dep) is False

    # Inside grace period (dt = 1.0s < 2.0s): DISABLED
    assert provider.is_geofence_active(timestamp=t_dep + 1.0) is False

    # Just before grace period ends (dt = 1.99s < 2.0s): DISABLED
    assert provider.is_geofence_active(timestamp=t_dep + 1.99) is False

    # At exact grace period expiry (dt = 2.0s >= 2.0s): ACTIVE
    assert provider.is_geofence_active(timestamp=t_dep + 2.0) is True

    # After grace period (dt = 3.5s): ACTIVE
    assert provider.is_geofence_active(timestamp=t_dep + 3.5) is True


def test_transition_departing_to_absent():
    """Transition DEPARTING -> ABSENT remains ACTIVE."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0
    provider.set_state(TrainState.DEPARTING, timestamp=t0)
    provider.set_state(TrainState.ABSENT, timestamp=t0 + 20.0)
    assert provider.is_geofence_active(timestamp=t0 + 20.0) is True


def test_direct_transition_absent_to_departing_no_grace():
    """If state transitions to DEPARTING from ABSENT, grace period does not apply."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 1000.0
    provider.set_state(TrainState.ABSENT, timestamp=t0)
    provider.set_state(TrainState.DEPARTING, timestamp=t0 + 5.0)
    # Since previous state was not STOPPED_DOORS_OPEN/PRESENT, geofence is active immediately
    assert provider.is_geofence_active(timestamp=t0 + 5.0) is True


# =====================================================================
# 3. Grace Period Configuration & Heartbeats
# =====================================================================

def test_configurable_grace_period():
    """Grace period can be configured to custom value (e.g. 5.0s)."""
    provider = SimulatedProvider(grace_period_sec=5.0, fail_safe_timeout_sec=20.0)
    t0 = 100.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0)
    t_dep = 120.0
    provider.set_state(TrainState.DEPARTING, timestamp=t_dep)

    # At dt = 3.0s < 5.0s: still in grace period -> False
    assert provider.is_geofence_active(timestamp=t_dep + 3.0) is False
    # At dt = 5.0s >= 5.0s: grace expired -> True
    assert provider.is_geofence_active(timestamp=t_dep + 5.0) is True


def test_heartbeat_during_departing_preserves_transition_timestamp():
    """Periodic updates with same state DEPARTING do not reset transition_timestamp."""
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    t0 = 100.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0)
    t_dep = 110.0
    provider.set_state(TrainState.DEPARTING, timestamp=t_dep)

    # Heartbeat at t = 110.5
    provider.set_state(TrainState.DEPARTING, timestamp=t_dep + 0.5)
    rec = provider.get_state_record()
    assert rec.transition_timestamp == t_dep  # transition timestamp unchanged

    # At t = 111.0 (< 2.0s from transition): disabled
    assert provider.is_geofence_active(timestamp=t_dep + 1.0) is False

    # At t = 112.1 (>= 2.0s from transition): active
    assert provider.is_geofence_active(timestamp=t_dep + 2.1) is True


# =====================================================================
# 4. Fail-Safe Tests
# =====================================================================

def test_fail_safe_on_unknown_or_missing_platform():
    """Unknown platform treats geofence as ACTIVE."""
    provider = SimulatedProvider()
    assert provider.is_geofence_active(platform_id="UNKNOWN_PL_99") is True


def test_fail_safe_on_stale_update_in_stopped_state():
    """If provider receives no update for T seconds while STOPPED_DOORS_OPEN, fail-safe activates geofence."""
    fail_safe_t = 10.0
    provider = SimulatedProvider(fail_safe_timeout_sec=fail_safe_t)
    t0 = 500.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0)

    # While fresh (e.g. 5.0s elapsed): disabled
    assert provider.is_geofence_active(timestamp=t0 + 5.0) is False
    assert provider.is_update_stale(timestamp=t0 + 5.0) is False

    # Right before timeout (9.9s elapsed): disabled
    assert provider.is_geofence_active(timestamp=t0 + 9.9) is False

    # Exceeding timeout (10.1s elapsed): fail-safe triggers -> ACTIVE
    assert provider.is_update_stale(timestamp=t0 + 10.1) is True
    assert provider.is_geofence_active(timestamp=t0 + 10.1) is True


def test_touch_heartbeat_prevents_fail_safe_trip():
    """Calling touch() updates record timestamp and keeps geofence disabled during long train stops."""
    provider = SimulatedProvider(fail_safe_timeout_sec=10.0)
    t0 = 500.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, timestamp=t0)

    # Train dwells for 25s, but receives heartbeats every 5s
    for dt in [5.0, 10.0, 15.0, 20.0, 25.0]:
        provider.touch(timestamp=t0 + dt)
        assert provider.is_geofence_active(timestamp=t0 + dt) is False

    # If heartbeat stops after 25s, at 36s (> 10s after last touch): fail-safe triggers
    assert provider.is_geofence_active(timestamp=t0 + 36.0) is True


# =====================================================================
# 5. Per-Platform State Isolation & Logging
# =====================================================================

def test_per_platform_isolation():
    """State changes on platform 1 do not alter platform 2."""
    provider = SimulatedProvider(grace_period_sec=2.0)
    t0 = 100.0

    provider.set_state(TrainState.STOPPED_DOORS_OPEN, platform_id="PL_DOWN_01", timestamp=t0)
    provider.set_state(TrainState.APPROACHING, platform_id="PL_UP_02", timestamp=t0)

    # PL_DOWN_01 is STOPPED -> geofence disabled
    assert provider.get_state(platform_id="PL_DOWN_01") == TrainState.STOPPED_DOORS_OPEN
    assert provider.is_geofence_active(platform_id="PL_DOWN_01", timestamp=t0) is False

    # PL_UP_02 is APPROACHING -> geofence active
    assert provider.get_state(platform_id="PL_UP_02") == TrainState.APPROACHING
    assert provider.is_geofence_active(platform_id="PL_UP_02", timestamp=t0) is True


def test_train_state_record_fields():
    """Assert TrainStateRecord captures complete provenance."""
    provider = SimulatedProvider()
    rec = provider.set_state(
        state=TrainState.DEPARTING,
        platform_id="PL_TEST_01",
        timestamp=123.456,
        source="mqtt_test",
        metadata={"signal": "GREEN"}
    )
    assert isinstance(rec, TrainStateRecord)
    assert rec.state == TrainState.DEPARTING
    assert rec.platform_id == "PL_TEST_01"
    assert rec.timestamp == 123.456
    assert rec.source == "mqtt_test"
    assert rec.metadata == {"signal": "GREEN"}


# =====================================================================
# 6. MQTT Ingestion Tests (SimulatedProvider)
# =====================================================================

def test_mqtt_topic_and_payload_parsing():
    """SimulatedProvider parses various MQTT topics and payload formats."""
    provider = SimulatedProvider()
    t0 = 200.0

    # 1. Plain string with platform in topic: metro/train_state/PL_NORTH
    rec1 = provider.handle_mqtt_message("metro/train_state/PL_NORTH", "STOPPED_DOORS_OPEN", timestamp=t0)
    assert rec1 is not None
    assert rec1.platform_id == "PL_NORTH"
    assert rec1.state == TrainState.STOPPED_DOORS_OPEN

    # 2. JSON payload with platform_id included: trains/PL_SOUTH/state
    rec2 = provider.handle_mqtt_message(
        "trains/PL_SOUTH/state",
        '{"state": "DEPARTING", "platform_id": "PL_SOUTH", "timestamp": 205.0}'
    )
    assert rec2 is not None
    assert rec2.platform_id == "PL_SOUTH"
    assert rec2.state == TrainState.DEPARTING
    assert rec2.timestamp == 205.0

    # 3. Default topic without platform in topic: metro/train_state
    rec3 = provider.simulate_mqtt_message("metro/train_state", '{"state": "APPROACHING"}', timestamp=t0 + 10.0)
    assert rec3 is not None
    assert rec3.platform_id == provider.default_platform_id
    assert rec3.state == TrainState.APPROACHING

    # 4. Binary payload
    rec4 = provider.handle_mqtt_message("metro/train_state/PL_EAST", b"ABSENT", timestamp=t0 + 15.0)
    assert rec4 is not None
    assert rec4.platform_id == "PL_EAST"
    assert rec4.state == TrainState.ABSENT


# =====================================================================
# 7. VisionProvider Stub Tests
# =====================================================================

def test_vision_provider_stub():
    """Assert VisionProvider conforms to TrainStateProvider contract."""
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    provider = VisionProvider(track_polygon=poly)

    assert provider.get_state() == TrainState.ABSENT
    assert provider.is_geofence_active() is True

    provider.set_state(TrainState.STOPPED_DOORS_OPEN)
    assert provider.is_geofence_active() is False

    provider.touch()
    assert provider.is_geofence_active() is False


# =====================================================================
# 8. Behavioral Classification: Person Beyond Line During DEPARTING
# =====================================================================

def test_person_already_beyond_line_when_departing():
    """CRITICAL TEST CASE:
    A person is standing beyond the yellow line during STOPPED_DOORS_OPEN.
    No alert is raised while boarding (geofence disabled).
    Train switches to DEPARTING.
    During grace period (2.0s), still no alert.
    After grace period expires, person remains beyond yellow line -> LINE_CROSSING alert triggers.
    """
    classifier = RuleBasedEventClassifier(line_crossing_thresh=4)
    provider = SimulatedProvider(grace_period_sec=2.0, fail_safe_timeout_sec=10.0)
    pid = "PL_DOWN_01"

    foot_in_yellow = (100, 200)
    zone_mgr = MockZoneManager(zone_map={foot_in_yellow: "yellow_band"})

    # 1. Train is STOPPED_DOORS_OPEN (passengers boarding)
    t_doors = 100.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, platform_id=pid, timestamp=t_doors)
    st = provider.get_state(pid)

    # Person stands in yellow band for 10 frames
    for f in range(1, 11):
        t_f = t_doors + (f * 0.1)
        det = make_det(track_id=1, foot_pt=foot_in_yellow)
        events = classifier.process_frame(
            frame_idx=f,
            timestamp=t_f,
            tracked_detections=[det],
            zone_manager=zone_mgr,
            train_state=st,
            train_state_provider=provider,
            platform_id=pid
        )
        # MUST BE SUPPRESSED: geofence disabled during boarding
        assert len(events) == 0, f"Expected 0 events in STOPPED_DOORS_OPEN, got {events}"

    # Verify track machine kept tracking consecutive frames
    assert classifier.track_machines[1].yellow_consecutive >= 10

    # 2. State switches to DEPARTING at t = 105.0
    t_dep = 105.0
    provider.set_state(TrainState.DEPARTING, platform_id=pid, timestamp=t_dep)
    st_dep = provider.get_state(pid)

    # 3. Inside grace period (e.g. t = 106.0, dt = 1.0s < 2.0s)
    # Person still beyond yellow line
    f_grace = 15
    det_grace = make_det(track_id=1, foot_pt=foot_in_yellow)
    events_grace = classifier.process_frame(
        frame_idx=f_grace,
        timestamp=t_dep + 1.0,
        tracked_detections=[det_grace],
        zone_manager=zone_mgr,
        train_state=st_dep,
        train_state_provider=provider,
        platform_id=pid
    )
    # MUST BE SUPPRESSED: still inside 2.0s grace period
    assert len(events_grace) == 0, f"Expected 0 events during grace period, got {events_grace}"

    # 4. Grace period EXPIRES (t = 107.5, dt = 2.5s >= 2.0s)
    # Person remains beyond yellow line
    f_alert = 20
    det_alert = make_det(track_id=1, foot_pt=foot_in_yellow)
    events_alert = classifier.process_frame(
        frame_idx=f_alert,
        timestamp=t_dep + 2.5,
        tracked_detections=[det_alert],
        zone_manager=zone_mgr,
        train_state=st_dep,
        train_state_provider=provider,
        platform_id=pid
    )
    # MUST TRIGGER LINE_CROSSING ALERT
    assert len(events_alert) == 1
    assert events_alert[0].event_type == EventType.LINE_CROSSING
    assert events_alert[0].track_id == 1
    assert events_alert[0].zone == "yellow_band"


# =====================================================================
# 9. Track-Zone Detection Remains ALWAYS Active
# =====================================================================

def test_track_zone_detection_always_active_in_stopped_state():
    """Track-zone (fence/jump/track intrusion) detection stays ALWAYS active regardless of train state."""
    classifier = RuleBasedEventClassifier(track_intrusion_thresh=2)
    provider = SimulatedProvider()
    pid = "PL_DOWN_01"

    # Train is STOPPED_DOORS_OPEN (yellow-line alerts disabled)
    t0 = 50.0
    provider.set_state(TrainState.STOPPED_DOORS_OPEN, platform_id=pid, timestamp=t0)
    st = provider.get_state(pid)

    foot_on_track = (100, 450)
    zone_mgr = MockZoneManager(zone_map={foot_on_track: "track_bed"})

    # Frame 1: person enters track bed
    det1 = make_det(track_id=2, foot_pt=foot_on_track)
    classifier.process_frame(
        frame_idx=1,
        timestamp=t0 + 0.1,
        tracked_detections=[det1],
        zone_manager=zone_mgr,
        train_state=st,
        train_state_provider=provider,
        platform_id=pid
    )

    # Frame 2: person remains on track bed
    det2 = make_det(track_id=2, foot_pt=foot_on_track)
    events = classifier.process_frame(
        frame_idx=2,
        timestamp=t0 + 0.2,
        tracked_detections=[det2],
        zone_manager=zone_mgr,
        train_state=st,
        train_state_provider=provider,
        platform_id=pid
    )

    # TRACK INTRUSION MUST TRIGGER EVEN IN STOPPED_DOORS_OPEN
    assert len(events) >= 1
    assert events[0].event_type in (EventType.TRACK_INTRUSION, EventType.ACCIDENTAL_FALL, EventType.DELIBERATE_JUMP)
    assert events[0].track_id == 2
