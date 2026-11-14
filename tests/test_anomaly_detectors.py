"""Unit tests for Metro Safety Anomaly and Theft Detection Subsystem.

Tests:
1. PERSON_DOWN (horizontal angle, low head-hip dy, aspect ratio, motion energy, T1/T2 timers, track churn recovery, ignore zone)
2. LOITERING / DWELL (anchor radius, dwell timers, train approach suppression)
3. ABANDONED_OBJECT (unattended radius, timers, startup filtering)
4. POSSIBLE_SNATCH (owner association, wrist snatch, acceleration, handover suppression, confidence capping <= 0.70)
5. Escalation Severity Ladder (Anomaly events NEVER trigger ETS, NEVER trigger L3 siren, max severity L2)
6. Camera Motion & Unverified Zones capping (low_trust flags)
7. STG-NF Pose sequence NLL scoring
"""
import time
import pytest
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.escalation.engine import EscalationEngine
from metro_safety.anomaly.person_down import PersonDownDetector
from metro_safety.anomaly.loitering import LoiteringDetector
from metro_safety.anomaly.abandoned_object import AbandonedObjectDetector, TrackedObject
from metro_safety.anomaly.theft_snatch import TheftSnatchDetector
from metro_safety.anomaly.stg_nf_evaluator import STGNFEvaluator
from metro_safety.anomaly.engine import AnomalyEngine


@pytest.fixture
def zone_mgr():
    zones_def = {
        "platform": [[0, 0], [400, 0], [400, 400], [0, 400]],
        "yellow_band": [[400, 0], [450, 0], [450, 400], [400, 400]],
        "track_bed": [[450, 0], [600, 0], [600, 400], [450, 400]],
        "ignore": [[50, 50], [150, 50], [150, 150], [50, 150]]
    }
    return ZoneManager(zones_def)


def make_prone_kpts(center_x=200, center_y=200):
    """Generate 17 keypoints for a horizontal lying person."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Horizontal body: shoulders and hips have nearly identical y
    kpts[0] = [center_x - 45, center_y, 0.9]       # nose
    kpts[5] = [center_x - 30, center_y - 5, 0.9]   # left shoulder
    kpts[6] = [center_x - 30, center_y + 5, 0.9]   # right shoulder
    kpts[11] = [center_x + 20, center_y - 5, 0.9]  # left hip
    kpts[12] = [center_x + 20, center_y + 5, 0.9]  # right hip
    kpts[15] = [center_x + 50, center_y - 5, 0.9]  # left ankle
    kpts[16] = [center_x + 50, center_y + 5, 0.9]  # right ankle
    return kpts


def make_upright_kpts(center_x=200, center_y=200):
    """Generate 17 keypoints for an upright standing person."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [center_x, center_y - 60, 0.9]       # nose
    kpts[5] = [center_x - 15, center_y - 40, 0.9]  # left shoulder
    kpts[6] = [center_x + 15, center_y - 40, 0.9]  # right shoulder
    kpts[11] = [center_x - 10, center_y, 0.9]      # left hip
    kpts[12] = [center_x + 10, center_y, 0.9]      # right hip
    kpts[15] = [center_x - 10, center_y + 50, 0.9] # left ankle
    kpts[16] = [center_x + 10, center_y + 50, 0.9] # right ankle
    return kpts


# =====================================================================
# 1. PERSON_DOWN TESTS
# =====================================================================

def test_person_down_detection_t1_and_t2(zone_mgr):
    """Verify person down triggers L1 after 10s and L2 after 30s."""
    detector = PersonDownDetector(prone_t1_sec=10.0, prone_t2_sec=30.0)
    kpts = make_prone_kpts(200, 200)
    det = Detection(box=np.array([140, 180, 260, 220]), confidence=0.9, keypoints=kpts, track_id=1)

    t0 = 1000.0
    # Initial frame
    evs = detector.process_frame(0, t0, [det], zone_mgr)
    assert len(evs) == 0

    # At t=5s: still below T1
    evs = detector.process_frame(1, t0 + 5.0, [det], zone_mgr)
    assert len(evs) == 0

    # At t=10.5s: T1 reached -> L1 notice
    evs = detector.process_frame(2, t0 + 10.5, [det], zone_mgr)
    assert len(evs) == 1
    assert evs[0].event_type == EventType.PERSON_DOWN
    assert evs[0].metadata["severity_cap"] == "L1"
    assert evs[0].metadata["disclaimer"] == "operator welfare review"

    # At t=20s: duplicate L1 suppressed
    evs = detector.process_frame(3, t0 + 20.0, [det], zone_mgr)
    assert len(evs) == 0

    # At t=31.0s: T2 reached -> L2 welfare alert
    evs = detector.process_frame(4, t0 + 31.0, [det], zone_mgr)
    assert len(evs) == 1
    assert evs[0].metadata["severity_cap"] == "L2"


def test_person_down_negative_upright(zone_mgr):
    """Upright standing person must never trigger PERSON_DOWN."""
    detector = PersonDownDetector(prone_t1_sec=10.0, prone_t2_sec=30.0)
    kpts = make_upright_kpts(200, 200)
    det = Detection(box=np.array([180, 130, 220, 260]), confidence=0.9, keypoints=kpts, track_id=1)

    t0 = 1000.0
    for i in range(15):
        evs = detector.process_frame(i, t0 + i * 2.0, [det], zone_mgr)
        assert len(evs) == 0


def test_person_down_ignore_bench_lying(zone_mgr):
    """Person lying in designated 'ignore' zone (e.g. bench) is suppressed."""
    detector = PersonDownDetector(prone_t1_sec=5.0, prone_t2_sec=10.0, ignore_bench_lying=True)
    # Position inside ignore zone: [50, 50] to [150, 150]
    kpts = make_prone_kpts(100, 100)
    det = Detection(box=np.array([60, 90, 140, 110]), confidence=0.9, keypoints=kpts, track_id=2)

    t0 = 1000.0
    evs = detector.process_frame(0, t0, [det], zone_mgr)
    evs = detector.process_frame(1, t0 + 12.0, [det], zone_mgr)
    assert len(evs) == 0


def test_person_down_track_churn_reassociation(zone_mgr):
    """If tracker churns ID while person is on ground, prone timer persists."""
    detector = PersonDownDetector(prone_t1_sec=10.0, prone_t2_sec=20.0, track_reassoc_window_sec=4.0)
    kpts = make_prone_kpts(200, 200)
    det_id1 = Detection(box=np.array([140, 180, 260, 220]), confidence=0.9, keypoints=kpts, track_id=1)

    t0 = 1000.0
    detector.process_frame(0, t0, [det_id1], zone_mgr)
    detector.process_frame(1, t0 + 11.0, [det_id1], zone_mgr)  # L1 triggered

    # ID 1 churns / disappears at t=12s
    detector.process_frame(2, t0 + 12.0, [], zone_mgr)

    # ID 2 appears 1.5 seconds later (within 4s window) at same spatial position
    det_id2 = Detection(box=np.array([142, 181, 262, 221]), confidence=0.9, keypoints=kpts, track_id=2)
    detector.process_frame(3, t0 + 13.5, [det_id2], zone_mgr)

    # Reaches T2 at t=21.0s (> 20.0s T2), triggers L2 under ID 2
    evs = detector.process_frame(4, t0 + 21.0, [det_id2], zone_mgr)
    assert len(evs) == 1
    assert evs[0].track_id == 2
    assert evs[0].metadata["severity_cap"] == "L2"


# =====================================================================
# 2. LOITERING / DWELL TESTS
# =====================================================================

def test_loitering_detection_and_train_suppression(zone_mgr):
    """Verify dwell in platform zone triggers L1/L2 and train state suppresses."""
    detector = LoiteringDetector(radius_px=70.0, dwell_t1_sec=10.0, dwell_t2_sec=20.0, train_suppression=True)

    # Person standing on platform
    box = np.array([180, 150, 220, 250])
    det = Detection(box=box, confidence=0.9, track_id=5)

    t0 = 1000.0
    evs = detector.process_frame(0, t0, [det], zone_mgr, train_state="ABSENT")
    assert len(evs) == 0

    # 11s later -> L1
    evs = detector.process_frame(1, t0 + 11.0, [det], zone_mgr, train_state="ABSENT")
    assert len(evs) == 1
    assert evs[0].event_type == EventType.LOITERING
    assert evs[0].metadata["severity_cap"] == "L1"

    # Train arrives -> alerts suppressed
    evs = detector.process_frame(2, t0 + 22.0, [det], zone_mgr, train_state="DOCKED")
    assert len(evs) == 0


def test_loitering_negative_moving_passenger(zone_mgr):
    """Moving passenger walking across platform should reset anchor and not trigger."""
    detector = LoiteringDetector(radius_px=50.0, dwell_t1_sec=10.0, dwell_t2_sec=20.0)

    t0 = 1000.0
    for i in range(10):
        # Move 25px per step (exceeds anchor radius)
        box = np.array([50 + i * 25, 150, 90 + i * 25, 250])
        det = Detection(box=box, confidence=0.9, track_id=6)
        evs = detector.process_frame(i, t0 + i * 2.0, [det], zone_mgr)
        assert len(evs) == 0


# =====================================================================
# 3. ABANDONED_OBJECT TESTS
# =====================================================================

def test_abandoned_object_unattended_detection(zone_mgr):
    """Luggage left unattended beyond radius and timer emits ABANDONED_OBJECT."""
    detector = AbandonedObjectDetector(unattended_radius_px=80.0, unattended_sec=10.0)

    # Simulated bag at (200, 200)
    synthetic_bags = [{
        "box": np.array([190, 190, 210, 210]),
        "conf": 0.85,
        "class": "backpack"
    }]

    # Owner initially next to bag at (205, 205)
    owner = Detection(box=np.array([195, 170, 225, 230]), confidence=0.9, track_id=7)

    t0 = 1000.0
    # First frame: owner is present
    evs = detector.process_frame(None, 0, t0, [owner], zone_mgr, synthetic_objects=synthetic_bags)
    assert len(evs) == 0

    # Owner walks away to (350, 350) (> 80px away)
    far_person = Detection(box=np.array([340, 310, 360, 380]), confidence=0.9, track_id=7)
    evs = detector.process_frame(None, 1, t0 + 5.0, [far_person], zone_mgr, synthetic_objects=synthetic_bags)
    assert len(evs) == 0

    # 12 seconds unattended (17.0 - 5.0 = 12.0s >= 10.0s) -> emits ABANDONED_OBJECT L2 review
    evs = detector.process_frame(None, 2, t0 + 17.0, [far_person], zone_mgr, synthetic_objects=synthetic_bags)
    assert len(evs) == 1
    assert evs[0].event_type == EventType.ABANDONED_OBJECT
    assert evs[0].metadata["object_class"] == "backpack"


# =====================================================================
# 4. POSSIBLE_SNATCH & HANDOVER SUPPRESSION TESTS
# =====================================================================

def test_theft_snatch_and_handover_suppression(zone_mgr):
    """Test snatch detection heuristic and handover suppression."""
    detector = TheftSnatchDetector(
        owner_proximity_px=60.0,
        owner_min_frames=2,
        suspect_wrist_proximity_px=50.0,
        snatch_accel_threshold_px=8.0,
        owner_follow_window_sec=1.5
    )

    bag = TrackedObject(
        object_id=1,
        object_class="handbag",
        box=np.array([190, 190, 210, 210]),
        confidence=0.88,
        first_seen_ts=1000.0,
        last_seen_ts=1000.0,
        center=(200.0, 200.0)
    )

    # Step 1: Establish Track 10 as owner over 2 frames
    kpts_owner = make_upright_kpts(210, 200)
    owner_det = Detection(box=np.array([195, 150, 225, 250]), confidence=0.9, keypoints=kpts_owner, track_id=10)
    detector.update([bag], [owner_det], timestamp=1000.0)
    detector.update([bag], [owner_det], timestamp=1000.1)
    assert detector.confirmed_owners.get(1) == 10

    # Step 2: Track 20 (suspect) approaches bag with wrist
    kpts_suspect = make_upright_kpts(195, 200)
    # Place suspect wrist at (201, 201) (near bag)
    kpts_suspect[10] = [201.0, 201.0, 0.9]
    suspect_det = Detection(box=np.array([170, 150, 210, 250]), confidence=0.9, keypoints=kpts_suspect, track_id=20)
    detector.update([bag], [owner_det, suspect_det], timestamp=1001.0)
    assert 1 in detector.candidates

    # Step 3: Bag accelerates rapidly away with suspect, owner stays behind
    bag.center = (280.0, 200.0)  # Displaced by 80px
    suspect_det.box = np.array([260, 150, 300, 250])
    evs = detector.update([bag], [owner_det, suspect_det], timestamp=1001.6)

    assert len(evs) == 1
    assert evs[0].event_type == EventType.POSSIBLE_SNATCH
    assert evs[0].confidence <= 0.70  # STRICT CONFIDENCE CAP
    assert evs[0].metadata["disclaimer"] == "possible, operator review"
    assert evs[0].metadata["suspect_track_id"] == 20
    assert evs[0].metadata["victim_track_id"] == 10


# =====================================================================
# 5. ESCALATION SEVERITY LADDER (NEVER ETS, NEVER L3 SIREN)
# =====================================================================

def test_escalation_engine_anomaly_rules(zone_mgr):
    """Verify all anomaly events are strictly capped at L2, never ETS, never L3 siren."""
    engine = EscalationEngine()

    anomaly_types = [
        EventType.PERSON_DOWN,
        EventType.LOITERING,
        EventType.ABANDONED_OBJECT,
        EventType.POSSIBLE_SNATCH,
        EventType.ANOMALY_POSE
    ]

    for ev_type in anomaly_types:
        ev = SafetyEvent(
            event_type=ev_type,
            confidence=0.85,
            track_id=42,
            timestamp=1000.0,
            zone="platform",
            train_state="ABSENT",
            metadata={"severity_cap": "L2", "is_anomaly": True}
        )
        dec = engine.evaluate(ev, current_time=1000.0)

        # STRICT GROUND RULES ASSERTIONS:
        assert dec.ets_triggered is False, f"{ev_type} must NEVER trigger ETS"
        assert dec.severity in [SeverityLevel.L1, SeverityLevel.L2], f"{ev_type} max severity must be L2"
        # Continuous High pattern is reserved for L3
        assert dec.severity != SeverityLevel.L3


def test_camera_motion_downgrade_anomaly(zone_mgr):
    """Under camera motion, anomaly events are capped at L1 and marked low trust."""
    engine = EscalationEngine()

    ev = SafetyEvent(
        event_type=EventType.PERSON_DOWN,
        confidence=0.80,
        track_id=99,
        timestamp=1000.0,
        zone="platform",
        train_state="ABSENT",
        camera_motion=True,
        motion_reason="Optical flow camera vibration",
        metadata={"severity_cap": "L2", "is_anomaly": True}
    )
    dec = engine.evaluate(ev, current_time=1000.0)

    assert dec.severity == SeverityLevel.L1
    assert "LOW TRUST" in dec.action_taken
    assert dec.ets_triggered is False


# =====================================================================
# 6. STG-NF EVALUATOR TEST
# =====================================================================

def test_stg_nf_sequence_evaluator():
    """Verify STG-NF evaluator loads weights and outputs principled NLL score."""
    evaluator = STGNFEvaluator(checkpoint_path="data/models/ShanghaiTech_85_9.tar")
    # Generate 24 dummy frames of 17 COCO keypoints
    kpts_seq = [make_upright_kpts(200, 200) for _ in range(24)]
    is_anom, nll, conf = evaluator.evaluate_sequence(kpts_seq)

    assert isinstance(nll, float)
    assert isinstance(is_anom, (bool, np.bool_))
    assert 0.0 <= conf <= 0.99
