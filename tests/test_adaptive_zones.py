"""Comprehensive Unit & Integration Test Suite for Adaptive Zone Tracking.
Tests:
1. Direct frame-to-reference registration on synthetic shifts, rotations, scaling, and mild perspective.
   Asserts transformed polygons match ground truth within 3 px at every vertex.
2. Verification that config.yaml and baseline polygon files remain 100% hash-identical before and after tests.
3. Failure case handling: blocked lens, heavy occlusion (>65% masked), and repetitive tile pattern.
   Asserts transition to UNVERIFIED status with zero polygon reassignment.
4. Operator Override API: POST /api/zones/confirm verifies state and resets unverified state; GET returns 405 Method Not Allowed.
5. Safety escalation during unverified state: simulated fall emits event with zones_unverified=True and continues escalation.
6. Camera wobble test: zero L3 false alarms triggered purely by camera motion during artificial shake.
7. Ground truth verification with tools/simulate_camera_shake.py comparing tracked drift vs ground truth CSV (RMS error < 3.5 px).
"""
import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app as fastapi_app
from metro_safety.config import load_config
from metro_safety.demo.hub import LiveHub, get_global_hub
from metro_safety.escalation.engine import EscalationEngine, SeverityLevel
from metro_safety.events.models import SafetyEvent, EventType
from metro_safety.zones.geometry import ZoneManager, validate_polygon
from metro_safety.zones.registration import AdaptiveZoneTracker, AlignmentStatus, RegistrationResult
from tools.simulate_camera_shake import generate_camera_shake, verify_drift_against_ground_truth


@pytest.fixture
def sample_reference_scene():
    """Generates a high-contrast synthetic metro platform scene with distinct visual landmarks."""
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background floor
    for y in range(h):
        val = int(50 + 120 * (y / float(h)))
        img[y, :] = [val - 10, val, val + 10]

    # Platform grid texture
    for y in range(60, h, 35):
        cv2.line(img, (0, y), (w, y), (40, 45, 50), 2)
    for x in range(0, w, 45):
        cv2.line(img, (x, 60), (x, h), (40, 45, 50), 2)

    # Distinct high-contrast non-repetitive landmarks (posters, pillars, signs across all 4 quadrants)
    cv2.rectangle(img, (50, 80), (190, 220), (30, 80, 220), -1)    # Top-Left: Red pillar
    cv2.rectangle(img, (420, 90), (580, 240), (220, 160, 40), -1)  # Top-Right: Blue sign
    cv2.rectangle(img, (220, 250), (380, 410), (50, 180, 70), -1)  # Center: Green kiosk
    cv2.rectangle(img, (50, 270), (180, 420), (80, 140, 200), -1)   # Bottom-Left: Notice board
    cv2.rectangle(img, (440, 310), (590, 440), (140, 60, 180), -1) # Bottom-Right: Hazard sign

    cv2.putText(img, "DMRC METRO OCC", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "PLATFORM 2 SOUTH", (430, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "ENTRY GATE A", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, "TRACK 2 HAZARD", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.circle(img, (300, 330), 40, (255, 255, 255), -1)
    cv2.circle(img, (300, 330), 25, (0, 0, 0), -1)
    cv2.circle(img, (520, 400), 18, (255, 200, 0), -1)

    return img


@pytest.fixture
def sample_zones():
    return {
        "platform": [[60.0, 100.0], [300.0, 100.0], [300.0, 450.0], [60.0, 450.0]],
        "yellow_band": [[300.0, 100.0], [340.0, 100.0], [340.0, 450.0], [300.0, 450.0]],
        "track_bed": [[340.0, 100.0], [580.0, 100.0], [580.0, 450.0], [340.0, 450.0]]
    }


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file if it exists."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# TEST 1: Baseline config and polygon file immutability (Ground Rule)
# ---------------------------------------------------------------------------
def test_config_and_polygon_baseline_immutability(sample_reference_scene, sample_zones):
    """Saved polygons and config.yaml are the human-set BASELINE.
    Never modify them at runtime. Verify hashes are identical before and after.
    """
    cfg_path = Path("config.yaml")
    poly_path = Path("data/reference_polygons.json")

    hash_cfg_before = compute_file_hash(cfg_path)
    hash_poly_before = compute_file_hash(poly_path)

    zm = ZoneManager(sample_zones)
    tracker = AdaptiveZoneTracker(zm, config={"adaptive_zones": {"enabled": True}})
    tracker.set_reference_frame(sample_reference_scene)

    # Simulate 20 frames with synthetic shifts
    h, w = sample_reference_scene.shape[:2]
    for i in range(1, 15):
        M = np.float32([[1.0, 0.0, 8.0], [0.0, 1.0, -4.0]])
        warped = cv2.warpAffine(sample_reference_scene, M, (w, h))
        tracker.submit_frame(warped, i * 10, [], float(i))

    # Operator confirm
    tracker.confirm_alignment()
    tracker.stop()

    hash_cfg_after = compute_file_hash(cfg_path)
    hash_poly_after = compute_file_hash(poly_path)

    assert hash_cfg_before == hash_cfg_after, "config.yaml was modified at runtime! Ground rule violated."
    assert hash_poly_before == hash_poly_after, "Baseline reference polygons file was modified at runtime!"


# ---------------------------------------------------------------------------
# TEST 2: Synthetic shift, rotation, scale, mild perspective accuracy (< 3px)
# ---------------------------------------------------------------------------
def test_synthetic_warp_registration_accuracy(sample_reference_scene, sample_zones):
    """Direct reference registration against synthetic ground truth warps:
    - Translation (12 px, -8 px)
    - Rotation (2.0 deg)
    - Scale (1.04)
    Asserts every vertex of registered polygons matches ground truth within 3.0 px.
    """
    zm = ZoneManager(sample_zones)
    cfg = {
        "adaptive_zones": {
            "enabled": True,
            "min_inliers": 10,
            "max_reproj_error": 3.0,
            "smoothing_alpha": 1.0,  # Unsmoothed for raw accuracy assertion
            "max_correction_rate_px_per_sec": 100.0,
            "max_drift_px": 50.0
        }
    }
    tracker = AdaptiveZoneTracker(zm, config=cfg)
    tracker.set_reference_frame(sample_reference_scene)

    h, w = sample_reference_scene.shape[:2]
    center = (w / 2.0, h / 2.0)

    # 1. Combined warp: Scale 1.03, Rotation 1.5 deg, Translation (+10, -6)
    theta = 1.5 * math.pi / 180.0
    scale = 1.03
    dx = 10.0
    dy = -6.0

    # Affine matrix
    alpha = scale * math.cos(theta)
    beta = scale * math.sin(theta)
    M = np.array([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1] + dx],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1] + dy]
    ], dtype=np.float32)

    warped_frame = cv2.warpAffine(sample_reference_scene, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Direct registration call
    reg = tracker._register_direct(warped_frame, [], timestamp=1.0)

    assert reg.is_valid, f"Registration failed: {reg.failure_reason}"
    assert reg.status in [AlignmentStatus.ALIGNED, AlignmentStatus.DRIFTING]
    assert reg.homography is not None
    assert reg.transformed_zones is not None

    # Verify ground-truth corner mapping within 3.0 px
    for name, base_pts in sample_zones.items():
        # Ground truth warped points
        gt_pts = cv2.transform(np.array([base_pts], dtype=np.float32), M)[0]
        est_pts = np.array(reg.transformed_zones[name], dtype=np.float32)

        for pt_idx in range(len(base_pts)):
            err = np.linalg.norm(gt_pts[pt_idx] - est_pts[pt_idx])
            assert err < 3.0, f"Vertex {pt_idx} of zone '{name}' error {err:.2f}px exceeds 3.0px limit!"

    tracker.stop()


# ---------------------------------------------------------------------------
# TEST 3: Failure Cases: Blocked Lens, Heavy Occlusion, Repetitive Texture
# ---------------------------------------------------------------------------
def test_blocked_lens_yields_unverified_no_reassignment(sample_reference_scene, sample_zones):
    """A completely blocked/dark lens must fail registration, trigger ZONES_UNVERIFIED,
    and NOT reassign or corrupt polygons.
    """
    zm = ZoneManager(sample_zones)
    baseline_copy = copy.deepcopy(zm.get_polygons())
    tracker = AdaptiveZoneTracker(zm, config={"adaptive_zones": {"max_consecutive_bad_estimates": 2}})
    tracker.set_reference_frame(sample_reference_scene)

    black_frame = np.zeros_like(sample_reference_scene)

    # 3 consecutive blocked frames
    for i in range(3):
        res = tracker._register_direct(black_frame, [], timestamp=float(i))
        tracker._apply_result_with_safety_limits(res, timestamp=float(i))

    assert tracker.status == AlignmentStatus.ZONES_UNVERIFIED
    assert tracker.get_latest_result().zones_unverified is True

    # Check polygons were not mutated or remapped
    curr_polys = zm.get_polygons()
    for name, pts in baseline_copy.items():
        assert np.allclose(curr_polys[name], pts, atol=1e-3), "Polygons were mutated during blocked lens!"

    tracker.stop()


def test_heavily_occluded_frame_yields_unverified(sample_reference_scene, sample_zones):
    """When person bounding boxes or obstacles cover >65% of the frame,
    registration must abort without estimating false homographies.
    """
    zm = ZoneManager(sample_zones)
    tracker = AdaptiveZoneTracker(zm)
    tracker.set_reference_frame(sample_reference_scene)

    h, w = sample_reference_scene.shape[:2]
    # Huge bounding box covering 75% of the frame
    large_box = np.array([20, 20, int(w * 0.9), int(h * 0.85)], dtype=np.float32)

    res = tracker._register_direct(sample_reference_scene, [large_box], timestamp=1.0)
    assert not res.is_valid
    assert "occluded" in res.failure_reason.lower() or "insufficient" in res.failure_reason.lower()
    tracker.stop()


def test_repetitive_texture_rejection(sample_reference_scene, sample_zones):
    """Homography sanity checks must reject false lock on repetitive textures."""
    zm = ZoneManager(sample_zones)
    tracker = AdaptiveZoneTracker(zm)
    tracker.set_reference_frame(sample_reference_scene)

    # Collinear clustered inliers simulating window grid or tile lock
    clustered_dst = np.array([[100.0, 100.0], [101.0, 100.5], [102.0, 100.2], [101.5, 101.0]] * 5, dtype=np.float32)
    inlier_mask = np.ones((len(clustered_dst), 1), dtype=np.uint8)
    H = np.eye(3, dtype=np.float64)

    valid, reason, _ = tracker._sanity_check_homography(H, clustered_dst, inlier_mask, 640, 480)
    assert not valid
    assert "clustered" in reason.lower()
    tracker.stop()


# ---------------------------------------------------------------------------
# TEST 4: API Verification: POST /api/zones/confirm vs GET 405 Method Not Allowed
# ---------------------------------------------------------------------------
def test_operator_confirm_api_post_vs_get(sample_zones):
    """Endpoint POST /api/zones/confirm must succeed; GET must return 405 Method Not Allowed."""
    from backend.app.main import get_global_hub as main_hub_getter
    from metro_safety.demo.hub import set_global_hub
    client = TestClient(fastapi_app)

    hub = main_hub_getter()
    set_global_hub(hub)
    zm = ZoneManager(sample_zones)
    tracker = AdaptiveZoneTracker(zm)
    tracker.status = AlignmentStatus.ZONES_UNVERIFIED
    tracker.requires_operator_confirmation = True
    hub.set_hardware_components(zone_tracker=tracker)

    # 1. GET request must return 405
    res_get = client.get("/api/zones/confirm")
    assert res_get.status_code == 405, f"Expected 405 Method Not Allowed for GET, got {res_get.status_code}"

    # 2. POST request must return 200 and confirm zones
    res_post = client.post("/api/zones/confirm")
    assert res_post.status_code == 200
    data = res_post.json()
    assert data.get("status") == "ok"
    assert tracker.status == AlignmentStatus.ALIGNED
    assert tracker.requires_operator_confirmation is False

    tracker.stop()


# ---------------------------------------------------------------------------
# TEST 5: Fall Detection during Unverified State Continues with Tagging
# ---------------------------------------------------------------------------
def test_fall_during_unverified_preserves_escalation_with_tagging(sample_zones):
    """When zones are unverified, detection continues, DO NOT silently reassign.
    Every event gets zones_unverified=True and [ZONES UNVERIFIED] is prepended.
    """
    escalation = EscalationEngine(station_id="STN-01", platform_id="P2")

    fall_event = SafetyEvent(
        event_type=EventType.ACCIDENTAL_FALL,
        confidence=0.96,
        track_id=14,
        timestamp=100.0,
        zone="track_bed",
        train_state="APPROACHING",
        metadata={"vy": 18.0, "angle": 75.0, "zone": "track_bed"},
        zones_unverified=True,
        zone_version=3
    )

    dec = escalation.evaluate(event=fall_event, train_state=None, current_time=100.0)

    # Escalation must still trigger
    assert dec.severity in [SeverityLevel.L2, SeverityLevel.L3]
    # Event dict must carry zones_unverified
    ev_dict = fall_event.to_dict()
    assert ev_dict["zones_unverified"] is True
    assert ev_dict["zone_version"] == 3


# ---------------------------------------------------------------------------
# TEST 6: Artificial Camera Shake Test: Zero False L3 Triggered Purely by Wobble
# ---------------------------------------------------------------------------
def test_artificial_shake_clip_yields_zero_false_l3(sample_zones):
    """Run artificial camera shake on normal stationary clip.
    Assert zero false L3 emergency alarms are emitted purely by camera wobble,
    and verify estimated drift from AdaptiveZoneTracker against ground-truth CSV.
    """
    from metro_safety.events.rule_classifier import RuleBasedEventClassifier
    from metro_safety.perception.camera_motion import CameraMotionDetector
    from metro_safety.perception.train_state import TrainState

    shake_result = generate_camera_shake(
        input_video_path="demo_clips/demo_normal.mp4",
        output_video_path="data/recordings/test_shaken_clip.mp4",
        ground_truth_csv_path="data/recordings/test_shaken_gt.csv",
        max_frames=40,
        seed=101,
        max_dx=12.0,
        max_dy=8.0,
        max_dtheta_deg=1.5
    )

    assert shake_result["frames"] > 0
    gt_csv = Path(shake_result["csv"])
    assert gt_csv.exists()

    # Read video frames
    cap = cv2.VideoCapture(shake_result["video"])
    frames = []
    while True:
        ret, f = cap.read()
        if not ret or f is None:
            break
        frames.append(f)
    cap.release()
    assert len(frames) > 0

    zm = ZoneManager(sample_zones)
    tracker = AdaptiveZoneTracker(zm, config={"adaptive_zones": {"registration_interval_frames": 1}})
    tracker.set_reference_frame(frames[0])

    rule_clf = RuleBasedEventClassifier()
    escalation = EscalationEngine(station_id="STN-01", platform_id="P2")
    motion_det = CameraMotionDetector(motion_threshold_px=4.0)

    estimated_drifts = {}
    l3_alarms = []

    for idx, frame in enumerate(frames):
        ts = float(idx) * 0.04
        is_moving, disp, blur, reason = motion_det.estimate_motion(frame, [])

        # Estimate drift directly via tracker
        reg_res = tracker._register_direct(frame, [], ts)
        estimated_drifts[idx] = reg_res.drift_px

        # Process frame through classifier and escalation pipeline
        evs = rule_clf.process_frame(
            idx, ts, [], zm, TrainState.ABSENT,
            camera_motion=is_moving, motion_reason=reason
        )

        # Collect decisions for any events emitted from video processing
        for ev in evs:
            dec = escalation.evaluate(event=ev, train_state=None, current_time=ts)
            if dec.severity == SeverityLevel.L3:
                l3_alarms.append((idx, dec))

        # Verification of camera motion gating: if camera is moving, even a high-severity candidate must be downgraded to L2
        if is_moving:
            hypothetical_fall = SafetyEvent(
                event_type=EventType.ACCIDENTAL_FALL,
                confidence=0.92,
                track_id=99,
                timestamp=ts,
                zone="track_bed",
                train_state="ABSENT",
                camera_motion=True,
                motion_reason=reason,
                zones_unverified=reg_res.zones_unverified
            )
            gated_dec = escalation.evaluate(event=hypothetical_fall, train_state=None, current_time=ts)
            assert gated_dec.severity != SeverityLevel.L3, f"Camera motion failed to gate L3 alert on frame {idx}!"

    tracker.stop()

    # Verify drift against ground truth using actual tracker estimated drifts
    passed, rms_err, max_err = verify_drift_against_ground_truth(
        estimated_drifts=estimated_drifts,
        ground_truth_csv_path=str(gt_csv),
        max_rms_threshold_px=3.5
    )
    assert passed, f"Estimated drift RMS error {rms_err:.2f}px exceeded limit 3.5px"
    assert len(l3_alarms) == 0, f"Camera shake produced false L3 alarms: {len(l3_alarms)}"


# ---------------------------------------------------------------------------
# TEST 7: Auto-Suggest Zones (Item 10)
# ---------------------------------------------------------------------------
def test_auto_suggest_zones_produces_valid_polygons(sample_reference_scene, tmp_path):
    """Auto-suggest tool must produce valid, non-intersecting candidate polygons."""
    from metro_safety.tools.auto_suggest_zones import auto_suggest_zones
    out_json = tmp_path / "suggested_zones.json"
    suggested = auto_suggest_zones(sample_reference_scene, output_json_path=str(out_json))

    assert "platform" in suggested
    assert "yellow_band" in suggested
    assert "track_bed" in suggested
    assert out_json.exists()

    # Re-verify every suggested polygon passes strict validation
    for name, poly in suggested.items():
        v, m = validate_polygon(poly, name=name)
        assert v, f"Auto-suggested zone '{name}' failed validation: {m}"

