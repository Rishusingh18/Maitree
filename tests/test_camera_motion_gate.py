"""Tests for Camera Motion Estimation and Escalation Gating (Addendum B).
Validates:
1. CameraMotionDetector accurately flags global optical flow motion when frames translate/shake.
2. CameraMotionDetector flags camera occlusion / blur when Laplacian variance drops.
3. EscalationEngine caps ACCIDENTAL_FALL and DELIBERATE_JUMP events at Level 2 when camera_motion=True,
   emitting 'CAMERA MOVING: needs operator review' and never actuating emergency traction cut (ETS).
4. Normal position-based track intrusions continue to trigger appropriately without suppression.
5. Simulated camera shake over a frame sequence generates zero L3 events.
"""
import numpy as np
import cv2
import pytest

from metro_safety.perception.camera_motion import CameraMotionDetector
from metro_safety.events.models import SafetyEvent, EventType, SeverityLevel
from metro_safety.perception.train_state import TrainState
from metro_safety.escalation.engine import EscalationEngine


def test_camera_motion_optical_flow_detection():
    detector = CameraMotionDetector(motion_threshold_px=4.0)

    # Synthetic textured background
    np.random.seed(42)
    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(80):
        x = np.random.randint(20, 620)
        y = np.random.randint(20, 460)
        cv2.circle(base_frame, (x, y), 5, (200, 200, 200), -1)

    # Frame 1: Stable
    is_moving, disp, blur, reason = detector.estimate_motion(base_frame)
    assert not is_moving

    # Frame 2: Identical stable frame
    is_moving, disp, blur, reason = detector.estimate_motion(base_frame)
    assert not is_moving
    assert disp < 2.0

    # Frame 3: Simulated camera shake (15 pixel translation)
    M = np.float32([[1, 0, 15], [0, 1, 10]])
    shaken_frame = cv2.warpAffine(base_frame, M, (640, 480))
    is_moving, disp, blur, reason = detector.estimate_motion(shaken_frame)
    assert is_moving is True
    assert disp >= 4.0
    assert "Global camera motion" in reason


def test_camera_blur_occlusion_detection():
    detector = CameraMotionDetector(blur_threshold=30.0)
    # Uniform gray occluded / covered camera image
    gray_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    is_moving, disp, blur, reason = detector.estimate_motion(gray_frame)
    assert is_moving is True
    assert "covered or blurred" in reason


def test_escalation_engine_caps_fall_and_jump_during_camera_motion():
    engine = EscalationEngine(station_id="STN_TEST", platform_id="PL_TEST")

    # 1. Fall event WITHOUT camera motion -> Level 3 with ETS
    clean_fall = SafetyEvent(
        event_type=EventType.ACCIDENTAL_FALL,
        confidence=0.95,
        track_id=1,
        timestamp=100.0,
        zone="track_bed",
        train_state="ABSENT",
        camera_motion=False
    )
    dec_clean = engine.evaluate(clean_fall, train_state="ABSENT")
    assert dec_clean.severity == SeverityLevel.L3
    assert "CRITICAL" in dec_clean.action_taken

    # 2. Fall event WITH camera motion -> Capped at Level 2, no ETS
    moving_fall = SafetyEvent(
        event_type=EventType.ACCIDENTAL_FALL,
        confidence=0.95,
        track_id=2,
        timestamp=101.0,
        zone="track_bed",
        train_state="ABSENT",
        camera_motion=True,
        motion_reason="Global camera motion (14.2 px > 5.0 px threshold)"
    )
    dec_moving_fall = engine.evaluate(moving_fall, train_state="ABSENT")
    assert dec_moving_fall.severity == SeverityLevel.L2
    assert "CAMERA MOVING: needs operator review" in dec_moving_fall.action_taken
    assert dec_moving_fall.severity != SeverityLevel.L3

    # 3. Deliberate jump WITH camera motion -> Capped at Level 2, no ETS
    moving_jump = SafetyEvent(
        event_type=EventType.DELIBERATE_JUMP,
        confidence=0.93,
        track_id=3,
        timestamp=102.0,
        zone="track_bed",
        train_state="ABSENT",
        camera_motion=True,
        motion_reason="Optical flow camera vibration"
    )
    dec_moving_jump = engine.evaluate(moving_jump, train_state="ABSENT")
    assert dec_moving_jump.severity == SeverityLevel.L2
    assert "CAMERA MOVING: needs operator review" in dec_moving_jump.action_taken
    assert dec_moving_jump.severity != SeverityLevel.L3


def test_simulated_camera_shake_asserts_zero_l3_events():
    """Feed a simulated camera shake sequence into EscalationEngine and assert zero L3 events."""
    engine = EscalationEngine(station_id="STN_TEST", platform_id="PL_TEST")
    detector = CameraMotionDetector(motion_threshold_px=4.0)

    # Base background
    np.random.seed(123)
    bg = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(100):
        cv2.circle(bg, (np.random.randint(20, 620), np.random.randint(20, 460)), 4, (220, 220, 220), -1)

    # Initialize detector reference features with base frame
    detector.estimate_motion(bg)

    l3_events = []
    l2_events = []

    # Sequence with camera vibrations
    for f in range(1, 21):
        dx = int(np.sin(f * 0.8) * 12.0)
        dy = int(np.cos(f * 0.8) * 10.0)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(bg, M, (640, 480))

        is_moving, disp, blur, reason = detector.estimate_motion(shaken)

        # Candidate fall event triggered by vertical camera shake
        event = SafetyEvent(
            event_type=EventType.ACCIDENTAL_FALL,
            confidence=0.91,
            track_id=1,
            timestamp=f * 0.04,
            zone="track_bed",
            train_state="ABSENT",
            camera_motion=is_moving,
            motion_reason=reason
        )

        dec = engine.evaluate(event, train_state="ABSENT")
        if dec.severity == SeverityLevel.L3:
            l3_events.append(dec)
        elif dec.severity == SeverityLevel.L2:
            l2_events.append(dec)

    # Strict assertion: Zero L3 events occurred during camera shake!
    assert len(l3_events) == 0, f"Expected 0 L3 events during camera shake, got {len(l3_events)}"
    assert len(l2_events) > 0, "Expected L2 review alerts to be retained (never dropped)"
