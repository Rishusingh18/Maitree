"""One-Command Live Mode: AI Metro Platform Safety System.
Starts in a single unified process:
1. Live Computer Vision Pipeline (reading phone CCTV stream at config.yaml source).
2. Central Control Room Flask Web Dashboard on http://localhost:5000 with real-time video feed (/video_feed).
3. Real Hardware Buzzer on host PC (winsound on Windows with cross-platform fallback).
4. Overlooker Alert Engine (desktop toast notifications, optional ntfy/Telegram push, cooldown, audit log).
5. Live-measured telemetry (FPS, latency, synthetic accuracy).

Usage:
  python -m metro_safety.demo.live
  python -m metro_safety.demo.live --source http://192.168.1.8:8080/video --port 5000
  python -m metro_safety.demo.live --source demo_clips/demo_jump.mp4 --loop
"""
import os
import sys
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
from werkzeug.serving import make_server

from metro_safety.config import load_config
from metro_safety.perception.frame_source import create_frame_source, FrameSource
from metro_safety.perception.base import Detection
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.perception.camera_motion import CameraMotionDetector
from metro_safety.perception.async_inference import AsyncInferenceWorker
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.zones.geometry import ZoneManager
from metro_safety.zones.registration import AdaptiveZoneTracker, AlignmentStatus
from metro_safety.perception.train_state import TrainState, SimulatedTrainStateProvider
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.learned_classifier import LearnedEventClassifier
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.escalation.engine import EscalationEngine, SeverityLevel
from metro_safety.escalation.buzzer import HardwareBuzzer
from metro_safety.escalation.overlooker import OverlookerNotifier
from metro_safety.demo.hub import LiveHub, get_global_hub, set_global_hub
from metro_safety.demo.dashboard import create_dashboard_app

logger = logging.getLogger("metro_safety.demo.live")

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


class DashboardServerThread(threading.Thread):
    """Runs the Flask Dashboard in a non-blocking daemon thread."""
    def __init__(self, app, host: str = "0.0.0.0", port: int = 5000):
        super().__init__(daemon=True, name="FlaskDashboardServer")
        self.host = host
        self.port = port
        self.server = make_server(host, port, app, threaded=True)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        logger.info(f"Dashboard server listening on http://{self.host}:{self.port}")
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def run_live_pipeline(
    source_uri: Optional[str] = None,
    port: int = 5000,
    config_path: Optional[str] = None,
    show_window: bool = True,
    loop_video: bool = False,
    max_frames: Optional[int] = None
):
    cfg = load_config(config_path)

    # 1. Determine active camera source (CLI flag > METRO_SOURCE_OVERRIDE > config.yaml)
    source_override = os.environ.get("METRO_SOURCE_OVERRIDE")
    if source_uri is not None:
        resolved_source = source_uri
    elif source_override:
        resolved_source = source_override
        logger.info(f"[Live] Using METRO_SOURCE_OVERRIDE: {resolved_source}")
    else:
        resolved_source = cfg["system"].get("source", "http://192.168.1.32:8080/video")

    print("\n" + "=" * 65)
    print("  🚇 DMRC AI METRO PLATFORM SAFETY SYSTEM - LIVE MODE")
    print(f"  Camera Source:     {resolved_source}")
    print(f"  Web Dashboard:     http://localhost:{port}")
    print(f"  Station ID:        {cfg['system']['station_id']} | Platform: {cfg['system']['platform_id']}")
    print(f"  Hardware Buzzer:   {'ENABLED' if cfg.get('buzzer', {}).get('audio_enabled', True) else 'DISABLED'}")
    print(f"  Overlooker Alerts: Desktop Toast + Dashboard Banner")
    print("=" * 65 + "\n")

    # 2. Initialize Shared Hub & Hardware Adapters
    hub = get_global_hub()
    hub.cctv_id = cfg["system"]["camera_id"]
    hub.station_id = cfg["system"]["station_id"]
    hub.platform_id = cfg["system"]["platform_id"]
    set_global_hub(hub)

    buzzer_cfg = cfg.get("buzzer", {})
    buzzer = HardwareBuzzer(
        audio_enabled=buzzer_cfg.get("audio_enabled", True),
        clear_timeout_sec=buzzer_cfg.get("clear_timeout_sec", 4.0),
        l1_freq=buzzer_cfg.get("l1_frequency", 800),
        l1_dur_ms=buzzer_cfg.get("l1_duration_ms", 150),
        l2_freq=buzzer_cfg.get("l2_frequency", 1200),
        l2_beep_ms=buzzer_cfg.get("l2_beep_ms", 200),
        l2_pause_ms=buzzer_cfg.get("l2_pause_ms", 100),
        l3_low_freq=buzzer_cfg.get("l3_low_freq", 1800),
        l3_high_freq=buzzer_cfg.get("l3_high_freq", 2200)
    )

    train_provider = SimulatedTrainStateProvider(initial_state=TrainState(cfg["train_state"]["default_state"]))
    hub.set_hardware_components(buzzer=buzzer, train_state_provider=train_provider)

    overlooker_cfg = cfg.get("overlooker", {})
    overlooker = OverlookerNotifier(
        cooldown_sec=overlooker_cfg.get("cooldown_sec", 10.0),
        desktop_notification=overlooker_cfg.get("desktop_notification", True),
        audit_log_file=overlooker_cfg.get("audit_log_file", "logs/overlooker_audit.log"),
        ntfy_config=overlooker_cfg.get("ntfy", {}),
        telegram_config=overlooker_cfg.get("telegram", {}),
        dashboard_hub=hub
    )

    # 3. Start Dashboard Server Thread
    flask_app = create_dashboard_app(hub)
    server_thread = DashboardServerThread(flask_app, host="0.0.0.0", port=port)
    server_thread.start()
    print(f"[Live] Dashboard running at: http://localhost:{port}")

    # 4. Initialize Perception & Safety Modules
    zm = ZoneManager(cfg["zones"])
    zone_tracker = AdaptiveZoneTracker(
        zone_manager=zm,
        config=cfg,
        reference_image_path=cfg.get("reference_frame")
    )
    hub.set_hardware_components(buzzer=buzzer, train_state_provider=train_provider, zone_tracker=zone_tracker)
    tracker = ByteTracker(
        track_high_thresh=cfg["tracking"].get("track_high_thresh", 0.22),
        track_low_thresh=cfg["tracking"].get("track_low_thresh", 0.12),
        new_track_thresh=cfg["tracking"].get("new_track_thresh", 0.18),
        track_buffer=cfg["tracking"].get("track_buffer", 30),
        match_thresh=cfg["tracking"].get("match_thresh", 0.6)
    )
    rule_clf = RuleBasedEventClassifier(
        line_crossing_thresh=cfg["events"]["line_crossing_consecutive_frames"],
        track_intrusion_thresh=cfg["events"]["track_intrusion_consecutive_frames"],
        fall_drop_vel_thresh=cfg["events"].get("fall_drop_velocity_threshold", 14.0),
        fall_angle_thresh=cfg["events"].get("fall_angle_threshold", 40.0),
        jump_vel_thresh=cfg["events"].get("jump_vertical_velocity_threshold", 12.0),
        min_event_frames=cfg["events"].get("min_motion_duration_frames", 4),
        min_drop_displacement=cfg["events"].get("fall_min_displacement_px", 10.0),
        min_jump_displacement=cfg["events"].get("jump_min_displacement_px", 8.0)
    )
    learned_clf = LearnedEventClassifier()
    gesture_det = DistressGestureDetector(
        min_persistence=cfg["gesture"]["min_persistence_frames"],
        use_macro_fallback=cfg["gesture"]["use_macro_fallback"]
    )
    escalation = EscalationEngine(
        station_id=cfg["system"]["station_id"],
        platform_id=cfg["system"]["platform_id"]
    )
    from metro_safety.anomaly.engine import AnomalyEngine
    anomaly_engine = AnomalyEngine(cfg)

    # YOLO pose model with CPU performance tuning
    inference_size = int(cfg["perception"].get("inference_size", 480))
    frame_skip = max(1, int(cfg["perception"].get("frame_skip", 1)))
    pose_detector = YOLOPoseDetector(
        model_name=cfg["perception"]["model_name"],
        confidence_threshold=cfg["perception"]["confidence_threshold"],
        device=cfg["perception"]["device"],
        inference_size=inference_size
    )
    if hasattr(pose_detector, "_model") and pose_detector._model is not None:
        anomaly_engine.set_object_detector(pose_detector._model)

    # Decoupled async inference worker running DNN on dedicated thread (enabling 25+ FPS display rate)
    async_inference = AsyncInferenceWorker(detector=pose_detector, frame_skip=frame_skip)
    async_inference.start()

    # Camera motion and blur detector (sparse optical flow on background features)
    cam_motion_cfg = cfg.get("perception", {})
    motion_detector = CameraMotionDetector(
        motion_threshold_px=cam_motion_cfg.get("camera_motion_threshold_px", 5.0),
        blur_threshold=cam_motion_cfg.get("camera_blur_threshold", 30.0)
    )

    # 5. Initialize FrameSource (threaded, buffer size 1, drop stale frames, auto-reconnect backoff)
    frame_source: FrameSource = create_frame_source(
        source_uri=resolved_source,
        loop_video=loop_video,
        fallback_path="demo_clips/demo_fall.mp4",
        allow_fallback=True
    )

    win_name = f"DMRC CCTV Safety System - [{cfg['system']['camera_id']}] (Press 'q' to stop)"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    import collections
    perf_capture = collections.deque(maxlen=100)
    perf_detect = collections.deque(maxlen=100)
    perf_track = collections.deque(maxlen=100)
    perf_motion = collections.deque(maxlen=100)
    perf_draw = collections.deque(maxlen=100)
    perf_encode = collections.deque(maxlen=100)
    perf_e2e = collections.deque(maxlen=100)

    last_cam_moving = False
    last_cam_disp = 0.0
    last_cam_blur = 100.0
    last_cam_reason = ""

    frame_idx = 0
    fps_history = []
    latency_history = []
    curr_fps = 0.0
    curr_lat = 0.0
    t_prev = time.perf_counter()
    active_alert_banner = None
    banner_expire_time = 0.0
    last_event_time = {}
    event_cooldown_sec = 5.0
    last_zone_event_time = {}
    zone_cooldown_sec = float(cfg.get("adaptive_zones", {}).get("unverified_zone_cooldown_sec", 4.0))
    cached_detections: List[Detection] = []

    print("[Live] Video pipeline active. Monitoring stream...")

    try:
        while True:
            t_start = time.perf_counter()

            try:
                # 1. Capture Stage
                t0_cap = time.perf_counter()
                ret, frame, frame_ts = frame_source.read()
                perf_capture.append((time.perf_counter() - t0_cap) * 1000.0)

                if not ret or frame is None:
                    hub.update_frame(None, is_online=False, status_msg=frame_source.source_name)
                    time.sleep(0.04)
                    continue

                frame_idx += 1
                h, w = frame.shape[:2]
                current_train_state = train_provider.get_state()

                # 2. Perception / Detection Stage (Decoupled background inference)
                t0_det = time.perf_counter()
                async_inference.submit_frame(frame, frame_idx)
                detections = async_inference.get_latest_detections()

                # Fallback for synthetic demo video clips
                if not detections:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25, 255, 255]))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pts = []
                    for cnt in contours:
                        if 2 <= cv2.contourArea(cnt) <= 120:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                pts.append([M["m10"] / M["m00"], M["m01"] / M["m00"], 0.9])
                    if len(pts) >= 6:
                        pts_arr = np.array(pts)
                        x1, y1 = np.min(pts_arr[:, 0]) - 15, np.min(pts_arr[:, 1]) - 20
                        x2, y2 = np.max(pts_arr[:, 0]) + 15, np.max(pts_arr[:, 1]) + 15
                        box = np.array([x1, y1, x2, y2], dtype=np.float32)
                        kpts = np.zeros((17, 3), dtype=np.float32)
                        for ki in range(17):
                            kpts[ki] = [np.mean(pts_arr[:, 0]), np.mean(pts_arr[:, 1]), 0.8]
                        if len(pts) >= 15:
                            sorted_by_y = pts_arr[np.argsort(pts_arr[:, 1])]
                            kpts[0] = sorted_by_y[0]
                            kpts[15] = sorted_by_y[-2]
                            kpts[16] = sorted_by_y[-1]
                        detections = [Detection(box=box, confidence=0.92, keypoints=kpts)]
                perf_detect.append((time.perf_counter() - t0_det) * 1000.0)

                # 3. Tracking Stage
                t0_trk = time.perf_counter()
                tracks = tracker.update(detections)
                perf_track.append((time.perf_counter() - t0_trk) * 1000.0)

                # 4. Motion Estimation Stage (safe speedup: evaluate every 2 frames, reuse state)
                t0_mot = time.perf_counter()
                if frame_idx % 2 == 0 or frame_idx <= 2:
                    person_boxes = [trk.box for trk in tracks] if tracks else [d.box for d in detections]
                    is_camera_moving, cam_disp, cam_blur, cam_reason = motion_detector.estimate_motion(frame, person_boxes)
                    last_cam_moving = is_camera_moving
                    last_cam_disp = cam_disp
                    last_cam_blur = cam_blur
                    last_cam_reason = cam_reason
                else:
                    is_camera_moving = last_cam_moving
                    cam_disp = last_cam_disp
                    cam_blur = last_cam_blur
                    cam_reason = last_cam_reason
                perf_motion.append((time.perf_counter() - t0_mot) * 1000.0)

                # 3. Behavior Classification & Anomaly Detection
                rule_events = rule_clf.process_frame(frame_idx, frame_ts, tracks, zm, current_train_state)
                learned_events = learned_clf.process_frame(frame_idx, frame_ts, tracks, zm, current_train_state)
                gesture_events = gesture_det.process_frame(frame_idx, frame_ts, tracks, zm, current_train_state)
                anomaly_events = anomaly_engine.process_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    person_detections=tracks,
                    zone_manager=zm,
                    timestamp=frame_ts,
                    train_state=current_train_state.value if hasattr(current_train_state, "value") else str(current_train_state),
                    camera_motion=is_camera_moving,
                    motion_reason=cam_reason,
                    zones_unverified=False,
                    zone_version=zm.zone_version
                )
                all_events = rule_events + learned_events + gesture_events + anomaly_events

                # Tag events with camera motion status for escalation gating
                for ev in all_events:
                    ev.camera_motion = is_camera_moving
                    ev.motion_reason = cam_reason

                # Check active intrusion in danger zones for buzzer clear-timeout
                has_intrusion = False
                for trk in tracks:
                    zone = zm.get_zone_for_point(trk.foot_anchor)
                    if zone in ["yellow_band", "track_bed", "danger_zone"]:
                        has_intrusion = True
                        break
                try:
                    buzzer.update_clear_status(has_intrusion)
                except Exception as b_err:
                    logger.exception(f"[Live] Error updating buzzer clear status: {b_err}")

                # 4. Closed-Loop Escalation, Buzzer & Overlooker Notifications (Per-event isolation)
                is_fallback = getattr(frame_source, "is_fallback", False)
                primary_tgt = getattr(frame_source, "primary_target", resolved_source)
                retry_stat = getattr(frame_source, "retry_status", "Active")
                is_demo = is_fallback or ("demo" in str(resolved_source).lower())

                if frame_idx == 1:
                    zone_tracker.on_first_frame(frame)
                zone_tracker.set_demo_mode(is_demo)

                cmd_path = Path("data/cache/zone_confirm.cmd")
                if cmd_path.exists():
                    try:
                        cmd_path.unlink()
                        c_ok, c_msg = zone_tracker.confirm_alignment()
                        resp_path = Path("data/cache/zone_confirm.resp")
                        resp_path.write_text(json.dumps({"status": "ok" if c_ok else "error", "message": c_msg}), encoding="utf-8")
                    except Exception:
                        pass

                person_boxes = [trk.box for trk in tracks] if tracks else [d.box for d in detections]
                zone_tracker.submit_frame(frame.copy(), frame_idx, person_boxes, frame_ts)
                reg_result = zone_tracker.get_latest_result()
                frame_zone_version = zm.zone_version
                frame_zones_unverified = reg_result.zones_unverified

                for ev in all_events:
                    ev.zones_unverified = frame_zones_unverified
                    ev.zone_version = frame_zone_version

                for ev in all_events:
                    try:
                        dec = escalation.evaluate(
                            event=ev,
                            train_state=current_train_state,
                            current_time=frame_ts,
                            snapshot=frame
                        )

                        # Event Deduplication: Cooldown per (track_id, event_type)
                        dedup_key = (ev.track_id, ev.event_type.name)
                        if frame_ts - last_event_time.get(dedup_key, 0.0) < event_cooldown_sec:
                            continue

                        # Dual cooldown during UNVERIFIED state to prevent alert storms
                        if frame_zones_unverified:
                            z_key = (ev.zone, ev.event_type.name)
                            if z_key in last_zone_event_time and (frame_ts - last_zone_event_time[z_key]) < zone_cooldown_sec:
                                continue
                            last_zone_event_time[z_key] = frame_ts

                        # Update last_event_time ONLY after both cooldown guards accept the event
                        last_event_time[dedup_key] = frame_ts

                        sev = dec.severity.name
                        kine = rule_clf.get_track_kinematics(ev.track_id)

                        ev_type_label = f"[DEMO] {ev.event_type.name}" if is_demo else ev.event_type.name
                        log_tag = "[Live Event - DEMO CLIP]" if is_demo else "[Live Event]"

                        # Per-event debug logging (track id, zone, vertical velocity, torso angle, confidence)
                        logger.info(
                            f"{log_tag} Track #{ev.track_id:02d} | Event: {ev_type_label} | "
                            f"Zone: {ev.zone} | vy: {kine['vy']:+05.1f} px/f | angle: {kine['angle']:04.1f} deg | "
                            f"Conf: {ev.confidence:.2f} | Severity: {sev}"
                        )

                        # Trigger Hardware Buzzer (Distress gesture is silent!)
                        if sev != "DISTRESS":
                            try:
                                buzzer.trigger(sev)
                            except Exception as b_trig_err:
                                logger.exception(f"[Live] Error triggering hardware buzzer: {b_trig_err}")

                        # Trigger Overlooker (L2, L3, DISTRESS)
                        try:
                            overlooker.notify(
                                event_type=ev_type_label,
                                severity=sev,
                                track_id=ev.track_id,
                                zone=ev.zone,
                                confidence=ev.confidence,
                                timestamp=frame_ts,
                                train_state=current_train_state.value
                            )
                        except Exception as ov_err:
                            logger.exception(f"[Live] Error in overlooker notification: {ov_err}")

                        # Set on-screen banner
                        active_alert_banner = (sev, f"{ev_type_label} - {dec.action_taken}")
                        banner_expire_time = time.time() + 3.0

                        # Push event to LiveHub for dashboard update
                        try:
                            hub.push_event({
                                "id": f"EV-{frame_idx}-{ev.track_id}",
                                "timestamp": frame_ts,
                                "time_str": time.strftime("%H:%M:%S", time.localtime()),
                                "track_id": ev.track_id,
                                "event_type": ev_type_label,
                                "zone": ev.zone,
                                "severity": sev,
                                "confidence": round(ev.confidence, 2),
                                "action": dec.action_taken,
                                "kinematics": kine,
                                "is_demo": is_demo,
                                "is_anomaly": bool(getattr(ev, "metadata", {}).get("is_anomaly", False)),
                                "anomaly_type": getattr(ev, "metadata", {}).get("anomaly_type", ""),
                                "snapshot_crop": getattr(ev, "metadata", {}).get("snapshot_crop", None)
                            })
                        except Exception as hub_ev_err:
                            logger.exception(f"[Live] Error pushing event to LiveHub: {hub_ev_err}")

                    except Exception as ev_err:
                        logger.exception(f"[Live] Error escalating event {ev.event_type} on frame {frame_idx}: {ev_err}")
                        continue

                # 5. Annotation & Visual Rendering Stage
                t0_draw = time.perf_counter()
                # Draw defined platform zones with adaptive telemetry
                status_badge_text = f"ALIGN: {reg_result.status.value} | DRIFT: {reg_result.drift_px:.1f}px | INL: {reg_result.inlier_count} | V:{reg_result.zone_version}"
                annotated = zm.draw_zones(
                    frame.copy(),
                    alpha=0.25,
                    unverified=reg_result.zones_unverified,
                    draw_baseline=True,
                    drift_px=reg_result.drift_px,
                    status_badge=status_badge_text
                )

                # Draw tracks, pose skeletons, and debug telemetry
                for trk in tracks:
                    box = trk.box.astype(int)
                    tid = trk.track_id
                    kine = rule_clf.get_track_kinematics(tid)

                    # Box
                    box_color = (0, 255, 0)
                    if kine["zone"] == "yellow_band":
                        box_color = (0, 230, 255)
                    elif kine["zone"] in ["track_bed", "danger_zone"]:
                        box_color = (0, 0, 255)

                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), box_color, 2)

                    # Foot anchor dot
                    foot = trk.foot_anchor
                    cv2.circle(annotated, (int(foot[0]), int(foot[1])), 6, (0, 0, 255), -1)

                    # Skeletons
                    if trk.keypoints is not None:
                        kpts = trk.keypoints
                        for pt in kpts:
                            if pt[0] > 0 and pt[1] > 0:
                                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                        for p1, p2 in COCO_PAIRS:
                            if p1 < len(kpts) and p2 < len(kpts):
                                pt1, pt2 = kpts[p1], kpts[p2]
                                if pt1[0] > 0 and pt2[0] > 0:
                                    cv2.line(annotated, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 120, 0), 1)

                    # Kinematic HUD overlay badge above person
                    overlay_text = f"ID #{tid:02d} | {kine['zone']} | vy:{kine['vy']:+04.1f} | angle:{kine['angle']:04.1f}deg"
                    (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    bx = max(10, box[0])
                    by = max(25, box[1] - 8)
                    cv2.rectangle(annotated, (bx - 2, by - th - 4), (bx + tw + 4, by + 4), (20, 20, 20), -1)
                    cv2.putText(annotated, overlay_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Top Live Telemetry HUD Bar
                cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 20), -1)
                buzzer_st = buzzer.get_status()
                buzzer_str = "MUTED" if buzzer_st["muted"] else buzzer_st["pattern"]
                det_fps, det_lat = async_inference.get_telemetry()
                if det_fps <= 0.0:
                    det_fps = curr_fps

                hud_txt = (
                    f"LIVE FPS: {curr_fps:04.1f} | DETECTOR FPS: {det_fps:04.1f} | LATENCY: {curr_lat:04.1f} ms | "
                    f"TRAIN: {current_train_state.value} | BUZZER: {buzzer_str}"
                )
                cv2.putText(annotated, hud_txt, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                # High-visibility fallback banner (amber/orange bar)
                if is_fallback:
                    cv2.rectangle(annotated, (0, 40), (w, 82), (20, 120, 245), -1)
                    cv2.putText(
                        annotated,
                        f"SOURCE: DEMO CLIP (phone stream unreachable) | Attempted: {primary_tgt}",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        annotated,
                        f"Status: {retry_stat}",
                        (15, 76),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (230, 240, 255),
                        1
                    )

                # Camera motion warning banner (prominent orange banner)
                if is_camera_moving:
                    cm_top = 84 if is_fallback else 42
                    cm_bot = cm_top + 34
                    cv2.rectangle(annotated, (0, cm_top), (w, cm_bot), (0, 140, 255), -1)
                    cv2.putText(
                        annotated,
                        f"CAMERA MOVING: needs operator review | {cam_reason}",
                        (15, cm_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        2
                    )

                # Bottom Escalation Banner
                if active_alert_banner and time.time() < banner_expire_time:
                    sev, banner_msg = active_alert_banner
                    banner_col = (0, 0, 220) if sev == "L3" else ((0, 140, 255) if sev == "L2" else (180, 50, 180))
                    cv2.rectangle(annotated, (0, h - 45), (w, h), banner_col, -1)
                    cv2.putText(annotated, f"[{sev}] {banner_msg}", (20, h - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                perf_draw.append((time.perf_counter() - t0_draw) * 1000.0)

                # 6. Encode Stage (publish latest frame to LiveHub for Web Dashboard /video_feed)
                t0_enc = time.perf_counter()
                try:
                    hub.update_frame(annotated, is_online=True)
                except Exception as pub_err:
                    logger.exception(f"[Live] Error updating dashboard frame: {pub_err}")
                perf_encode.append((time.perf_counter() - t0_enc) * 1000.0)

                # End-to-end latency from capture timestamp to JPEG publish
                t_frame_end = time.perf_counter()
                e2e_ms = (t_frame_end - t0_cap) * 1000.0
                perf_e2e.append(e2e_ms)
                latency_history.append(e2e_ms)
                if len(latency_history) > 30:
                    latency_history.pop(0)

                dt = t_frame_end - t_prev
                t_prev = t_frame_end
                if dt > 0:
                    fps_history.append(1.0 / dt)
                    if len(fps_history) > 30:
                        fps_history.pop(0)

                curr_fps = float(np.mean(fps_history)) if fps_history else 0.0
                curr_lat = float(np.median(perf_e2e)) if perf_e2e else 0.0
                e2e_p95 = float(np.percentile(perf_e2e, 95)) if perf_e2e else 0.0

                fwd_timings = async_inference.get_forward_timings()
                det_fps, det_lat = async_inference.get_telemetry()
                if det_fps <= 0.0:
                    det_fps = round(1000.0 / fwd_timings["median_ms"], 1) if fwd_timings["median_ms"] > 0 else curr_fps

                stage_timings = {
                    "capture": {"median_ms": round(float(np.median(perf_capture)), 2), "p95_ms": round(float(np.percentile(perf_capture, 95)), 2)},
                    "detect_forward_pass": fwd_timings if fwd_timings["median_ms"] > 0 else {"median_ms": round(float(np.median(perf_detect)), 2), "p95_ms": round(float(np.percentile(perf_detect, 95)), 2)},
                    "track": {"median_ms": round(float(np.median(perf_track)), 2), "p95_ms": round(float(np.percentile(perf_track, 95)), 2)},
                    "motion_estimation": {"median_ms": round(float(np.median(perf_motion)), 2), "p95_ms": round(float(np.percentile(perf_motion, 95)), 2)},
                    "draw": {"median_ms": round(float(np.median(perf_draw)), 2), "p95_ms": round(float(np.percentile(perf_draw, 95)), 2)},
                    "encode": {"median_ms": round(float(np.median(perf_encode)), 2), "p95_ms": round(float(np.percentile(perf_encode, 95)), 2)},
                    "end_to_end": {"median_ms": round(curr_lat, 2), "p95_ms": round(e2e_p95, 2)}
                }

                # Update dashboard telemetry with honest FPS, latency, and stage timings
                try:
                    hub.update_telemetry(
                        fps=curr_fps,
                        latency_ms=curr_lat,
                        train_state=current_train_state.value,
                        is_fallback=is_fallback,
                        source_target=primary_tgt,
                        retry_status=retry_stat,
                        camera_motion=is_camera_moving,
                        camera_motion_reason=cam_reason,
                        detector_fps=det_fps,
                        stage_timings=stage_timings
                    )
                except Exception as tel_err:
                    logger.exception(f"[Live] Telemetry update error: {tel_err}")

                # Display on desktop window if enabled
                if show_window:
                    cv2.imshow(win_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Live] User pressed 'q'. Shutting down...")
                        break

            except Exception as frame_err:
                logger.exception(f"[Live] Error processing frame {frame_idx}: {frame_err}")
                # Continue video loop regardless of frame error!
                time.sleep(0.01)

            if max_frames and frame_idx >= max_frames:
                print(f"\n[Live] Reached target frame limit ({max_frames}).")
                break

    except KeyboardInterrupt:
        print("\n[Live] KeyboardInterrupt received. Shutting down...")

    finally:
        if 'stage_timings' in locals() and stage_timings:
            print("\n" + "="*52)
            print("PIPELINE STAGE PROFILE SUMMARY (ms):")
            print(f"{'Stage':<20} | {'Median (ms)':<12} | {'p95 (ms)':<10}")
            print("-" * 52)
            for st_name, metrics in stage_timings.items():
                print(f"{st_name:<20} | {metrics['median_ms']:<12.2f} | {metrics['p95_ms']:<10.2f}")
            print("="*52)
            print(f"Measured Display FPS: {curr_fps:.1f} | Detector FPS: {det_fps:.1f} | Latency: {curr_lat:.1f} ms\n")

        print("[Live] Cleaning up hardware resources...")
        try:
            if 'zone_tracker' in locals() and zone_tracker:
                zone_tracker.stop()
        except Exception:
            pass
        try:
            async_inference.stop()
        except Exception:
            pass
        buzzer.stop()
        frame_source.release()
        if show_window:
            cv2.destroyAllWindows()
        server_thread.shutdown()
        print("[Live] System shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMRC AI Metro Platform Safety System - Unified Live Pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: Phone stream URL (e.g. http://192.168.1.8:8080/video), webcam (0), or video file")
    parser.add_argument("--port", type=int, default=5000,
                        help="Dashboard web server port (default 5000)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path")
    parser.add_argument("--no-window", action="store_true",
                        help="Run headlessly without OpenCV desktop window")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video files continuously")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Exit after processing N frames")
    args = parser.parse_args()

    run_live_pipeline(
        source_uri=args.source,
        port=args.port,
        config_path=args.config,
        show_window=not args.no_window,
        loop_video=args.loop,
        max_frames=args.max_frames
    )
