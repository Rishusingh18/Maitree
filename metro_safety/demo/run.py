"""Single-command end-to-end pipeline runner for AI Metro Platform Safety System.
Supports:
- Live camera streams (HTTP MJPEG/RTSP from phone IP webcam apps, or webcam index 0)
- Local video files (with fallback)
- Real-time display window with live annotations (zones, tracks, events, severity banners, buzzer/ETS indicators)
- On-screen live FPS and end-to-end processing latency counter
- Threaded frame grabber dropping stale frames (buffer size 1)
- Automatic reconnection with exponential backoff on dropped streams
- JSON audit log & optional output video recording
"""
from typing import List, Dict, Optional, Tuple, Union
import argparse
from pathlib import Path
import json
import os
import time
import threading
import numpy as np
import cv2

from metro_safety.config import load_config
from metro_safety.zones.geometry import ZoneManager
from metro_safety.zones.registration import AdaptiveZoneTracker, AlignmentStatus
from metro_safety.perception.train_state import TrainState, SimulatedTrainStateProvider
from metro_safety.perception.frame_source import create_frame_source, FrameSource, VideoFileSource
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.learned_classifier import LearnedEventClassifier
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.escalation.engine import EscalationEngine
from metro_safety.perception.base import Detection
from metro_safety.events.models import SeverityLevel

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def run_pipeline(
    source_uri: Optional[Union[str, int]] = None,
    output_video: Optional[Union[Path, str]] = None,
    output_log: Path | str = "demo_events.json",
    train_state_val: str = "ABSENT",
    config_path: Optional[str] = None,
    show_window: bool = True,
    max_frames: Optional[int] = None,
    fallback_video: str = "demo_clips/demo_fall.mp4",
    loop_video: bool = False,
    stop_event: Optional[threading.Event] = None
):
    cfg = load_config(config_path)

    # Determine source URI: CLI flag > METRO_SOURCE_OVERRIDE > config.yaml > fallback
    source_override = os.environ.get("METRO_SOURCE_OVERRIDE")
    if source_uri is not None:
        resolved_source = source_uri
    elif source_override:
        resolved_source = source_override
    else:
        resolved_source = cfg["system"].get("source", "http://192.168.1.32:8080/video")

    alias_map = {
        "fall": "demo_clips/demo_fall.mp4",
        "jump": "demo_clips/demo_jump.mp4",
        "distress": "demo_clips/demo_distress.mp4"
    }
    if str(resolved_source).lower() in alias_map:
        resolved_source = alias_map[str(resolved_source).lower()]

    zm = ZoneManager(cfg["zones"])
    zone_tracker = AdaptiveZoneTracker(
        zone_manager=zm,
        config=cfg,
        reference_image_path=cfg.get("reference_frame")
    )
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
    from metro_safety.anomaly.engine import AnomalyEngine
    anomaly_engine = AnomalyEngine(cfg)
    # Connect to shared LiveHub so run.py also feeds the dashboard
    from metro_safety.demo.hub import get_global_hub
    hub = get_global_hub()
    hub.cctv_id = cfg["system"]["camera_id"]
    hub.station_id = cfg["system"]["station_id"]
    hub.platform_id = cfg["system"]["platform_id"]

    # Connect or initialize real hardware PC buzzer
    buzzer_cfg = cfg.get("buzzer", {})
    buzzer = getattr(hub, "buzzer", None)
    if buzzer is None:
        try:
            from metro_safety.escalation.buzzer import HardwareBuzzer
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
            hub.set_hardware_components(buzzer=buzzer, zone_tracker=zone_tracker)
        except Exception as _b_err:
            buzzer = None
            hub.set_hardware_components(zone_tracker=zone_tracker)
    else:
        if "audio_enabled" in buzzer_cfg:
            buzzer.audio_enabled = bool(buzzer_cfg["audio_enabled"])
        if "clear_timeout_sec" in buzzer_cfg:
            buzzer.clear_timeout_sec = float(buzzer_cfg["clear_timeout_sec"])
        hub.set_hardware_components(buzzer=buzzer, zone_tracker=zone_tracker)

    escalation = EscalationEngine(
        station_id=cfg["system"]["station_id"],
        platform_id=cfg["system"]["platform_id"],
        buzzer=buzzer
    )
    train_state = TrainState(train_state_val)
    if not hasattr(hub, "train_state_provider") or hub.train_state_provider is None:
        hub.set_hardware_components(train_state_provider=SimulatedTrainStateProvider(default_platform_id=cfg["system"]["platform_id"]))
    if hasattr(hub, "train_state_provider") and hub.train_state_provider:
        hub.train_state_provider.set_state(train_state, platform_id=cfg["system"]["platform_id"])
    hub.set_train_state(train_state.value, platform_id=cfg["system"]["platform_id"])

    # Initialize FrameSource (threaded, buffer size 1, auto-reconnect, with fallback)
    print(f"[Pipeline] Initializing FrameSource for: {resolved_source}")
    frame_source: FrameSource = create_frame_source(
        source_uri=resolved_source,
        fallback_path=fallback_video,
        allow_fallback=True,
        loop_video=loop_video
    )
    print(f"[Pipeline] Active source: {frame_source.source_name}")

    # Initialize YOLO pose model behind abstract interface
    pose_detector = None
    try:
        from metro_safety.perception.yolo_pose import YOLOPoseDetector
        pose_detector = YOLOPoseDetector(
            model_name=cfg["perception"]["model_name"],
            confidence_threshold=cfg["perception"]["confidence_threshold"],
            device=cfg["perception"]["device"],
            inference_size=int(cfg["perception"].get("inference_size", 480))
        )
        print(f"[Pipeline] Loaded YOLO pose detector ({cfg['perception']['model_name']})")
        if hasattr(pose_detector, "_model") and pose_detector._model is not None:
            names = getattr(pose_detector._model, "names", {})
            if any(cls_name in names.values() for cls_name in ["backpack", "handbag", "suitcase"]):
                anomaly_engine.set_object_detector(pose_detector._model)
    except Exception as e:
        print(f"[Pipeline] [Notice] YOLO model load failed ({e}), using synthetic detection tracker.")

    # Initialize optional video recorder
    writer = None
    window_name = f"DMRC CCTV Safety System - [{cfg['system']['camera_id']}] (Press 'q' to stop)"
    if show_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except (cv2.error, Exception) as e:
            print(f"[Pipeline] [Notice] GUI window unavailable ({e}). Automatically switching to headless mode.")
            show_window = False

    events_logged = []
    frame_idx = 0
    fps_history = []
    latency_history = []
    t_prev_frame = time.perf_counter()
    last_zone_event_time = {}
    zone_cooldown_sec = float(cfg.get("adaptive_zones", {}).get("unverified_zone_cooldown_sec", 4.0))

    try:
        while True:
            if stop_event and stop_event.is_set():
                print("[Pipeline] Stop signal received. Exiting vision loop.")
                break

            t_frame_start = time.perf_counter()

            # Read latest non-stale frame from threaded source
            ret, frame, frame_ts = frame_source.read()
            if not ret or frame is None:
                # If reading a non-looping video file, we have reached EOF
                if isinstance(frame_source, VideoFileSource) and not frame_source.loop:
                    print(f"[Pipeline] End of video reached after {frame_idx} frames.")
                    break
                # If reading live stream, wait briefly for new frame
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Adaptive zone tracking: reference capture on first frame and demo mode toggle
            if frame_idx == 0:
                zone_tracker.on_first_frame(frame)

            is_fallback = getattr(frame_source, "is_fallback", False)
            is_demo = is_fallback or ("demo" in str(resolved_source).lower())
            zone_tracker.set_demo_mode(is_demo)

            # Lazy initialize video writer if output path requested
            if output_video and writer is None:
                out_v_path = Path(output_video)
                out_v_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_v_path), fourcc, 25.0, (w, h))

            # 1. Perception Detection
            detections = []
            if pose_detector and pose_detector._model is not None:
                detections = pose_detector.detect(frame)
            else:
                # Fallback synthetic detection for simulated clips
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

            # 2. Tracking
            tracked_dets = tracker.update(detections)

            # Check for cross-process zone confirmation command
            cmd_path = Path("data/cache/zone_confirm.cmd")
            if cmd_path.exists():
                try:
                    cmd_content = cmd_path.read_text(encoding="utf-8")
                    cmd_data = json.loads(cmd_content)
                    cmd_path.unlink()
                    req_id = cmd_data.get("req_id")
                    cmd_ts = float(cmd_data.get("timestamp", 0))
                    if (time.time() - cmd_ts) <= 2.0:
                        c_ok, c_msg = zone_tracker.confirm_alignment()
                        resp_path = Path("data/cache/zone_confirm.resp")
                        resp_path.write_text(json.dumps({
                            "req_id": req_id,
                            "status": "ok" if c_ok else "error",
                            "message": c_msg
                        }), encoding="utf-8")
                    else:
                        logger.warning(f"[Pipeline] Discarding timed-out zone confirmation command: req_id={req_id}")
                except Exception:
                    pass

            # Check for cross-process buzzer control command (acknowledge / mute)
            buzzer_cmd = Path("data/cache/buzzer_control.cmd")
            if buzzer_cmd.exists():
                try:
                    b_content = buzzer_cmd.read_text(encoding="utf-8")
                    b_data = json.loads(b_content)
                    buzzer_cmd.unlink()
                    if (time.time() - float(b_data.get("timestamp", 0))) <= 5.0:
                        act = b_data.get("action")
                        if act == "acknowledge" and buzzer:
                            buzzer.acknowledge()
                        elif act == "toggle_mute" and buzzer:
                            buzzer.toggle_mute()
                except Exception:
                    pass

            # Adaptive Zone Tracking: Non-blocking frame registration
            person_boxes = [d.box for d in tracked_dets] if tracked_dets else [d.box for d in detections]
            zone_tracker.submit_frame(frame.copy(), frame_idx, person_boxes, frame_ts)
            reg_result = zone_tracker.get_latest_result()
            frame_zone_version = zm.zone_version
            frame_zones_unverified = reg_result.zones_unverified

            # 3. Behavior Events, Gestures & Anomaly Detection
            current_train_state = hub.train_state_provider.get_state() if hasattr(hub, "train_state_provider") and hub.train_state_provider else train_state
            rule_evs = rule_clf.process_frame(
                frame_idx,
                frame_ts,
                tracked_dets,
                zm,
                current_train_state,
                train_state_provider=getattr(hub, "train_state_provider", None),
                platform_id=getattr(hub, "platform_id", None)
            )
            gest_evs = gesture_det.process_frame(frame_idx, frame_ts, tracked_dets, zm, current_train_state)
            cam_moving = getattr(rule_clf, "camera_moving", False)
            cam_reason = getattr(rule_clf, "motion_reason", "Optical flow vibration")
            anom_evs = anomaly_engine.process_frame(
                frame=frame,
                frame_idx=frame_idx,
                person_detections=tracked_dets,
                zone_manager=zm,
                timestamp=frame_ts,
                train_state=current_train_state.value if hasattr(current_train_state, "value") else str(current_train_state),
                camera_motion=cam_moving,
                motion_reason=cam_reason,
                zones_unverified=frame_zones_unverified,
                zone_version=frame_zone_version
            )
            all_evs = rule_evs + gest_evs + anom_evs

            # Tag events with consistent geometry snapshot
            for ev in all_evs:
                ev.zones_unverified = frame_zones_unverified
                ev.zone_version = frame_zone_version

            # Per-frame escalation lifecycle: clear confirmed resolved tracks (present in safe platform with no active hazards)
            active_hazard_tracks = {ev.track_id for ev in all_evs if hasattr(ev, 'track_id')}
            has_intrusion = False
            for det in tracked_dets:
                zone = zm.get_zone_for_point(det.foot_anchor)
                if zone in ["yellow_band", "track_bed", "danger_zone"]:
                    has_intrusion = True
                if det.track_id is not None and det.track_id not in active_hazard_tracks:
                    if zone == "platform":
                        escalation.clear_track(det.track_id)
            if buzzer:
                try:
                    buzzer.update_clear_status(has_intrusion)
                except Exception:
                    pass

            # 4. Closed-Loop Escalation: Evaluate all events first before deduplication/cooldown filters
            SEVERITY_PRIORITY = {
                SeverityLevel.L3: 4,
                SeverityLevel.L2: 3,
                SeverityLevel.L1: 2,
                SeverityLevel.DISTRESS: 1,
                SeverityLevel.NORMAL: 0
            }
            current_decision = None
            for ev_idx, ev in enumerate(all_evs):
                dec = escalation.evaluate(event=ev, train_state=current_train_state, current_time=frame_ts, snapshot=frame)
                if current_decision is None or SEVERITY_PRIORITY.get(dec.severity, 0) > SEVERITY_PRIORITY.get(current_decision.severity, 0):
                    current_decision = dec
                if reg_result.zones_unverified and not dec.action_taken.startswith("[ZONES UNVERIFIED]"):
                    dec.action_taken = f"[ZONES UNVERIFIED] {dec.action_taken}"

                # Deduplication / Cooldown filter for notification & logging
                if frame_zones_unverified:
                    z_name = getattr(ev, "zone", "platform")
                    ev_type_name = getattr(getattr(ev, "event_type", None), "name", str(getattr(ev, "event_type", "EVENT")))
                    z_key = (z_name, ev_type_name)
                    if z_key in last_zone_event_time and (frame_ts - last_zone_event_time[z_key]) < zone_cooldown_sec:
                        continue
                    last_zone_event_time[z_key] = frame_ts

                ev_data = ev.to_dict() if hasattr(ev, 'to_dict') else {}
                log_entry = {
                    "frame": frame_idx,
                    "timestamp": round(frame_ts, 3),
                    "event": ev_data,
                    "severity": dec.severity.value,
                    "action": dec.action_taken,
                    "buzzer_active": dec.buzzer_active,
                    "ets_triggered": dec.ets_triggered,
                    "latency_ms": dec.latency_ms,
                    "zones_unverified": frame_zones_unverified,
                    "zone_version": frame_zone_version
                }
                events_logged.append(log_entry)

                # Push to live hub for dashboard & SSE streaming (unique ID per event in frame)
                try:
                    norm_event_type = ev_data.get("event_type", getattr(getattr(ev, "event_type", None), "value", str(getattr(ev, "event_type", "INCIDENT"))))
                    hub.push_event({
                        "id": f"EV-{int(frame_ts * 1000)}-{frame_idx}-{ev_idx}",
                        "timestamp": frame_ts,
                        "time_str": time.strftime("%H:%M:%S", time.localtime(frame_ts)),
                        "track_id": getattr(ev, "track_id", 0),
                        "event_type": norm_event_type,
                        "zone": getattr(ev, "zone", "platform"),
                        "severity": dec.severity.value,
                        "confidence": getattr(ev, "confidence", 0.95),
                        "action": dec.action_taken,
                        "is_demo": bool(is_demo),
                        "is_anomaly": bool(getattr(ev, "metadata", {}).get("is_anomaly", False)),
                        "anomaly_type": getattr(ev, "metadata", {}).get("anomaly_type", ""),
                        "snapshot_crop": getattr(ev, "metadata", {}).get("snapshot_crop", None),
                        "session_id": hub.session_id,
                        "zones_unverified": frame_zones_unverified,
                        "zone_version": frame_zone_version
                    }, frame=frame)
                except Exception:
                    pass

            # Calculate live FPS & end-to-end processing latency
            t_now = time.perf_counter()
            frame_latency_ms = (t_now - t_frame_start) * 1000.0
            latency_history.append(frame_latency_ms)
            if len(latency_history) > 30:
                latency_history.pop(0)

            dt_fps = t_now - t_prev_frame
            t_prev_frame = t_now
            instant_fps = (1.0 / dt_fps) if dt_fps > 0 else 25.0
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = float(np.mean(fps_history))
            avg_lat_ms = float(np.mean(latency_history))

            # 5. Visual Rendering & Live Annotation
            status_badge_text = f"ALIGN: {reg_result.status.value} | DRIFT: {reg_result.drift_px:.1f}px | INL: {reg_result.inlier_count} | V:{reg_result.zone_version}"
            annotated = zm.draw_zones(
                frame,
                alpha=0.25,
                unverified=reg_result.zones_unverified,
                draw_baseline=True,
                drift_px=reg_result.drift_px,
                status_badge=status_badge_text
            )

            # Draw tracked persons
            for det in tracked_dets:
                bx = det.box.astype(int)
                tid = det.track_id or 0
                foot = det.foot_anchor
                zone_name = zm.get_zone_for_point(foot)

                # Color box depending on zone
                box_color = (0, 0, 255) if zone_name in ["track_bed", "danger_zone"] else ((0, 255, 255) if zone_name == "yellow_band" else (0, 255, 0))
                cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), box_color, 2)
                cv2.putText(annotated, f"ID:{tid} [{zone_name.upper()}]", (bx[0], max(18, bx[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                cv2.circle(annotated, (int(foot[0]), int(foot[1])), 5, (0, 0, 255), -1)

                # COCO Skeleton & Keypoint joints
                if det.keypoints is not None:
                    for p1_i, p2_i in COCO_PAIRS:
                        p1 = det.keypoints[p1_i]
                        p2 = det.keypoints[p2_i]
                        if p1[2] > 0.20 and p2[2] > 0.20:
                            cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
                    for ki in range(17):
                        kp = det.keypoints[ki]
                        if kp[2] > 0.20:
                            cv2.circle(annotated, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

            # Severity Banner
            banner_h = 75
            sev = current_decision.severity if current_decision else SeverityLevel.NORMAL
            if sev == SeverityLevel.L3:
                banner_color = (0, 0, 220)      # Bright Red
                status_text = "LEVEL 3 EMERGENCY: TRAIN STOP RECOMMENDED / ETS TRIPPED"
            elif sev == SeverityLevel.L2:
                banner_color = (0, 140, 255)    # Orange
                status_text = "LEVEL 2 ALERT: PERSON ON TRACK (BUZZER + CONTROL ROOM)"
            elif sev == SeverityLevel.L1:
                banner_color = (0, 220, 240)    # Yellow
                status_text = "LEVEL 1 WARNING: YELLOW LINE CROSSED (PLATFORM BUZZER)"
            elif sev == SeverityLevel.DISTRESS:
                banner_color = (180, 0, 180)    # Purple
                status_text = "SILENT DISTRESS GESTURE: DISPATCHING STATION SECURITY"
            else:
                banner_color = (30, 130, 30)    # Green
                ts_val = current_train_state.value if hasattr(current_train_state, "value") else str(current_train_state)
                status_text = f"PLATFORM SECURE | TRAIN: {ts_val}"

            cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_color, -1)
            cv2.putText(annotated, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Live Latency, FPS, and Source Telemetry counter
            src_label = frame_source.source_name
            if len(src_label) > 40:
                src_label = src_label[:37] + "..."
            stats_line = f"FPS: {avg_fps:.1f} | LATENCY: {avg_lat_ms:.1f} ms | SRC: {src_label}"
            cv2.putText(annotated, stats_line, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

            # Hardware Indicators (Buzzer & ETS)
            buzzer_on = escalation.buzzer.is_active(escalation.platform_id)
            ets_on = escalation.trip_switch.is_tripped()

            b_color = (0, 0, 255) if buzzer_on else (80, 80, 80)
            e_color = (0, 0, 255) if ets_on else (80, 80, 80)
            cv2.circle(annotated, (w - 180, 40), 8, b_color, -1)
            cv2.putText(annotated, "BUZZER", (w - 165, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.circle(annotated, (w - 75, 40), 8, e_color, -1)
            cv2.putText(annotated, "ETS", (w - 60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Write frame if recording
            if writer:
                writer.write(annotated)

            # Publish to shared LiveHub for web dashboard
            try:
                hub.update_frame(annotated, is_online=True)
                hub.update_telemetry(
                    fps=avg_fps,
                    latency_ms=avg_lat_ms,
                    train_state=current_train_state.value if hasattr(current_train_state, "value") else str(current_train_state),
                    is_fallback=getattr(frame_source, "is_fallback", False),
                    source_target=getattr(frame_source, "primary_target", str(resolved_source)),
                    retry_status=getattr(frame_source, "retry_status", "Active")
                )
            except Exception:
                pass

            # Display window
            if show_window:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Pipeline] User requested exit.")
                    break

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

    finally:
        frame_source.release()
        if zone_tracker:
            try:
                zone_tracker.stop()
            except Exception:
                pass
        if buzzer:
            try:
                buzzer.deactivate()
            except Exception:
                pass
        if writer:
            writer.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # Save Event Log
    out_log_path = Path(output_log)
    out_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": frame_source.source_name,
            "frames_processed": frame_idx,
            "total_events": len(events_logged),
            "events": events_logged
        }, f, indent=2)

    print(f"[Pipeline] Finished. Frames processed: {frame_idx}. Events logged to: {out_log_path}")
    if output_video:
        print(f"[Pipeline] Video written to: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Metro Platform Safety System Live Stream & File Pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: Phone stream URL (http://192.168.1.32:8080/video), webcam index (0), video file path, or alias (fall/jump/distress)")
    parser.add_argument("--video", type=str, default=None,
                        help="Legacy alias for --source")
    parser.add_argument("--out-video", type=str, default=None,
                        help="Optional output annotated video path")
    parser.add_argument("--out-log", type=str, default="demo_events.json",
                        help="JSON events audit log output path")
    parser.add_argument("--train-state", type=str, default="ABSENT",
                        help="Train state: ABSENT, APPROACHING, PRESENT")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless mode (disable live GUI window)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Exit after processing N frames")
    parser.add_argument("--loop", action="store_true",
                        help="Loop pre-recorded video file continuously")
    parser.add_argument("--fallback-video", type=str, default="demo_clips/demo_fall.mp4",
                        help="Pre-recorded fallback video if live stream is unavailable")
    args = parser.parse_args()

    chosen_source = args.source if args.source is not None else args.video

    run_pipeline(
        source_uri=chosen_source,
        output_video=args.out_video,
        output_log=args.out_log,
        train_state_val=args.train_state,
        config_path=args.config,
        show_window=not args.no_window,
        max_frames=args.max_frames,
        fallback_video=args.fallback_video,
        loop_video=args.loop
    )
