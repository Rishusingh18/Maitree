"""Offline Replay & Kinematic Diagnostic Tool.
Replays a recorded video clip through the full platform safety pipeline with live
kinematic telemetry overlays (vertical velocity vy, torso tilt angle, and active zone)
to tune jump and fall thresholds without having to repeatedly physically re-act.

Usage:
  python -m metro_safety.tools.replay --video data/recordings/jump_test_1.mp4
  python -m metro_safety.tools.replay --video demo_clips/demo_jump.mp4 --train-state APPROACHING
"""
import time
import argparse
from pathlib import Path
import json
import cv2
import numpy as np

from typing import Optional, Any
from metro_safety.config import load_config
from metro_safety.perception.base import Detection
from metro_safety.perception.frame_source import create_frame_source
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.learned_classifier import LearnedEventClassifier
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.escalation.engine import EscalationEngine

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def replay_clip(
    video_path: str,
    config_path: str = None,
    train_state_val: str = "ABSENT",
    show_window: bool = True,
    output_log: str = "data/recordings/replay_audit.json",
    loop: bool = False,
    hub: Optional[Any] = None,
    buzzer: Optional[Any] = None,
    max_frames: Optional[int] = None
):
    cfg = load_config(config_path)
    train_state = TrainState(train_state_val)

    print(f"\n=======================================================")
    print(f"Replaying: {video_path}")
    print(f"Train State: {train_state.value}")
    print(f"Configured Fall Drop Threshold: {cfg['events'].get('fall_drop_velocity_threshold', 14.0)} px/frame")
    print(f"Configured Fall Angle Threshold: {cfg['events'].get('fall_angle_threshold', 40.0)} deg")
    print(f"Configured Jump Velocity Threshold: {cfg['events'].get('jump_vertical_velocity_threshold', 12.0)} px/frame")
    print(f"=======================================================\n")

    frame_source = create_frame_source(video_path, loop_video=loop)
    zm = ZoneManager(cfg["zones"])
    tracker = ByteTracker(
        track_high_thresh=cfg["tracking"]["track_high_thresh"],
        match_thresh=cfg["tracking"]["match_thresh"]
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
    gesture_det = DistressGestureDetector()
    escalation = EscalationEngine(station_id=cfg["system"]["station_id"], platform_id=cfg["system"]["platform_id"])

    # Load YOLO model
    pose_detector = YOLOPoseDetector(
        model_name=cfg["perception"]["model_name"],
        confidence_threshold=cfg["perception"]["confidence_threshold"],
        device=cfg["perception"]["device"]
    )

    win_name = f"Replay Telemetry: {Path(video_path).name} (SPACE=pause, 'q'=quit)"
    if show_window:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    events_logged = []
    paused = False

    while True:
        if not paused:
            ret, frame, _ = frame_source.read()
            if not ret or frame is None:
                print(f"\nReached end of replay after {frame_idx} frames.")
                break
            frame_idx += 1
            t_now = frame_idx / 25.0

            # 1. Perception
            detections = []
            if pose_detector and pose_detector._model is not None:
                detections = pose_detector.detect(frame)
            if not detections:
                # Fallback synthetic detection for simulated/rendered clips
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
            tracks = tracker.update(detections)
            # 3. Events
            rule_events = rule_clf.process_frame(frame_idx, t_now, tracks, zm, train_state)
            learned_events = learned_clf.process_frame(frame_idx, t_now, tracks, zm, train_state)
            gesture_events = gesture_det.process_frame(frame_idx, t_now, tracks, zm, train_state)
            all_events = rule_events + learned_events + gesture_events

            for ev in all_events:
                dec = escalation.evaluate(event=ev, train_state=train_state, current_time=t_now, snapshot=frame)
                kine = rule_clf.get_track_kinematics(ev.track_id)
                sev_name = dec.severity.name if hasattr(dec.severity, "name") else str(dec.severity)
                ev_name = ev.event_type.name if hasattr(ev.event_type, "name") else str(ev.event_type)

                print(f"[REPLAY EVENT] Frame {frame_idx:04d} (t={t_now:.2f}s) | Track #{ev.track_id:02d} -> {ev_name} [{sev_name}] "
                      f"| Zone: {ev.zone} | vy={kine['vy']} px/f | angle={kine['angle']} deg | conf={ev.confidence:.2f}")

                ev_record = {
                    "frame": frame_idx,
                    "timestamp": t_now,
                    "time_str": time.strftime("%H:%M:%S", time.localtime()),
                    "track_id": ev.track_id,
                    "event_type": ev_name,
                    "severity": sev_name,
                    "zone": ev.zone,
                    "confidence": round(ev.confidence, 2),
                    "action": dec.action_taken,
                    "kinematics": kine
                }
                events_logged.append(ev_record)

                if buzzer and sev_name != "DISTRESS":
                    buzzer.trigger(sev_name)
                if hub:
                    hub.push_event(ev_record)

            if max_frames and frame_idx >= max_frames:
                print(f"\nReached max frame limit ({max_frames}).")
                break

        if show_window:
            display = frame.copy()
            h, w = display.shape[:2]

            # Draw Zones
            zm.draw_zones(display, alpha=0.25)

            # Draw Skeletons & Kinematic Telemetry
            for trk in tracks:
                box = trk.box.astype(int)
                tid = trk.track_id
                kine = rule_clf.get_track_kinematics(tid)

                # Bounding box
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

                # Keypoints
                if trk.keypoints is not None:
                    kpts = trk.keypoints
                    for pt in kpts:
                        if pt[0] > 0 and pt[1] > 0:
                            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                    for p1, p2 in COCO_PAIRS:
                        if p1 < len(kpts) and p2 < len(kpts):
                            pt1, pt2 = kpts[p1], kpts[p2]
                            if pt1[0] > 0 and pt2[0] > 0:
                                cv2.line(display, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 120, 0), 1)

                # Kinematic HUD Box above person
                overlay_text = f"ID #{tid:02d} | {kine['zone']} | vy:{kine['vy']:+05.1f} | angle:{kine['angle']:04.1f}deg"
                (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bx = max(10, box[0])
                by = max(30, box[1] - 8)
                cv2.rectangle(display, (bx - 2, by - th - 4), (bx + tw + 4, by + 4), (20, 20, 20), -1)
                cv2.putText(display, overlay_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Top Replay Status Bar
            cv2.rectangle(display, (0, 0), (w, 35), (20, 20, 20), -1)
            status_txt = f"FRAME: {frame_idx:04d} | TRACKS: {len(tracks)} | TRAIN: {train_state.value} | EVENTS: {len(events_logged)}"
            cv2.putText(display, status_txt, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(20 if not paused else 100) & 0xFF
            if key == ord(' '):
                paused = not paused
                print(f"[REPLAY] {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord('q'):
                print("[REPLAY] Exiting.")
                break

    frame_source.release()
    if show_window:
        cv2.destroyAllWindows()

    out_p = Path(output_log)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"events": events_logged, "total_events": len(events_logged)}, f, indent=2)
    print(f"Replay telemetry audit written to: {out_p}")
    return events_logged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay video clip with kinematic telemetry")
    parser.add_argument("--video", type=str, required=True, help="Video path to replay")
    parser.add_argument("--train-state", type=str, default="ABSENT", help="Train state: ABSENT, APPROACHING, PRESENT")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--out-log", type=str, default="data/recordings/replay_audit.json", help="Output audit JSON")
    parser.add_argument("--loop", action="store_true", help="Loop video continuously")
    parser.add_argument("--no-window", action="store_true", help="Run headlessly")
    args = parser.parse_args()

    replay_clip(
        video_path=args.video,
        config_path=args.config,
        train_state_val=args.train_state,
        show_window=not args.no_window,
        output_log=args.out_log,
        loop=args.loop
    )
