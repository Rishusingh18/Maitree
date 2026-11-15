"""Video Upload and Forensic Analysis Processor for Metro Platform Safety System.

Accepts an uploaded platform CCTV or simulation clip, runs full computer vision inference:
- Polygon safety zones (platform, yellow_band, track_bed, danger_zone)
- YOLOv8 Pose / Synthetic Joint detection
- ByteTracker object tracking
- Behavior, Gesture, and Anomaly Engines
- Visual overlays: colored bounding boxes, pose skeleton links, severity banners
- Registers all detected incidents into LiveHub for real-time audit and alert dispatch
- Encodes output video and maintains JPEG frame cache for low-latency browser streaming
"""
import os
import time
import uuid
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from metro_safety.config import load_config
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.base import Detection
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.escalation.engine import EscalationEngine, SeverityLevel
from metro_safety.perception.train_state import TrainState
from metro_safety.demo.hub import get_global_hub, save_incident_snapshot

logger = logging.getLogger("metro_safety.video_processor")

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial features
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Upper body & Arms
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Lower body & Legs
]

# Global cache of processed video frames for MJPEG streaming: video_id -> List[bytes]
PROCESSED_FRAME_CACHE: Dict[str, List[bytes]] = {}
PROCESSED_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def get_cached_processed_stream(video_id: str):
    """Generator for streaming processed video frames as MJPEG."""
    frames = PROCESSED_FRAME_CACHE.get(video_id, [])
    if not frames:
        # If not in memory, try to load from disk frames directory if available
        frames_dir = Path("data/uploads") / f"frames_{video_id}"
        if frames_dir.exists():
            img_files = sorted(frames_dir.glob("*.jpg"))
            frames = [p.read_bytes() for p in img_files]
            if frames:
                PROCESSED_FRAME_CACHE[video_id] = frames

    if not frames:
        # Yield single placeholder error frame
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Processed stream not found", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", blank)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return

    # Loop frames continuously for smooth playback in browser
    idx = 0
    while True:
        frame_bytes = frames[idx % len(frames)]
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        idx += 1
        time.sleep(0.04)  # ~25 FPS


def process_uploaded_video(
    input_path: Path,
    original_filename: str = "uploaded_video.mp4",
    config_path: Optional[str] = None,
    register_to_hub: bool = True,
    platform_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process an uploaded video through the safety AI pipeline and register incidents."""
    cfg = load_config(config_path)
    video_id = f"VID-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = uploads_dir / f"frames_{video_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = uploads_dir / f"processed_{video_id}.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale zones to match video resolution
    raw_zones = cfg.get("zones", {})
    ref_w, ref_h = 1920.0, 1080.0
    sx, sy = float(w) / ref_w, float(h) / ref_h

    scaled_zones: Dict[str, List[List[float]]] = {}
    for zname, pts in raw_zones.items():
        if pts and len(pts) >= 3:
            scaled_zones[zname] = [[round(p[0] * sx, 1), round(p[1] * sy, 1)] for p in pts]

    zm = ZoneManager(scaled_zones)

    # Initialize detectors & classifiers
    pose_detector = YOLOPoseDetector(
        model_name=cfg["perception"].get("model_name", "yolov8n-pose.pt"),
        confidence_threshold=cfg["perception"].get("confidence_threshold", 0.18),
        device=cfg["perception"].get("device", "cpu")
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
    gesture_det = DistressGestureDetector(
        min_persistence=cfg["gesture"]["min_persistence_frames"],
        use_macro_fallback=cfg["gesture"]["use_macro_fallback"]
    )
    escalation = EscalationEngine(
        station_id=cfg["system"].get("station_id", "DMRC_STN_042"),
        platform_id=platform_id or cfg["system"].get("platform_id", "PL_DOWN_01")
    )

    try:
        from metro_safety.anomaly.engine import AnomalyEngine
        anomaly_engine = AnomalyEngine(cfg)
    except Exception:
        anomaly_engine = None

    hub = get_global_hub()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    detected_incidents: List[Dict[str, Any]] = []
    registered_event_types: set = set()
    frame_bytes_list: List[bytes] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_ts = float(frame_idx) / float(fps)
        time_str = time.strftime("%H:%M:%S", time.gmtime(frame_ts))

        # 1. Detection (YOLO with fallback for simulated synthetic clips)
        detections = []
        if pose_detector and pose_detector._model is not None:
            detections = pose_detector.detect(frame)

        if not detections:
            # Fallback synthetic detection for simulated test clips
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

        # 3. Classifications & Anomaly Detection
        rule_evs = rule_clf.process_frame(
            frame_idx,
            frame_ts,
            tracked_dets,
            zm,
            TrainState.ABSENT,
            camera_motion=False,
            motion_reason="Stable"
        )
        gesture_evs = gesture_det.process_frame(frame_idx, frame_ts, tracked_dets, zm, TrainState.ABSENT)
        anomaly_evs = []
        if anomaly_engine is not None:
            anomaly_evs = anomaly_engine.process_frame(
                frame=frame,
                frame_idx=frame_idx,
                person_detections=tracked_dets,
                zone_manager=zm,
                train_state="ABSENT"
            )

        all_evs = rule_evs + gesture_evs + anomaly_evs
        current_decision = None

        # Evaluate escalation & register events
        for ev in all_evs:
            dec = escalation.evaluate(event=ev, train_state=None, current_time=frame_ts)
            _sev_rank = {'NORMAL': 0, 'L1': 1, 'DISTRESS': 2, 'L2': 3, 'L3': 4}
            _dec_s = dec.severity.name if hasattr(dec.severity, 'name') else str(dec.severity)
            if current_decision is None:
                current_decision = dec
            else:
                _curr_s = current_decision.severity.name if hasattr(current_decision.severity, 'name') else str(current_decision.severity)
                if _sev_rank.get(_dec_s, 0) > _sev_rank.get(_curr_s, 0):
                    current_decision = dec

            # Deduplicate by (event_type, track_id) to avoid flooding identical events
            dedup_key = (ev.event_type.value, ev.track_id or 0)
            if dedup_key not in registered_event_types:
                registered_event_types.add(dedup_key)

                sev_name = dec.severity.name if dec else "L1"
                ev_id = f"EV-UPL-{int(time.time()*1000)}-{ev.track_id or 0}-{ev.event_type.value}"

                rec = {
                    "id": ev_id,
                    "timestamp": time.time(),
                    "time_str": time_str,
                    "frame_idx": frame_idx,
                    "video_id": video_id,
                    "track_id": ev.track_id or 0,
                    "event_type": ev.event_type.value,
                    "severity": sev_name,
                    "zone": ev.zone or "platform",
                    "confidence": round(float(ev.confidence), 2),
                    "action": dec.action_taken if dec else "Threat detected in uploaded footage",
                    "source": f"Upload: {original_filename}",
                    "cctv_id": f"UPLOAD-{video_id[:8]}"
                }

                if register_to_hub:
                    hub.push_event(rec, frame=frame)
                    rec["snapshot_url"] = f"/api/snapshots/{ev_id}.jpg"

                detected_incidents.append(rec)

        # 4. Render Visual Overlays
        # A. Safety Zone Polygons
        annotated = zm.draw_zones(
            frame,
            alpha=0.28,
            unverified=False,
            draw_baseline=True,
            status_badge=f"VIDEO ANALYSIS | FRAME {frame_idx}/{total_frames} | {time_str}"
        )

        # B. Person Bounding Boxes & Skeletons
        for det in tracked_dets:
            bx = det.box.astype(int)
            tid = det.track_id or 0
            foot = det.foot_anchor
            zone_name = zm.get_zone_for_point(foot)

            # Red for track/danger, Yellow for yellow band, Green for platform
            box_color = (0, 0, 255) if zone_name in ["track_bed", "danger_zone"] else ((0, 230, 255) if zone_name == "yellow_band" else (0, 200, 0))
            cv2.rectangle(annotated, (bx[0], bx[1]), (bx[2], bx[3]), box_color, 2)
            cv2.putText(annotated, f"ID:{tid} [{zone_name.upper()}]", (bx[0], max(18, bx[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2)
            cv2.circle(annotated, (int(foot[0]), int(foot[1])), 5, (0, 0, 255), -1)

            # Skeletons
            if det.keypoints is not None:
                for p1_i, p2_i in COCO_PAIRS:
                    p1 = det.keypoints[p1_i]
                    p2 = det.keypoints[p2_i]
                    if p1[2] > 0.15 and p2[2] > 0.15:
                        cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
                for ki in range(17):
                    kp = det.keypoints[ki]
                    if kp[2] > 0.15:
                        cv2.circle(annotated, (int(kp[0]), int(kp[1])), 3, (0, 200, 255), -1)

        # C. Incident Banner Overlay
        if current_decision and current_decision.severity != SeverityLevel.NORMAL:
            sev = current_decision.severity
            banner_color = (0, 0, 220) if sev == SeverityLevel.L3 else ((0, 140, 255) if sev == SeverityLevel.L2 else (0, 220, 240))
            cv2.rectangle(annotated, (0, 0), (w, 42), banner_color, -1)
            banner_text = f"🚨 {sev.name}: {current_decision.action_taken}"
            cv2.putText(annotated, banner_text[:80], (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Write to output MP4
        if writer is not None:
            writer.write(annotated)

        # Encode JPEG for fast browser preview and MJPEG loop
        ret_enc, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret_enc:
            jpeg_bytes = jpeg_buf.tobytes()
            frame_bytes_list.append(jpeg_bytes)
            # Write keyframes to disk frames_dir
            if frame_idx % 2 == 0 or frame_idx < 30:
                (frames_dir / f"f_{frame_idx:05d}.jpg").write_bytes(jpeg_bytes)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Cache processed frames in memory for fast MJPEG streaming
    PROCESSED_FRAME_CACHE[video_id] = frame_bytes_list

    duration_sec = round(float(frame_idx) / float(fps), 2)
    meta = {
        "status": "ok",
        "video_id": video_id,
        "filename": original_filename,
        "total_frames": frame_idx,
        "duration_sec": duration_sec,
        "fps": fps,
        "resolution": [w, h],
        "incidents_count": len(detected_incidents),
        "incidents": detected_incidents,
        "video_url": f"/api/processed_video/{video_id}",
        "stream_url": f"/api/processed_stream/{video_id}",
        "thumbnail_url": f"/api/processed_frame/{video_id}/0" if frame_bytes_list else ""
    }
    PROCESSED_METADATA_CACHE[video_id] = meta

    logger.info(
        f"[VideoProcessor] Processed video {original_filename} ({frame_idx} frames). "
        f"Found {len(detected_incidents)} incidents. Output: {out_video_path}"
    )
    return meta
