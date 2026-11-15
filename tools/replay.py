"""Video Replay and Anomaly Visualizer Tool with HUD.

Loads a video clip or recorded CCTV footage, runs full perception and anomaly detection,
and renders an informative HUD overlay showing:
- Safety zones
- Prone timers (for PERSON_DOWN)
- Dwell timers (for LOITERING)
- Bag-owner association links & unattended timers (for ABANDONED_OBJECT)
- Snatch alerts & suppression status (for POSSIBLE_SNATCH)
- Live FPS, latency, and event notices.

Usage:
    python tools/replay.py --source demo_clips/demo_fall.mp4
    python tools/replay.py --source demo_clips/demo_fall.mp4 --save output.mp4
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Ensure repo root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from metro_safety.config import load_config
from metro_safety.zones.geometry import ZoneManager
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.perception.frame_source import create_frame_source
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.anomaly.engine import AnomalyEngine
from metro_safety.events.rule_classifier import RuleBasedEventClassifier

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def draw_hud(
    frame: np.ndarray,
    zone_manager: ZoneManager,
    tracked_dets: List[Any],
    anomaly_engine: AnomalyEngine,
    events: List[Any],
    fps: float,
    frame_idx: int
) -> np.ndarray:
    """Draw rich HUD overlay on frame."""
    out = frame.copy()

    # 1. Draw zones
    colors = {
        "platform": (80, 160, 80),
        "yellow_band": (30, 200, 240),
        "track_bed": (40, 40, 220),
        "danger_zone": (30, 30, 230),
        "ignore": (120, 120, 120)
    }
    overlay = out.copy()
    for name, poly in zone_manager.zones.items():
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(out, [pts], True, colors.get(name, (200, 200, 200)), 2)
            cv2.fillPoly(overlay, [pts], colors.get(name, (100, 100, 100)))
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

    # 2. Draw persons & skeletons
    for det in tracked_dets:
        tid = det.track_id or 0
        box = det.box.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 255, 180), 2)

        # Pose
        if det.keypoints is not None:
            kps = det.keypoints
            for p1, p2 in COCO_PAIRS:
                if p1 < len(kps) and p2 < len(kps):
                    pt1, pt2 = kps[p1], kps[p2]
                    if pt1[2] > 0.3 and pt2[2] > 0.3:
                        cv2.line(out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 200, 100), 2)
            for kp in kps:
                if kp[2] > 0.3:
                    cv2.circle(out, (int(kp[0]), int(kp[1])), 3, (0, 220, 255), -1)

        # Label & Anomaly Track HUD
        label_lines = [f"Track #{tid}"]

        # Prone timer
        pd_detector = getattr(anomaly_engine, "person_down_detector", None)
        pd_state = pd_detector.track_states.get(tid) if pd_detector and hasattr(pd_detector, "track_states") else None
        if pd_state and getattr(pd_state, "is_currently_prone", False):
            p_time = pd_state.prone_duration
            p_tag = "L2 ALERT" if p_time >= 30.0 else ("L1 NOTICE" if p_time >= 10.0 else "Monitoring")
            label_lines.append(f"Prone: {p_time:.1f}s [{p_tag}]")

        # Loiter timer
        loit_detector = getattr(anomaly_engine, "loitering_detector", None)
        loit_state = loit_detector.track_states.get(tid) if loit_detector and hasattr(loit_detector, "track_states") else None
        if loit_state:
            d_time = loit_state.dwell_duration
            d_tag = "L2" if d_time >= 60.0 else ("L1" if d_time >= 30.0 else "Normal")
            label_lines.append(f"Dwell: {d_time:.1f}s [{d_tag}]")

        y_off = max(20, box[1] - 8)
        for line in reversed(label_lines):
            color = (0, 255, 255) if "L2" in line else ((0, 200, 255) if "L1" in line else (0, 255, 180))
            cv2.putText(out, line, (box[0], y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_off -= 16

    # 3. Draw Luggage Objects & Owner Links
    ao_detector = getattr(anomaly_engine, "abandoned_object_detector", None)
    theft_detector = getattr(anomaly_engine, "theft_detector", None)
    if ao_detector and hasattr(ao_detector, "tracked_objects"):
        for obj in ao_detector.tracked_objects.values():
            bx = obj.box.astype(int)
            obj_color = (0, 0, 255) if obj.is_unattended else (255, 160, 50)
            cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), obj_color, 2)
            obj_label = f"Bag #{obj.object_id} ({obj.object_class})"
            if obj.is_unattended:
                obj_label += f" UNATTENDED {obj.unattended_duration:.1f}s"
            cv2.putText(out, obj_label, (bx[0], max(15, bx[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, obj_color, 1, cv2.LINE_AA)

            # Draw link to owner if associated
            if theft_detector and hasattr(theft_detector, "confirmed_owners"):
                owner_id = theft_detector.confirmed_owners.get(obj.object_id)
                if owner_id:
                    matching_owner = next((d for d in tracked_dets if d.track_id == owner_id), None)
                    if matching_owner is not None:
                        pcenter = (int((matching_owner.box[0] + matching_owner.box[2]) / 2), int((matching_owner.box[1] + matching_owner.box[3]) / 2))
                        bcenter = (int(obj.center[0]), int(obj.center[1]))
                        cv2.line(out, bcenter, pcenter, (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(out, f"Owner #{owner_id}", (int((bcenter[0] + pcenter[0]) / 2), int((bcenter[1] + pcenter[1]) / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # 4. Top Telemetry Banner
    cv2.rectangle(out, (10, 10), (380, 75), (20, 24, 30), -1)
    cv2.rectangle(out, (10, 10), (380, 75), (60, 70, 85), 1)
    cv2.putText(out, f"FRAME: {frame_idx:05d} | FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    ev_summary = f"Active Events: {len(events)}"
    cv2.putText(out, ev_summary, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    if events:
        latest_ev = events[-1]
        ev_title = f"{latest_ev.event_type.value} ({latest_ev.confidence*100:.0f}%)"
        cv2.putText(out, ev_title, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

    return out


def main():
    parser = argparse.ArgumentParser(description="AI Metro Anomaly Replay & HUD Tool")
    parser.add_argument("--source", type=str, required=True, help="Path to video clip or stream URL")
    parser.add_argument("--save", type=str, default=None, help="Optional output video file path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    zm = ZoneManager(cfg["zones"])
    tracker = ByteTracker(
        track_high_thresh=cfg["tracking"].get("track_high_thresh", 0.22),
        track_low_thresh=cfg["tracking"].get("track_low_thresh", 0.12),
        new_track_thresh=cfg["tracking"].get("new_track_thresh", 0.18),
        track_buffer=cfg["tracking"].get("track_buffer", 30),
        match_thresh=cfg["tracking"].get("match_thresh", 0.6)
    )

    pose_detector = None
    try:
        pose_detector = YOLOPoseDetector(
            model_name=cfg["perception"]["model_name"],
            confidence_threshold=cfg["perception"]["confidence_threshold"],
            device=cfg["perception"]["device"],
            inference_size=int(cfg["perception"].get("inference_size", 480))
        )
    except Exception as e:
        print(f"[Replay] Pose detector warning: {e}")

    anomaly_engine = AnomalyEngine(cfg)
    if pose_detector and hasattr(pose_detector, "_model") and pose_detector._model is not None:
        names = getattr(pose_detector._model, "names", {})
        if any(cls_name in names.values() for cls_name in ["backpack", "handbag", "suitcase"]):
            anomaly_engine.set_object_detector(pose_detector._model)

    rule_clf = RuleBasedEventClassifier(
        line_crossing_thresh=cfg["events"]["line_crossing_consecutive_frames"],
        track_intrusion_thresh=cfg["events"]["track_intrusion_consecutive_frames"]
    )

    source = create_frame_source(args.source, allow_fallback=False, loop_video=False)
    writer = None

    frame_idx = 0
    t_prev = time.perf_counter()
    fps = 25.0

    print(f"[Replay] Starting replay on {args.source}...")
    while True:
        ret, frame, frame_ts = source.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        if args.save and writer is None:
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))

        # Detect & track
        dets = pose_detector.detect(frame) if pose_detector else []
        tracked = tracker.update(dets)

        # Classify
        rule_evs = rule_clf.process_frame(frame_idx, frame_ts, tracked, zm, "ABSENT")
        anom_evs = anomaly_engine.process_frame(
            frame=frame,
            frame_idx=frame_idx,
            person_detections=tracked,
            zone_manager=zm,
            timestamp=frame_ts,
            train_state="ABSENT"
        )
        all_evs = rule_evs + anom_evs

        # FPS calculation
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        fps = (1.0 / dt) if dt > 0 else 25.0

        hud_frame = draw_hud(frame, zm, tracked, anomaly_engine, all_evs, fps, frame_idx)

        if writer:
            writer.write(hud_frame)

        cv2.imshow("Metro Anomaly Replay HUD (q to quit)", hud_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    if writer:
        writer.release()
        print(f"[Replay] Saved annotated video to {args.save}")
    cv2.destroyAllWindows()
    print(f"[Replay] Processed {frame_idx} frames successfully.")


if __name__ == "__main__":
    main()
