"""Single-Command Master Launcher for DMRC Metro Platform Safety System.

Integrates:
1. ML Computer Vision Pipeline (reading phone CCTV stream at config.yaml with auto-fallback)
2. FastAPI Production Backend on http://localhost:8000
3. Enterprise Frontend Control Room Dashboard at http://localhost:8000/
4. Live AI MJPEG Video Feed with Pose & Zone Overlays at http://localhost:8000/video_feed
5. Hardware Buzzer & Real-Time Alert Engine
"""
import os
import sys
import time
import argparse
import threading
import numpy as np
import uvicorn
import cv2

from metro_safety.config import load_config
from metro_safety.demo.hub import get_global_hub, set_global_hub, LiveHub
from metro_safety.escalation.buzzer import HardwareBuzzer
from metro_safety.perception.train_state import TrainState, SimulatedTrainStateProvider
from metro_safety.escalation.overlooker import OverlookerNotifier
from metro_safety.zones.geometry import ZoneManager
from metro_safety.tracking.tracker import ByteTracker
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.learned_classifier import LearnedEventClassifier
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.perception.yolo_pose import YOLOPoseDetector
from metro_safety.perception.camera_motion import CameraMotionDetector
from metro_safety.perception.async_inference import AsyncInferenceWorker
from metro_safety.escalation.engine import EscalationEngine, SeverityLevel
from metro_safety.perception.frame_source import create_frame_source, VideoFileSource
from backend.app.main import app as fastapi_app

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def run_vision_loop(stop_event: threading.Event, source_uri: str = None, config_path: str = None):
    """Runs the ML Computer Vision inference loop in a background thread."""
    cfg = load_config(config_path)
    hub = get_global_hub()

    source = source_uri or cfg["system"].get("source", "http://192.168.1.70:8080/video")
    print(f"[ML Pipeline] Initializing camera source: {source}")

    buzzer_cfg = cfg.get("buzzer", {})
    buzzer = HardwareBuzzer(audio_enabled=buzzer_cfg.get("audio_enabled", True))
    train_provider = SimulatedTrainStateProvider(initial_state=TrainState(cfg["train_state"]["default_state"]))
    hub.set_hardware_components(buzzer=buzzer, train_state_provider=train_provider)

    overlooker = OverlookerNotifier(dashboard_hub=hub, audit_log_file="logs/overlooker_audit.log")

    zm = ZoneManager(cfg["zones"])
    tracker = ByteTracker()
    rule_clf = RuleBasedEventClassifier()
    learned_clf = LearnedEventClassifier()
    gesture_det = DistressGestureDetector()
    yolo_model = YOLOPoseDetector(model_name="yolov8n-pose.pt")
    motion_det = CameraMotionDetector()
    async_inference = AsyncInferenceWorker(yolo_model, max_queue_size=1)
    async_inference.start()

    escalation = EscalationEngine(
        on_buzzer_l1=buzzer.play_l1_pulse,
        on_buzzer_l2=buzzer.play_l2_alarm,
        on_buzzer_l3=buzzer.play_l3_siren,
        on_buzzer_clear=buzzer.stop,
        on_ets_trigger=lambda: hub.trip_ets(),
        on_silent_alert=overlooker.notify_distress_gesture
    )

    frame_source = create_frame_source(
        source_uri=source,
        fallback_uri="demo_clips/demo_fall.mp4",
        loop_video=True
    )

    # Set initial source state on hub using correct attributes
    hub.source_target = source
    hub.is_fallback = getattr(frame_source, 'is_fallback', False)

    print("[ML Pipeline] YOLOv8 Pose & Zone Analysis active. Streaming to LiveHub.")
    t_prev = time.perf_counter()

    try:
        while not stop_event.is_set():
            ret, frame, ts = frame_source.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            t_now = time.perf_counter()
            fps = 1.0 / max(t_now - t_prev, 1e-5)
            t_prev = t_now

            # Motion & Inference
            is_moving, motion_reason = motion_det.check_motion(frame)
            async_inference.submit_frame(frame)
            detections = async_inference.get_latest_detections(timeout=0.04) or []

            tracks = tracker.update(detections) if not is_moving else []

            # Safety events
            active_events = []
            for track in tracks:
                zone_cat = zm.classify_person_location(track.keypoints or np.empty((0, 2)), track.bbox)
                events = rule_clf.update(track, zone_cat, train_provider.get_state())
                active_events.extend(events)

            highest_sev = SeverityLevel.NONE
            for ev in active_events:
                escalation.handle_event(ev)
                if ev.severity.value > highest_sev.value:
                    highest_sev = ev.severity
                # Push event to hub
                hub.push_event({
                    "id": f"EV-{int(time.time()*1000)}",
                    "timestamp": time.time(),
                    "time_str": time.strftime("%H:%M:%S", time.localtime()),
                    "track_id": getattr(ev, 'track_id', 0),
                    "event_type": getattr(ev, 'event_type', str(ev)),
                    "zone": str(zone_cat),
                    "severity": str(ev.severity.name) if hasattr(ev, 'severity') else "UNKNOWN",
                    "confidence": float(getattr(ev, 'confidence', 0.0)),
                    "action": str(getattr(ev, 'description', '')),
                    "is_demo": False,
                    "session_id": hub.session_id
                })

            # Annotate frame
            annotated = zm.draw_zones(frame.copy(), alpha=0.35)
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if track.keypoints is not None:
                    for pt in track.keypoints:
                        if pt[0] > 0 and pt[1] > 0:
                            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            # Latency calculation
            latency_ms = (time.perf_counter() - t_now) * 1000

            # Publish frame to hub (correct API)
            hub.update_frame(annotated, is_online=True)

            # Publish telemetry to hub (correct API)
            hub.update_telemetry(
                fps=fps,
                latency_ms=latency_ms,
                detector_fps=getattr(async_inference, 'measured_fps', fps),
                is_fallback=getattr(frame_source, 'is_fallback', False),
                source_target=source,
                camera_motion=is_moving,
                camera_motion_reason=motion_reason if is_moving else "Stable"
            )

    finally:
        async_inference.stop()
        buzzer.stop()
        frame_source.release()
        print("[ML Pipeline] Cleanly shut down.")

def main():
    parser = argparse.ArgumentParser(description="DMRC AI Metro Platform Safety System - Master Launcher")
    parser.add_argument("--source", type=str, default=None, help="CCTV stream or video file")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default 8000)")
    args = parser.parse_args()

    stop_event = threading.Event()

    # Start ML pipeline in daemon thread
    ml_thread = threading.Thread(
        target=run_vision_loop,
        args=(stop_event, args.source),
        daemon=True,
        name="MLVisionPipelineThread"
    )
    ml_thread.start()

    print("\n" + "=" * 70)
    print("  🚇 DMRC AI METRO PLATFORM SAFETY SYSTEM — INTEGRATED PLATFORM")
    print(f"  • Enterprise Dashboard:    http://localhost:{args.port}/")
    print(f"  • Live CCTV Video Feed:    http://localhost:{args.port}/video_feed")
    print(f"  • Swagger API Docs:        http://localhost:{args.port}/docs")
    print(f"  • Health & Readiness:      http://localhost:{args.port}/health")
    print(f"  • Camera Input:            {args.source or 'Phone CCTV (config.yaml)'}")
    print("=" * 70 + "\n")

    # Start FastAPI server on main thread
    try:
        uvicorn.run(fastapi_app, host="0.0.0.0", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("\n[System] Shutting down...")
    finally:
        stop_event.set()
        ml_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()
