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
import uvicorn

from metro_safety.config import load_config
from metro_safety.demo.hub import get_global_hub
from metro_safety.demo.run import run_pipeline
from backend.app.main import app as fastapi_app


def run_vision_loop(stop_event: threading.Event, source_uri: str = None, config_path: str = None):
    """Runs the ML Computer Vision inference loop in a background thread."""
    cfg = load_config(config_path)
    source = source_uri or cfg["system"].get("source", "http://192.168.1.70:8080/video")
    print(f"[ML Pipeline] Initializing camera source: {source}")

    try:
        run_pipeline(
            source_uri=source,
            config_path=config_path,
            show_window=False,
            loop_video=True,
            stop_event=stop_event
        )
    except Exception as e:
        print(f"[ML Pipeline] Error in vision loop: {e}")
    finally:
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
