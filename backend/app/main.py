import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, Request, Response, Depends, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.router import api_router, get_api_key
from .config import settings
from metro_safety.demo.hub import get_global_hub

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(settings.APP_NAME)

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST prediction router
app.include_router(api_router, prefix="/api")

@app.get("/health", tags=["Health"])
def health_check():
    """Health & Readiness probe."""
    hub = get_global_hub()
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "cctv_status": hub.cctv_status,
        "source": hub.source_target or "Active",
        "is_online": hub._is_online,
        "fps": hub.live_fps
    }

def _mjpeg_generator():
    """Streams MJPEG frames from the ML LiveHub using correct hub API."""
    hub = get_global_hub()
    yield from hub.mjpeg_generator()

@app.get("/video_feed", tags=["Live Stream"])
def video_feed():
    """Live MJPEG video feed with real-time YOLOv8 pose and safety zone annotations."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/telemetry", tags=["Live Telemetry"])
def get_telemetry():
    """Live measured telemetry (FPS, end-to-end latency, train state, buzzer status)."""
    hub = get_global_hub()
    return JSONResponse(hub.get_telemetry())

@app.get("/api/events", tags=["Live Events"])
def get_events(limit: int = 50):
    """Recent safety incident event logs."""
    hub = get_global_hub()
    return JSONResponse({"events": hub.get_events(limit=limit)})

@app.post("/api/control", tags=["Operator Controls"])
async def process_control(request: Request, authorized: bool = Depends(get_api_key)):
    """Operator controls (Acknowledge buzzer, Mute, Test Beep, ETS trigger, Signal hold, Train state)."""
    hub = get_global_hub()
    try:
        body = await request.json()
        action = body.get("action", "").lower()
        value = body.get("value")

        if action == "acknowledge":
            res = hub.acknowledge_alarm()
        elif action == "mute":
            res = hub.toggle_buzzer_mute()
        elif action == "test_beep":
            pattern = str(value) if value else "L2"
            res = hub.test_buzzer(pattern=pattern)
        elif action == "ets_trip":
            res = hub.trip_ets()
        elif action == "ets_reset":
            res = hub.reset_ets()
        elif action in ["signal_hold", "hold_signal"]:
            res = hub.hold_signal()
        elif action == "set_train_state":
            if not value:
                return JSONResponse({"status": "error", "message": "value required for set_train_state"}, status_code=400)
            res = hub.set_train_state(str(value))
        else:
            return JSONResponse({"status": "error", "message": f"Unknown action: {action}"}, status_code=400)

        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Control action error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.post("/api/acknowledge", tags=["Operator Controls"])
async def api_acknowledge(authorized: bool = Depends(get_api_key)):
    """Acknowledge active audible alarm and silence buzzer."""
    hub = get_global_hub()
    return JSONResponse(hub.acknowledge_alarm())

@app.post("/api/buzzer/toggle_mute", tags=["Operator Controls"])
async def api_buzzer_toggle_mute(authorized: bool = Depends(get_api_key)):
    """Toggle hardware buzzer mute state."""
    hub = get_global_hub()
    return JSONResponse(hub.toggle_buzzer_mute())

@app.post("/api/buzzer/test", tags=["Operator Controls"])
async def api_buzzer_test(request: Request, authorized: bool = Depends(get_api_key)):
    """Dispatch operator test buzzer beep."""
    hub = get_global_hub()
    try:
        body = await request.json()
    except Exception:
        body = {}
    pattern = body.get("pattern", "L2") if isinstance(body, dict) else "L2"
    return JSONResponse(hub.test_buzzer(pattern=pattern))

@app.post("/api/ets/trigger", tags=["Operator Controls"])
async def api_ets_trigger(authorized: bool = Depends(get_api_key)):
    """Manual trigger of Emergency Trip Switch (traction power cut)."""
    hub = get_global_hub()
    return JSONResponse(hub.trip_ets())

@app.post("/api/ets/reset", tags=["Operator Controls"])
async def api_ets_reset(authorized: bool = Depends(get_api_key)):
    """Reset Emergency Trip Switch to energize traction power."""
    hub = get_global_hub()
    return JSONResponse(hub.reset_ets())

@app.post("/api/train_state", tags=["Operator Controls"])
async def api_train_state(request: Request, authorized: bool = Depends(get_api_key)):
    """Override current train state (ABSENT, APPROACHING, DOCKED, DEPARTING)."""
    hub = get_global_hub()
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict) or "state" not in body or not body["state"]:
        return JSONResponse({"status": "error", "message": "Missing or invalid train state in request body"}, status_code=400)
    from metro_safety.perception.train_state import TrainState
    state_str = str(body["state"]).strip()
    try:
        parsed_state = TrainState.from_str(state_str)
    except ValueError:
        return JSONResponse({"status": "error", "message": f"Invalid train state value: {state_str}"}, status_code=400)
    platform_id = body.get("platform_id")
    return JSONResponse(hub.set_train_state(parsed_state.value, platform_id=platform_id))

@app.get("/api/snapshot", tags=["Live Stream"])
def get_snapshot():
    """Return single JPEG snapshot of current camera feed."""
    hub = get_global_hub()
    frame_bytes = hub.get_latest_jpeg()
    if not frame_bytes:
        return Response(content=b"", status_code=503)
    return Response(content=frame_bytes, media_type="image/jpeg")

@app.get("/api/snapshots/{snapshot_id}", tags=["Live Stream"])
def get_incident_snapshot(snapshot_id: str):
    """Return single JPEG snapshot of a specific recorded incident."""
    clean_id = os.path.basename(snapshot_id).strip()
    if not clean_id.endswith(".jpg"):
        clean_id += ".jpg"
    snapshot_path = os.path.join("data", "snapshots", clean_id)
    if not os.path.isfile(snapshot_path):
        return Response(content=b"Snapshot not found", status_code=404)
    with open(snapshot_path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")

@app.post("/api/zones/confirm", tags=["Operator Controls"])
async def confirm_zones(authorized: bool = Depends(get_api_key)):
    """Operator confirmation of current or proposed zone alignment."""
    hub = get_global_hub()
    res = hub.confirm_zone_alignment()
    status_code = 200 if res.get("status") == "ok" else 400
    return JSONResponse(res, status_code=status_code)

@app.api_route("/api/zones/confirm", methods=["GET"], tags=["Operator Controls"])
async def confirm_zones_get():
    """Explicitly reject GET requests with 405 Method Not Allowed."""
    return JSONResponse(
        {"status": "error", "message": "Method Not Allowed. Use POST /api/zones/confirm."},
        status_code=405
    )

@app.post("/api/video/upload", tags=["Video Forensics"])
@app.post("/api/upload_video", tags=["Video Forensics"])
async def upload_video(
    file: UploadFile = File(...),
    register_events: bool = Form(True),
    platform_id: Optional[str] = Form(None)
):
    """Upload a platform video to run full AI inference (zones, pose skeletons, bounding boxes)
    and register detected safety incidents into the LiveHub security audit log.
    """
    try:
        from metro_safety.tools.video_processor import process_uploaded_video
        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(file.filename or "video.mp4")
        temp_path = uploads_dir / f"raw_{int(time.time()*1000)}_{safe_name}"

        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        result = process_uploaded_video(
            input_path=temp_path,
            original_filename=safe_name,
            register_to_hub=register_events,
            platform_id=platform_id
        )
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error processing uploaded video: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/video/demo/{clip_name}", tags=["Video Forensics"])
async def run_demo_clip(clip_name: str, register_events: bool = True, platform_id: Optional[str] = None):
    """Process an internal demo clip (e.g. demo_fall.mp4, demo_jump.mp4, demo_distress.mp4) without uploading."""
    try:
        from metro_safety.tools.video_processor import process_uploaded_video
        safe_name = os.path.basename(clip_name)
        if not safe_name.endswith(".mp4"):
            safe_name += ".mp4"
        clip_path = Path("demo_clips") / safe_name
        if not clip_path.exists():
            return JSONResponse({"status": "error", "message": f"Demo clip '{safe_name}' not found"}, status_code=404)
        result = process_uploaded_video(
            input_path=clip_path,
            original_filename=safe_name,
            register_to_hub=register_events,
            platform_id=platform_id
        )
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error running demo clip: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/processed_stream/{video_id}", tags=["Video Forensics"])
def get_processed_stream(video_id: str):
    """Stream annotated processed video frames as MJPEG for low-latency browser playback."""
    from metro_safety.tools.video_processor import get_cached_processed_stream
    return StreamingResponse(
        get_cached_processed_stream(video_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/processed_video/{video_id}", tags=["Video Forensics"])
def get_processed_video(video_id: str):
    """Download or stream the rendered MP4 file of the processed video."""
    safe_id = os.path.basename(video_id).strip()
    video_path = Path("data/uploads") / f"processed_{safe_id}.mp4"
    if not video_path.exists():
        return Response(content=b"Video not found", status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4", filename=f"annotated_{safe_id}.mp4")

@app.get("/api/processed_frame/{video_id}/{frame_num}", tags=["Video Forensics"])
def get_processed_frame(video_id: str, frame_num: int = 0):
    """Return a single annotated JPEG frame from the processed video."""
    from metro_safety.tools.video_processor import PROCESSED_FRAME_CACHE
    frames = PROCESSED_FRAME_CACHE.get(video_id)
    if frames and 0 <= frame_num < len(frames):
        return Response(content=frames[frame_num], media_type="image/jpeg")
    frame_path = Path("data/uploads") / f"frames_{video_id}" / f"f_{frame_num:05d}.jpg"
    if frame_path.exists():
        return Response(content=frame_path.read_bytes(), media_type="image/jpeg")
    return Response(content=b"Frame not found", status_code=404)

# Mount the enterprise frontend at /
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
