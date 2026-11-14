import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import StreamingResponse, JSONResponse
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

# Mount the enterprise frontend at /
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
