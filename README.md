# DMRC AI Metro Platform Safety System — Integrated Master Platform

A unified Computer Vision and Machine Learning safety system built for Delhi Metro Rail Corporation (DMRC) standards.

---

## 🏛️ Integrated Architecture

```text
[ Phone CCTV Stream / IP Webcam: http://192.168.1.70:8080/video ]
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Unified Master Engine (run_system.py)                │
│                                                                        │
│  ┌──────────────────────────────┐  ┌────────────────────────────────┐  │
│  │   ML Computer Vision Engine  │  │    FastAPI Production Backend  │  │
│  │  - YOLOv8 Pose (17 Kpts)     │  │  - GET /                       │  │
│  │  - ByteTracker (Multi-Object)│  │  - GET /video_feed             │  │
│  │  - Geometric Platform Zones  │  │  - GET /api/telemetry          │  │
│  │  - Temporal State Machine    │  │  - POST /api/control           │  │
│  │  - Distress Gesture Detector │  │  - POST /api/predict/crossing  │  │
│  │  - Hardware Buzzer Alarm     │  │  - POST /api/predict/gesture   │  │
│  │  - Desktop Overlooker Alerts │  │  - POST /api/predict/behaviour │  │
│  └──────────────┬───────────────┘  └────────────────┬───────────────┘  │
│                 │                                   │                  │
│                 ▼                                   ▼                  │
│         [ Shared LiveHub ] ──────────────► [ Enterprise UI (HTML5) ]   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 One-Command Launch (Backend + ML + Frontend)

Run the unified master launcher:

```powershell
python run_system.py
```

Or specify a different camera source directly from the command line:

```powershell
python run_system.py --source "http://192.168.1.70:8080/video"
```

---

## 🌐 Live Access URLs

Once launched, access everything from your browser:

* **Enterprise Metro Command Dashboard (UI)**: [http://localhost:8000/](http://localhost:8000/)  
  *(Displays the full enterprise control room; CAM-101 streams the live AI camera feed).*
* **Live AI Stream with Keypoint & Zone Overlays**: [http://localhost:8000/video_feed](http://localhost:8000/video_feed)
* **Interactive OpenAPI Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Real-time Pipeline Telemetry API**: [http://localhost:8000/api/telemetry](http://localhost:8000/api/telemetry)
* **Health & Readiness Probe**: [http://localhost:8000/health](http://localhost:8000/health)
