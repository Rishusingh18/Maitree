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

---

## 🔍 Anomaly, Welfare & Theft Detection Subsystems

The platform includes autonomous anomaly detection modules designed as **operator welfare review alerts** (strictly capped at Level 2 severity, zero automated Emergency Trip Switch actuation, anonymous track IDs):

1. **`PERSON_DOWN` (Passenger Welfare Alert)**: Detects persons lying horizontally on the platform/ground (low head-to-hip vertical span, torso angle $< 35^\circ$ from horizontal, wide aspect ratio $> 1.15$, low motion energy). Distinguishes temporary crouching from unresponsive collapse ($T_1 = 10\text{s} \rightarrow \text{L1 Notice}$, $T_2 = 30\text{s} \rightarrow \text{L2 Welfare Alert}$). Spatial-temporal re-association maintains timers across tracking ID churn.
2. **`LOITERING` (Extended Dwell Monitor)**: Monitors stationary dwell inside critical zones beyond configurable thresholds ($T_1 = 30\text{s}$, $T_2 = 60\text{s}$). Automatically suppresses dwell alerts during train approach, docking, and passenger boarding.
3. **`ABANDONED_OBJECT` (Unattended Luggage)**: Decimated YOLO luggage detection (backpack, handbag, suitcase) runs at configurable frame intervals (default 10) to conserve CPU. Tracks bags, detects owner separation ($> 130\text{ px}$ for $> 15\text{s}$), and stores timestamped snapshot crops.
4. **`POSSIBLE_SNATCH` (Theft Prototype)**: Tracks bag-owner association via sustained wrist/body proximity. Detects opportunistic snatching when a second person's wrist approaches the bag, the bag accelerates rapidly away ($> 10\text{ px/frame}$), and the owner does not follow. Suppresses consensual handovers and walking companions. Confidence capped at $0.70$ and labeled *"possible, operator review"*.
5. **`STG-NF` (Spatio-Temporal Graph Normalizing Flows)**: Pretrained ICCV 2023 normalizing flow network scoring rolling 24-frame pose sequences via negative log-likelihood (NLL).

### Tools & Utilities

* **HUD Video Replay Tool**:
  ```powershell
  python tools/replay.py --source demo_clips/demo_fall.mp4
  ```
* **Anomaly Evaluation Harness**:
  ```powershell
  python tools/eval_anomaly.py
  ```
