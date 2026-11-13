# Assumptions Log - AI Metro Platform Safety System

This document logs all engineering assumptions made during the design, development, and validation of the AI Metro Platform Safety System in accordance with autonomy rules.

## 1. Environment & Runtime
- **OS & Python**: Windows 11 running Python 3.11 with an isolated `.venv`. All paths utilize `pathlib.Path` for cross-platform compatibility (Windows PowerShell & Linux).
- **Inference Hardware Target**: Modest CPU-first execution. Ultralytics YOLOv8 pose models (nano: `yolov8n-pose.pt`) run via PyTorch CPU runtime, supplemented by configurable frame-skipping and resolution scaling to sustain target real-time or near real-time platforms (>15-20 FPS on CPU).
- **Synthetic Ground Truth**: Real footage of track falls/jumps on metro CCTV is rarely publicly accessible or annotated due to safety, privacy, and legal constraints. Therefore, a high-fidelity synthetic scenario generator (generating both kinematic 2D/3D skeletal trajectories and rendered visual CCTV frames with perspective zones) is used for training and formal quantitative evaluation.

## 2. Spatial Geometry & CCTV Perspective
- **Zone Structure**: CCTV cameras overlook the platform and tracks from an elevated diagonal angle. The default camera coordinate space is mapped into 4 sequential convex/concave polygons:
  1. `platform`: Safe passenger waiting and walking area.
  2. `yellow_band`: Warning buffer boundary (the tactile/yellow safety strip).
  3. `track_bed`: Danger area where tracks run.
  4. `danger_zone`: The combined yellow line edge + track bed where train collisions can occur.
- **Keypoint Ground-Plane Anchoring**: A person's spatial zone membership is determined by their ground contact points (ankles/heels when visible with confidence > 0.4; falling back to the midpoint of the bottom bounding box edge `(x_center, y_max)`).

## 3. Train State Dynamics
- Train states are modeled as `ABSENT`, `APPROACHING`, or `PRESENT`.
- Vision-based train detection uses optical flow / large moving connected foreground bounding boxes intersecting the track bed polygon.
- Alert severity for track intrusions and line crossings is dynamically boosted when `train_state == APPROACHING`.

## 4. Behavior Event Classification
- **Line Crossing**: Foot anchors crossing into `yellow_band` for at least $N_{cross} = 4$ consecutive frames.
- **Track Intrusion**: Foot/body anchors entering `track_bed` for at least $N_{track} = 3$ consecutive frames while upright or moving.
- **Accidental Fall**: Characterized by sudden downward acceleration of the torso/center-of-mass ($\Delta y / \Delta t > \tau_{drop}$), dramatic shift of torso orientation towards horizontal ($|\theta| < 35^\circ$ from horizontal), loss of upright skeletal posture without prior high-speed run or intentional crouch, ending within the track bed.
- **Deliberate Jump**: Characterized by directed horizontal approach towards the platform edge, controlled posture or purposeful crouch/leap trajectory, maintaining deliberate forward trajectory into the track bed.
- **Distress Gesture**: Detection of the international "Signal for Help" (open palm raised $\rightarrow$ thumb folded $\rightarrow$ fingers closed over thumb) tracked across temporal keypoints. If hand landmark resolution is degraded under CCTV distance/blur, the system falls back to the bilateral "Arms Crossed Overhead" (wrists crossed above head level) distress signal.

## 5. Escalation & Responsible AI Safety Controls
- **Severity Ladder**:
  - `L1`: Warning buzzer on local platform for minor line crossing.
  - `L2`: Local buzzer + Control Room audio/visual snapshot alert for person on track (train absent).
  - `L3`: Critical emergency alert + simulated Emergency Trip Switch (ETS) engagement for falls/jumps or intrusion with approaching train.
  - `DISTRESS`: Silent alert directly to Control Room/station security (no audible buzzer to protect the victim).
- **ETS Safety Interlock**: An ETS action requires explicit human confirmation within a configurable timeout window (default 5 seconds) before autonomous hard-trip, unless configured in full-auto mode.
- **Privacy**: No facial recognition, facial landmarking, biometric profiling, or persistent cross-session tracking. Track IDs are ephemeral integer tokens reset per video session.
