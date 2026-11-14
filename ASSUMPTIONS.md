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

## 6. Live Phone Pipeline & Control Room Dashboard Integration
- **Camera Stream**: Phone camera streams over local Wi-Fi at `http://192.168.1.32:8080/video`. Threaded reader drops stale frames (buffer size 1) to eliminate video lag.
- **Hardware Buzzer**: Real audible alarm on laptop uses Windows `winsound.Beep` in an asynchronous background daemon thread so computer vision inference is never blocked. Non-Windows fallbacks emit terminal bells or silent stubs.
- **Buzzer Patterns & Clear Logic**: L1 emits a short single pulse (800Hz), L2 emits a repeating pulse (1200Hz), and L3 emits an alternating high-intensity siren (1800Hz / 2200Hz). Buzzer continues until either an operator clicks "Acknowledge" or all tracks remain clear of the danger zone for 4.0 continuous seconds.
- **Overlooker Notifications**: Windows desktop notifications are dispatched via PowerShell WinRT toast API. Outbound webhooks (ntfy.sh / Telegram) are optional and disabled by default for 100% offline security. A 10-second per-event-type cooldown prevents alert storms.
- **Telemetry Transparency**: Synthetic benchmark accuracy is explicitly labeled as "synthetic benchmark" on the dashboard, distinguishing it from live operational telemetry (live FPS, live latency).

## 7. Escalation Architecture & Robustness Bugfixes
- **Signature & Type Safety**: `EscalationEngine.evaluate` requires `(event, train_state=None, current_time=None, snapshot=None)`. Explicit runtime type checks intercept misplaced numpy arrays or invalid types with informative `TypeError` exceptions.
- **Fault-Tolerant Pipeline**: Per-frame and per-event processing in `live.py` are wrapped in granular try/except blocks with full traceback logging. Any exception in hardware buzzer, Windows notifications, webhook push, or web publishing is isolated so video processing and tracking never halt.
- **Confidence Calibration**: No model or classifier ever reports 100% (1.0) confidence; all confidences are strictly calibrated and capped at 0.99 via `SafetyEvent.__post_init__`.
- **False-Alert Gating**: Deliberate jump and accidental fall events require sustained motion across $\ge 4$ observation frames, cumulative vertical displacement ($\ge 10$ px for fall, $\ge 8$ px for jump), and multi-frame downward velocity confirmation.

## 8. Dashboard Video Feed Hardening, Fallback Visibility & CPU Tuning
- **Browser MJPEG Rendering**: Standard RFC 2046 multipart formatting (`multipart/x-mixed-replace; boundary=frame`) without part-level `Content-Length` headers avoids Chromium multipart frame-drop bugs that resulted in black screens. Condition variables and sequence numbers support multi-client concurrent consumption without frame stealing.
- **Cross-Process Shared Buffer**: Dual in-memory and disk-backed cache (`logs/cache/live_frame.jpg`, `logs/cache/live_telemetry.json`, `logs/cache/live_events.json`) ensures seamless real-time video feed and telemetry whether the dashboard runs embedded in `demo/live.py` or as a standalone process alongside `demo/run.py`.
- **Fallback Visibility & Auto-Recovery**: When primary phone stream is unreachable and falls back to a synthetic demo clip, prominent amber banners appear on both the desktop OpenCV window and the web dashboard displaying attempted URL and retry status. A background thread continually probes the primary stream and automatically restores live camera capture upon return.
- **Event Deduplication & DEMO Tagging**: A 5-second cooldown per `(track_id, event_type)` eliminates redundant duplicate triggers for ongoing incidents. Events triggered during synthetic fallback are marked with `is_demo: True` and prefixed with `[DEMO]` in the control room table and audit logs.
- **CPU Performance Optimization**: Configurable `perception.inference_size: 320` and `perception.frame_skip: 1` deliver sustained 12-18 FPS on host CPU (64x faster than unscaled 1080p), with real-time FPS and latency reported transparently.

## 9. Polygon Validation, Zone Semantics, Camera Motion Gate & Browser Verification
- **Polygon Geometric Validation**: `validate_polygon` strictly detects bowtie/self-intersecting edges using CCW segment intersection tests, rejecting polygons with $< 3$ points or tiny areas ($< 100\text{ px}^2$). Self-intersection checks precede area checks because bowtie topology can cause self-canceling contour areas. `ZoneManager` refuses invalid polygons on startup with informative errors rather than silently accepting corrupted zones.
- **Automatic Centroid Repair**: `reorder_polygon_around_centroid` orders vertices angularly by $\text{atan2}(y - c_y, x - c_x)$ around their geometric centroid, unwinding crossed/bowtie polygons into clean convex/star-shaped hulls while preserving spatial coverage. The calibration tool exposes this via hotkey `'f'`. A backup of the previous config is preserved in `config.yaml.bak`.
- **Strict Zone Semantics**: Persons positioned outside defined platform, yellow band, track bed, or danger zones return `"outside"` and are ignored by safety event classifiers. An optional `"ignore"` zone type (e.g. for benches, seating, or vending machines) is supported in `config.yaml` and drawable via calibration tools to discard false detections in non-passenger-platform areas.
- **Camera-Motion Gate**: Sparse Lucas-Kanade optical flow on background features (masking out person bounding boxes) estimates global camera translation, and Laplacian variance detects camera blur/occlusion. When global motion exceeds the configured threshold ($5\text{ px}$), events are tagged `camera_motion=True`. Fall and jump events occurring during camera motion are capped at Level 2 with a visible `"CAMERA MOVING: needs operator review"` banner and zero automated ETS emergency stop actuations. Position-based track intrusions remain unaffected unless camera misalignment is flagged.
- **Decoupled Asynchronous Inference**: `AsyncInferenceWorker` runs YOLO deep neural network inference on a dedicated background thread, buffering the latest frame and returning detections without blocking the main video capture, tracking, and rendering loops. This enables the main loop to run at 25-30+ FPS while CPU inference runs at 15-20 FPS.
- **Real Browser Verification**: Playwright Chromium runs end-to-end against `http://localhost:5000/`, validating that the video `<img>` renders real frames (`naturalWidth > 0`, `naturalHeight > 0`), that canvas pixel data changes across consecutive seconds (`snap1 != snap2`), and that no video error overlays are shown. Streaming requests against `/video_feed` verify RFC 2046 multipart frames and non-black pixel values ($\text{mean} > 5.0$).

## 10. Train State Agreement, Dead Backend Detection, Session Tagging & Stage Profiling
- **Single Source of Truth for Train Movement**: Control room button highlights (`#btn-train-absent`, `#btn-train-approaching`, `#btn-train-present`) are strictly driven by backend `data.train_state` from `/api/telemetry` rather than optimistic local click events. This ensures header badges, telemetry cards, and the simulation controller always remain in 100% agreement across all connected operators.
- **Dead Backend Detection & Graceful Fallback**: The client-side dashboard tracks consecutive telemetry polling failures. After $\ge 3$ consecutive failed polls ($\sim 1.2\text{ s}$), the dashboard transitions into a high-visibility dead state: displays a red `"BACKEND DISCONNECTED - data is stale"` banner with the timestamp of the last successful telemetry update, disables all interactive operator control buttons, and replaces the video view with a styled dashed `"NO SIGNAL"` placeholder. Polling continues in the background, automatically recovering and restoring full functionality as soon as the server becomes reachable.
- **Action Verification & Session Tagging**: Operator actions (Mute Buzzer, Test Beep, Test Siren, ENGAGE ETS, Reset ETS, Acknowledge Alarm) each visibly change backend state in `/api/telemetry` and publish structured audit records into `/api/events`. Each operational session generates a unique `session_id` (`SES-<uuid>`), tagging all events so historical disk records can be distinguished from current runs.
- **Granular Stage Profiling & Safe CPU Speedups**: The live pipeline tracks execution latency across 6 distinct stages (`capture`, `detect`, `track`, `motion_estimation`, `draw`, `encode`) using rolling 100-frame circular buffers to compute median and $p95$ percentiles. Safe optimizations—including optical flow decimation (evaluating sparse LK motion every 2 frames), 320px pose inference size, and JPEG quality 70—enable sustained $>60$ FPS display loop rates and $\sim 64$ FPS decoupled pose inference on CPU with $<20\text{ ms}$ end-to-end latency.

## 11. Performance Transparency, Motion Reason Consistency & Evaluation Scope
- **Camera Motion State Truth**: Camera motion reason strings strictly reflect the measured motion relative to the threshold. When displacement drops below threshold, the message transitions to `"Camera stabilizing (K/N still frames, X.X px)"` and strictly clears after $N$ still frames. It is guaranteed to never format a message claiming `disp > threshold` when `disp <= threshold`.
- **Honest Profiling Baseline**: Operational performance is benchmarked on Full HD (1920x1080) phone camera input (`data/recordings/phone_stream_recorded.mp4`). True detector inference is measured on the model forward pass (`self.detector.detect()`, median 42.6 ms / 23.5 FPS), and end-to-end latency is measured from frame capture timestamp to JPEG publish (median 90.1 ms). The 60+ FPS previously observed is explicitly documented as applying solely to synthetic 640x480 demo clips.
- **Evaluation & Benchmark Reality**: The >90% accuracy figures (100.0% GBDT, 94.4% rule-based) are synthetic kinematic test benchmarks and have not been demonstrated on real crowd footage. False-alert rate is documented as unmeasured under live operational conditions.

## 12. Adaptive Zone Tracking Architecture & Ground Rules
- **Direct Reference Registration (No Chaining)**: Every $N$ frames (default 10), live CCTV frames are registered directly against the human-calibrated reference frame (`data/reference_frame.jpg`). Frame-to-frame chaining is strictly avoided to prevent cumulative drift, mathematical accumulation of small errors, and runaway off-plane polygon distortion.
- **Reference Frame Immutability**: Saved polygons (`data/reference_polygons.json`) and `config.yaml` represent the human-set ground-truth baseline. They are never modified or rewritten at runtime. Only the interactive calibration tool is authorized to write to disk. All dynamic homography alignments and operator confirmation overrides reside strictly in memory.
- **Automatic Reference Capture on Missing Baseline**: If zone polygons exist in configuration but the reference frame image is missing, the system captures `data/reference_frame_auto.jpg` on startup, enters state `UNVERIFIED_REFERENCE` (in memory only), emits high-visibility operator warning banners, and requires operator confirmation before treating zones as verified. Video perception and detection continue uninterrupted.
- **Feature Detection & Masking**: SIFT feature detection (with automatic ORB fallback) uses CLAHE lighting equalization and downscales to 640px wide for sub-45ms execution on host CPU. Tracked person bounding boxes (with 18px spatial padding) and top HUD overlays are strictly masked out. If mask coverage exceeds 65%, registration is aborted to prevent homography distortion from crowd occlusions.
- **Spatial Zone Weighting**: Feature points located within 150px of safety zones are prioritized during RANSAC sample consensus to anchor the transformation plane to the platform floor rather than background walls or ceilings.
- **Homography Sanity Gates**: Candidate homographies must satisfy strict bounds: scale in $[0.85, 1.15]$, perspective terms $|H_{20}|, |H_{21}| \le 0.002$, non-zero determinant in $[0.40, 2.20]$, reprojection error $\le 3.0$ px, minimum inlier count $\ge 15$, and spatial spread covering at least 3 of 4 quadrants with variance $\ge 25.0$ px to reject false locks on repetitive floor tiles or window grids.
- **Temporal Smoothing & Safety Limits**: Transformed polygon vertices are constrained by exponential moving average ($\alpha = 0.25$) and a maximum correction rate of 5.0 px/second. Total drift exceeding 50.0 px breaches the safety threshold, halting automatic adaptation and requiring operator confirmation. Every transformed polygon is re-validated through `validate_polygon` prior to atomic swap.
- **Unverified Hysteresis & Storm Suppression**: Transition to `ZONES_UNVERIFIED` requires $N = 5$ consecutive invalid estimates. When unverified, last good zones are preserved without silent reassignment, events are tagged with `zones_unverified=True`, alerts are prepended with `[ZONES UNVERIFIED]`, and a dual cooldown keyed on `(zone, event_type)` suppresses alert storms. Recovery to `ALIGNED` requires sustained good estimates over 3.0 seconds.
- **Operator Override API**: `POST /api/zones/confirm` accepts alignment overrides in memory and audits operator identity and session ID. `GET /api/zones/confirm` is strictly rejected with HTTP 405 Method Not Allowed.




