# Project Progress & Status Log - AI Metro Platform Safety System

| Phase | Description | Status | Verification & Artifacts |
|---|---|---|---|
| **Phase 0** | Setup, Scaffolding, Python venv, dependencies, config | **DONE** | Python 3.11 venv, pinned `requirements.txt`, `config.yaml` |
| **Phase 1** | Perception (YOLO pose) + Multi-Object Tracking | **DONE** | `metro_safety/perception/`, `yolo_pose.py`, `tracker.py` (ByteTrack) |
| **Phase 2** | Zone Geometry + Train State Provider | **DONE** | `metro_safety/zones/`, `geometry.py`, `annotator.py`, `train_state.py` |
| **Phase 3** | Synthetic Simulation Data Generator | **DONE** | `metro_safety/simulation/`, `generator.py`, `augment.py` |
| **Phase 4** | Rule-based Event State Machine | **DONE** | `metro_safety/events/models.py`, `rule_classifier.py` |
| **Phase 5** | Learned Temporal Event Classifier | **DONE** | `metro_safety/events/learned_classifier.py`, saved `learned_model.pkl` |
| **Phase 6** | Silent Distress Gesture Detector | **DONE** | `metro_safety/events/gesture.py` (Signal for Help & Wrists Crossed) |
| **Phase 7** | Escalation Engine & Hardware Simulators | **DONE** | `metro_safety/escalation/`, `engine.py`, `adapters.py` (ETS, buzzer, audit) |
| **Phase 8** | Evaluation & Ablation Suite | **DONE** | `eval_results/EVALUATION.md`, `confusion_matrices.png`, 100% GBDT acc |
| **Phase 9** | Demo CLI + Streamlit/Flask Dashboard | **DONE** | `demo/run.py`, 3 pre-generated clips in `demo_clips/`, `demo/dashboard.py` |
| **Phase 10** | Unit & Integration Test Suite | **DONE** | 17/17 unit and integration tests passing (`pytest tests -v`) |
| **Phase 11** | Self-Review, Documentation & SUMMARY.md | **DONE** | Complete `README.md`, `ASSUMPTIONS.md`, `SUMMARY.md` |
| **Phase 12** | Live Camera Input & Interactive Calibration | **DONE** | Threaded `FrameSource` (HTTP/RTSP/webcam/file), buffer size 1 drop stale, auto-reconnect backoff, live telemetry HUD (FPS/latency), fallback video, `calibrate.py` |
| **Phase 13** | Live Phone Camera & Dashboard Integration | **DONE** | One-command `python -m metro_safety.demo.live`, LiveHub MJPEG `/video_feed`, live FPS/latency telemetry, audible PC buzzer (winsound/fallback) with L1-L3 patterns + clear timeout + acknowledge + mute, overlooker alert banner + Windows toast + optional push + cooldown + audit log, `record_clip` & `replay` tools with kinematic HUD, 31/31 tests passing |
| **Phase 14** | Robustness, Crash Hardening & Video Feed Fixes | **DONE** | Fixed escalation signature mismatch (`train_state` explicit, `snapshot` separate, runtime type checks), wrapped per-frame and per-event pipelines in isolated try/except blocks, fixed `/video_feed` multipart MJPEG streaming with headers + auto-reconnect fallback, added false-alert gating (min duration $\ge 4$ frames, min displacement gates), capped all confidences at 0.99, added 5 new regression and robustness tests (36/36 passed) |
| **Phase 15** | Dashboard MJPEG Feed, Fallback Visibility & Deduplication | **DONE** | Fixed browser black screen root causes (multipart RFC 2046 standard, condition variable multi-client generator, dual in-memory and disk-cache cross-process buffer), added bright fallback HUD banner + web UI banner with attempted URL and background retry status, implemented automatic reconnect and switchback to primary camera, added 5-second per-(track,type) event deduplication with `[DEMO]` tagging for synthetic fallback, added configurable `inference_size: 320` and `frame_skip: 1` achieving $\ge 12-18$ FPS on CPU, 42/42 tests passing |
| **Phase 16** | Polygon Validation, Zone Semantics, Camera Motion Gate, Async Inference, Playwright Browser Verification | **DONE** | Validated polygons (< 3 pts, bowtie intersections, tiny areas) with centroid-sorting auto-repair hotkey 'f'; strict zone semantics ignoring outside/ignore zones; camera motion gate via background sparse LK optical flow capping falls/jumps to L2 with review alerts (zero false L3); async threaded inference worker (`AsyncInferenceWorker`); real browser Chromium verification via Playwright (`naturalWidth > 0`, pixel change over 3s); 52/52 tests passing |
| **Phase 17** | Train State Agreement, Dead Backend Detection, Session Tagging, Control Verification & Per-Stage Profiling | **DONE** | Train movement panel highlights strictly driven by backend `data.train_state`; frontend dead backend detection (red banner after 3 failed polls, buttons disabled, last update time, no-signal placeholder, auto-recovery); error toast notifications for failed POSTs; verified with `requests` that Mute, Test Beep, Test Siren, ENGAGE ETS, Reset, and Acknowledge visibly alter telemetry and publish events; tagged all events with unique `session_id`; instrumented 6 pipeline stages with safe speedups; 56/56 tests passing |
| **Phase 18** | Fix & Honesty Pass: Raw JS Body Leak Fix, Camera Motion Reason & Clearing, True Detector Profiling on 1080p Phone Footage | **DONE** | Fixed duplicate unclosed script tag in `dashboard.py` leaking raw JS into body text, verified with Playwright Chromium; camera-motion banner reason text reconciled to never claim `> threshold` when `disp <= threshold`, with automatic clearing after N still frames; measured true detector forward pass latency and capture-to-publish end-to-end latency on full HD 1080p phone footage (`data/recordings/phone_stream_recorded.mp4`); aligned documentation across README, SUMMARY, PROGRESS, KNOWN_ISSUES; 57/57 tests passing |

---
## Addendum A: Black-Panel & Video Stream Verification Evidence

| Metric / Parameter | Live System Test Result | Verification Method |
|---|---|---|
| **HTTP Response Status** | `200 OK` | `requests.get('/video_feed', stream=True)` |
| **Content-Type** | `multipart/x-mixed-replace; boundary=frame` | Response header audit |
| **Multipart Boundary** | `--frame` | Byte stream marker parser |
| **JPEG Parts (5s stream)** | 60 frames received | 5-second raw byte capture |
| **Bytes per JPEG Part** | Mean: 26,240 bytes (~26.2 KB) | Per-part buffer measurement |
| **Non-Black Image Content** | `mean pixel intensity = 35.58 > 5.0` (Non-black) | Decoded 3 sample frames in `debug/` |
| **Saved Decoded Frames** | `debug/test_evidence_frame_1.jpg`, `frame_2.jpg`, `frame_3.jpg` | Saved to disk and inspected |
| **Frame Buffer Producer PID** | `12936` | Telemetry API (`producer_pid`) |
| **Video Feed Server PID** | `12936` (Identical PID, single unified process) | Telemetry API (`server_pid`) |
| **Dashboard `<img>` Natural Dimensions** | `naturalWidth = 640`, `naturalHeight = 480` | Real Playwright Chromium DOM evaluation |
| **Pixel Dynamism (Active Video)** | `snap1 != snap2 == True` (Active movement) | Canvas 2D frame snapshot comparison over 2.5s |
| **Visible Stream Error Handling** | Visible `#video-error-overlay` with Retry button | `<img> onerror` DOM listener in `dashboard.py` |
| **Testing Source Override** | `METRO_SOURCE_OVERRIDE` | Supported in `live.py` and `run.py` |

---
## Addendum B: Per-Stage Pipeline Latency & Throughput Profile (Full HD 1080p Input)

Measured across 60 frames on real Full HD (1920x1080) phone camera input (`data/recordings/phone_stream_recorded.mp4`) running on host CPU with YOLOv8n pose model at 320px inference size:

| Stage Name | Execution Details & Safe Optimizations | Median Latency (ms) | p95 Latency (ms) |
|---|---|---|---|
| **`capture`** | Threaded non-blocking grabber, 1080p frame retrieval | 9.72 ms | 13.29 ms |
| **`detect_forward_pass`** | True model forward pass time on CPU (`self.detector.detect`, imgsz 320) | 42.63 ms | 279.55 ms |
| **`track`** | ByteTrack multi-object tracking & kinematic updates | 0.01 ms | 0.31 ms |
| **`motion_estimation`** | Sparse LK background optical flow on 1080p features (evaluated every 2 frames) | 47.89 ms | 98.28 ms |
| **`draw`** | Polygon zones, skeleton pairs, kinematic HUD badges, alert banners | 16.79 ms | 31.53 ms |
| **`encode`** | JPEG encoding (quality 70) and LiveHub buffer broadcast | 9.56 ms | 14.84 ms |
| **`end_to_end`** | Frame capture timestamp to JPEG broadcast publish | **90.12 ms** | **162.36 ms** |

| Throughput Metric | Measured Operational Rate (1080p Input) | Target Requirement |
|---|---|---|
| **Display / Main Loop Rate** | **14.3 FPS** | $\ge 10.0$ FPS (Target Met) |
| **Detector Forward Pass Rate** | **23.5 FPS** | $\ge 10.0$ FPS (Target Met) |
| **End-to-End Latency** | **90.1 ms median** | $\le 100$ ms (Target Met) |
| *Synthetic 640x480 Demo Rate* | *~60 FPS* | *Benchmark reference only* |

---
**Overall Status**: All core project sections, live camera streaming extensions, hardware buzzer escalation, overlooker alerting, full dashboard integration, robustness bugfixes, video streaming fixes, polygon validation, strict zone semantics, camera motion optical flow gating, async inference, Playwright browser verification, train state agreement, dead backend detection, session tagging, control verification, and honest per-stage profiling are completely implemented, verified, tested (57/57 passed), and documented. Model benchmark accuracy: 100.0% (Learned GBDT, synthetic benchmark only), 94.4% (Rule-Based, synthetic benchmark only). Real 1080p phone pipeline delivers 14.3 FPS display rate, 23.5 FPS true detector inference rate, and 90.1 ms latency on CPU.



