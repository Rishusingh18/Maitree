# Known Issues & Environmental Mitigations

### 1. Git Installation & Tooling
- **Status**: Commits and version history are managed using the portable MinGit binary (`.\mingit\cmd\git.exe`, Git version 2.44.0.windows.1) rather than a pure-Python shim. MinGit runs deterministically in PowerShell without requiring elevated Windows UAC Administrator privileges.

### 2. Live Stream Performance vs Synthetic Demo Clips
- **Issue**: Performance on synthetic 640x480 demo clips (~60 FPS) does not reflect real operational throughput on full HD video.
- **Measured Reality**: On full HD (1920x1080) phone camera input (`data/recordings/phone_stream_recorded.mp4`):
  - **Display / Main Loop Rate**: **14.3 FPS**
  - **True Detector Forward Pass Rate**: **23.5 FPS** (median model forward pass: 42.6 ms on CPU with `imgsz=320`)
  - **End-to-End Latency (Capture to JPEG Publish)**: **90.1 ms median**, **162.4 ms p95**
- **Mitigation**: Decoupled asynchronous worker (`AsyncInferenceWorker`) ensures the video capture and web streaming loops remain responsive at 14+ FPS even when deep pose inference runs at 23 FPS on CPU.

### 3. Fixed Zone Polygons (No Adaptive Zones Yet)
- **Issue**: Zone polygons (`platform`, `yellow_band`, `track_bed`, `danger_zone`) in `config.yaml` are static 2D planar projections calibrated for a fixed camera position.
- **Limitation**: The system does not currently feature adaptive dynamic zone recalibration if the physical camera angle or zoom shifts. If the camera is moved, manual recalibration via `calibrate.py` is required.

### 4. Unmeasured Real-World False-Alert Rate
- **Issue**: Prior documentation claimed 0.00 false alerts/hour based on synthetic test scripts. This has not been empirically verified over continuous 24/7 live operation in high-density passenger stations.
- **Mitigation**: The false-alert rate is explicitly marked as **unmeasured in live operational conditions**. Multi-frame persistence gates ($\ge 4$ frames), displacement thresholds ($\ge 10$ px fall, $\ge 8$ px jump), and optical flow camera shake suppression are implemented to minimize potential false triggers.

### 5. Distress Gesture Tested on Synthetic Data Only
- **Issue**: Typical station CCTV cameras (1080p mounted at ceiling height 4-6 meters) resolve hands with only 10-25 pixels across, making individual thumb tucks noisy in low lighting or motion blur.
- **Limitation**: Both the micro-pose ("Signal for Help") and macro-pose ("Wrists Crossed Overhead") gesture detectors have been evaluated **strictly on synthetic generation and staged test clips**, and have not been validated on real metro crowd footage under adverse lighting or occlusions.
