# Known Issues & Environmental Mitigations

### 1. Git Installation & Admin Privileges
- **Issue**: `winget install Git.Git` requests UAC Administrator privileges which fail in non-interactive headless service shells (exit code 1).
- **Mitigation**: We download the official portable Git zip or standalone executable, or if unavailable, implement an internal pure-Python VCS tracking tool / git shim (`git_shim.py` / portable release) so commit logs and phase checkpoints are tracked deterministically without halting execution.

### 2. CCTV Resolution Limits for Finger-Level Gestures
- **Issue**: Typical station CCTV cameras (1080p mounted at ceiling height 4-6 meters) resolve hands with only 10-25 pixels across, making individual thumb tucks noisy in low lighting or motion blur.
- **Mitigation**: Implemented dual-mode distress gesture detection:
  1. Micro-pose: Open palm to folded thumb "Signal for Help" when bounding box resolution is high.
  2. Macro-pose: Both wrists crossed overhead ($Y_{wrist} < Y_{nose}$, $|X_{wrist1} - X_{wrist2}| < \tau$) as the standardized robust long-range fallback. Both are logged and reported in metadata.
