"""Adaptive Zone Tracking and Frame-to-Reference Registration.
Registers live CCTV frames directly against the human-calibrated reference frame:
- Direct reference registration (never chained frame-to-frame).
- SIFT detector with ORB fallback, CLAHE lighting equalization, 640px downscaling.
- Person-box and HUD masking, spatial zone weighting.
- Rigorous homography sanity checks (scale, perspective, determinant, reprojection, 4-quadrant spread, repetitive texture rejection).
- Temporal smoothing, max drift limit, max correction rate, atomic zone set swap.
- Hysteresis state machine (ALIGNED, DRIFTING, ZONES_UNVERIFIED, UNVERIFIED_REFERENCE, DISABLED_DEMO).
- Non-blocking background worker thread with strict time budget.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import copy
import json
import logging
import math
import threading
import time
import cv2
import numpy as np

from metro_safety.zones.geometry import ZoneManager, validate_polygon

logger = logging.getLogger("metro_safety.zones.registration")


class AlignmentStatus(str, Enum):
    ALIGNED = "ALIGNED"
    DRIFTING = "DRIFTING"
    ZONES_UNVERIFIED = "ZONES UNVERIFIED"
    UNVERIFIED_REFERENCE = "UNVERIFIED_REFERENCE"
    DISABLED_DEMO = "DISABLED-DEMO"


@dataclass
class RegistrationResult:
    status: AlignmentStatus
    homography: Optional[np.ndarray] = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    confidence: float = 0.0
    drift_px: float = 0.0
    scale: float = 1.0
    reg_time_ms: float = 0.0
    is_valid: bool = False
    failure_reason: str = ""
    transformed_zones: Optional[Dict[str, List[List[float]]]] = None
    zone_version: int = 1
    requires_operator_confirmation: bool = False
    zones_unverified: bool = False


class AdaptiveZoneTracker:
    """Manages robust, non-blocking frame-to-reference homography registration
    and updates ZoneManager polygons with strict safety bounds.
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        config: Optional[Dict[str, Any]] = None,
        reference_image_path: Optional[str] = None
    ):
        self.zone_manager = zone_manager
        cfg = config or {}
        self.adaptive_cfg = cfg.get("adaptive_zones", {})

        self.enabled: bool = self.adaptive_cfg.get("enabled", True)
        self.interval_frames: int = int(self.adaptive_cfg.get("registration_interval_frames", 5))
        self.time_budget_ms: float = float(self.adaptive_cfg.get("time_budget_ms", 60.0))
        self.max_drift_px: float = float(self.adaptive_cfg.get("max_drift_px", 50.0))
        self.max_correction_rate_px_per_sec: float = float(self.adaptive_cfg.get("max_correction_rate_px_per_sec", 30.0))
        self.smoothing_alpha: float = float(self.adaptive_cfg.get("smoothing_alpha", 0.45))
        self.min_scale: float = float(self.adaptive_cfg.get("min_scale", 0.85))
        self.max_scale: float = float(self.adaptive_cfg.get("max_scale", 1.15))
        self.min_inliers: int = int(self.adaptive_cfg.get("min_inliers", 15))
        self.max_reproj_error: float = float(self.adaptive_cfg.get("max_reproj_error", 3.0))
        self.max_consecutive_bad: int = int(self.adaptive_cfg.get("max_consecutive_bad_estimates", 5))
        self.recovery_good_sec: float = float(self.adaptive_cfg.get("recovery_good_seconds", 3.0))
        self.audit_log_path: Path = Path(self.adaptive_cfg.get("audit_log_file", "logs/safety_audit.log"))

        # Reference image path with multi-path fallback
        self.ref_path_str: str = reference_image_path or self.adaptive_cfg.get("reference_frame", "data/reference_frame.jpg")
        self.ref_path = Path(self.ref_path_str)
        if not self.ref_path.exists():
            for candidate in [Path("data/reference_frame.jpg"), Path("data/calibrated_frame.jpg"), Path("data/reference_frame_auto.jpg")]:
                if candidate.exists():
                    self.ref_path = candidate
                    self.ref_path_str = str(candidate)
                    break

        # Baseline human-calibrated polygons (NEVER modified at runtime)
        self.baseline_zones: Dict[str, List[List[float]]] = copy.deepcopy(self.zone_manager.baseline_zones)
        self.current_zones: Dict[str, List[List[float]]] = copy.deepcopy(self.baseline_zones)
        self.last_good_zones: Dict[str, List[List[float]]] = copy.deepcopy(self.baseline_zones)
        self.zone_version: int = getattr(self.zone_manager, "zone_version", 1)

        # State tracking
        self.status = AlignmentStatus.ALIGNED
        self.consecutive_bad_estimates: int = 0
        self.consecutive_good_frames: int = 0
        self.last_correction_time: float = time.time()
        self.current_drift_px: float = 0.0
        self.current_scale: float = 1.0
        self.requires_operator_confirmation: bool = False
        self.drift_breached: bool = False

        # In-memory proposal for operator confirmation
        self.proposed_homography: Optional[np.ndarray] = None
        self.proposed_zones: Optional[Dict[str, List[List[float]]]] = None

        # Feature matching resources
        self._sift = None
        self._orb = None
        self._init_detector()

        # CLAHE for lighting invariance
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Reference frame data
        self.reference_frame: Optional[np.ndarray] = None
        self.ref_gray_640: Optional[np.ndarray] = None
        self.ref_kps = None
        self.ref_des = None
        self.ref_scale_factor: float = 1.0
        self.ref_w: int = 1920
        self.ref_h: int = 1080

        # Worker thread synchronization & state tracking
        self._lock = threading.RLock()
        self._job_event = threading.Event()
        self._stop_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_boxes: List[np.ndarray] = []
        self._pending_ts: float = 0.0
        self._pending_idx: int = 0
        self._mode_generation: int = 0
        self._recovery_start_time: Optional[float] = None
        self.accepted_alignment_zones: Dict[str, List[List[float]]] = copy.deepcopy(self.baseline_zones)

        # Check reference frame on startup
        if self.ref_path.exists():
            self._load_reference_frame(str(self.ref_path))
        else:
            self.status = AlignmentStatus.UNVERIFIED_REFERENCE
            self.requires_operator_confirmation = True
            logger.warning(
                f"[AdaptiveZones] Reference frame missing at {self.ref_path}. "
                f"State initialized to UNVERIFIED_REFERENCE. Operator verification required."
            )

        self.latest_result = RegistrationResult(
            status=self.status,
            drift_px=0.0,
            scale=1.0,
            zone_version=self.zone_version,
            requires_operator_confirmation=self.requires_operator_confirmation,
            zones_unverified=(self.status != AlignmentStatus.ALIGNED)
        )

        # Start background registration worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AdaptiveZoneRegistrationWorker"
        )
        self._worker_thread.start()

    def _init_detector(self):
        """Initialize SIFT with ORB fallback."""
        try:
            if hasattr(cv2, "SIFT_create"):
                self._sift = cv2.SIFT_create()
                self._detector_type = "SIFT"
            else:
                self._orb = cv2.ORB_create(nfeatures=2000)
                self._detector_type = "ORB"
        except Exception:
            self._orb = cv2.ORB_create(nfeatures=2000)
            self._detector_type = "ORB"

    def _load_reference_frame(self, path: str):
        """Load and preprocess reference frame."""
        frame = cv2.imread(path)
        if frame is None:
            logger.error(f"[AdaptiveZones] Failed to load reference frame from {path}")
            with self._lock:
                self.status = AlignmentStatus.UNVERIFIED_REFERENCE
                self.requires_operator_confirmation = True
            return
        self.set_reference_frame(frame)

    def set_reference_frame(self, frame: np.ndarray):
        """Set reference frame in memory and compute reference keypoints."""
        with self._lock:
            self.reference_frame = frame.copy()
            self.ref_h, self.ref_w = frame.shape[:2]

            # Scale to 640px wide for feature detection
            target_w = 640.0
            self.ref_scale_factor = target_w / float(self.ref_w)
            target_h = int(self.ref_h * self.ref_scale_factor)
            resized = cv2.resize(frame, (int(target_w), target_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            self.ref_gray_640 = self._clahe.apply(gray)

            # Mask top HUD area on reference
            mask = np.ones_like(self.ref_gray_640, dtype=np.uint8) * 255
            hud_h = int(75 * self.ref_scale_factor)
            mask[:hud_h, :] = 0

            if self._detector_type == "SIFT" and self._sift is not None:
                self.ref_kps, self.ref_des = self._sift.detectAndCompute(self.ref_gray_640, mask)
            elif self._orb is not None:
                self.ref_kps, self.ref_des = self._orb.detectAndCompute(self.ref_gray_640, mask)

            if self.ref_des is None or len(self.ref_des) == 0:
                self.status = AlignmentStatus.UNVERIFIED_REFERENCE
                self.requires_operator_confirmation = True
                logger.warning("[AdaptiveZones] Reference frame yielded no descriptors; state set to UNVERIFIED_REFERENCE.")
            logger.info(
                f"[AdaptiveZones] Reference frame initialized ({self.ref_w}x{self.ref_h}). "
                f"Features detected: {len(self.ref_kps) if self.ref_kps else 0}"
            )

    def on_first_frame(self, frame: np.ndarray):
        """Called on the first frame if reference frame was absent at startup."""
        if self.reference_frame is not None:
            return

        with self._lock:
            ref_file = Path("data/reference_frame.jpg")
            auto_path = Path("data/reference_frame_auto.jpg")
            ref_file.parent.mkdir(parents=True, exist_ok=True)
            if not ref_file.exists():
                cv2.imwrite(str(ref_file), frame)
            cv2.imwrite(str(auto_path), frame)
            self.ref_path = ref_file if ref_file.exists() else auto_path
            self.ref_path_str = str(self.ref_path)
            self.set_reference_frame(frame)
            self.status = AlignmentStatus.ALIGNED
            self.requires_operator_confirmation = False
            logger.info(
                f"[AdaptiveZones] Reference frame auto-captured to {self.ref_path}. "
                f"Tracking initialized and active."
            )

    def set_demo_mode(self, is_demo: bool):
        """Disable registration on synthetic demo/fallback and reset state on switch."""
        with self._lock:
            current_is_demo = getattr(self, "_is_demo_mode", None)
            if current_is_demo == is_demo:
                return
            self._is_demo_mode = is_demo
            self._mode_generation += 1
            self._pending_frame = None
            self._pending_boxes = []
            self._recovery_start_time = None
            self.proposed_zones = None
            self.proposed_homography = None
            if is_demo:
                self.status = AlignmentStatus.DISABLED_DEMO
            else:
                if self.status == AlignmentStatus.DISABLED_DEMO:
                    prior_req = getattr(self, "requires_operator_confirmation", False)
                    if not self.ref_path.exists() or self.reference_frame is None or prior_req:
                        self.status = AlignmentStatus.UNVERIFIED_REFERENCE
                        self.requires_operator_confirmation = True
                    else:
                        self.status = AlignmentStatus.ALIGNED
                        self.requires_operator_confirmation = False
                    self.consecutive_bad_estimates = 0
                    self.consecutive_good_frames = 0
                    self.current_drift_px = 0.0
                    self.current_scale = 1.0
                    self.current_zones = copy.deepcopy(self.baseline_zones)
                    self.accepted_alignment_zones = copy.deepcopy(self.baseline_zones)
                    self.zone_manager.update_zones(self.current_zones)

    def submit_frame(self, frame: np.ndarray, frame_idx: int, tracked_boxes: List[np.ndarray], timestamp: float):
        """Submit frame to background worker non-blockingly."""
        if not self.enabled or self.status == AlignmentStatus.DISABLED_DEMO:
            return
        if self.reference_frame is None or self.ref_des is None:
            return
        if frame_idx % self.interval_frames != 0:
            return

        # Try to acquire lock non-blockingly; if worker is still running previous frame, skip
        if not self._lock.acquire(blocking=False):
            return

        try:
            self._pending_frame = frame.copy()
            self._pending_boxes = [b.copy() for b in tracked_boxes] if tracked_boxes else []
            self._pending_ts = timestamp
            self._pending_idx = frame_idx
            self._job_event.set()
        finally:
            self._lock.release()

    def get_latest_result(self) -> RegistrationResult:
        """Fetch latest registration telemetry atomically."""
        with self._lock:
            res = copy.copy(self.latest_result)
            res.status = self.status
            res.requires_operator_confirmation = self.requires_operator_confirmation
            res.zones_unverified = (self.status != AlignmentStatus.ALIGNED)
            res.zone_version = self.zone_version
            return res

    def _worker_loop(self):
        """Background thread executing direct frame-to-reference registration within time budget."""
        while not self._stop_event.is_set():
            if not self._job_event.wait(timeout=0.1):
                continue
            self._job_event.clear()

            with self._lock:
                frame = self._pending_frame
                boxes = self._pending_boxes
                ts = self._pending_ts
                idx = self._pending_idx
                job_gen = self._mode_generation
                self._pending_frame = None

            if frame is None:
                continue

            try:
                t0 = time.perf_counter()
                reg_result = self._register_direct(frame, boxes, ts)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                reg_result.reg_time_ms = round(dt_ms, 2)
                if dt_ms > self.time_budget_ms:
                    reg_result.is_valid = False
                    reg_result.failure_reason = f"Registration exceeded time budget ({dt_ms:.1f}ms > {self.time_budget_ms}ms)"

                with self._lock:
                    self._apply_result_with_safety_limits(reg_result, ts, generation=job_gen)
                    self.latest_result = reg_result
            except Exception as ex:
                logger.error(f"[AdaptiveZones] Error in registration worker step: {ex}")

    def _register_direct(self, frame: np.ndarray, boxes: List[np.ndarray], timestamp: float) -> RegistrationResult:
        """Register live frame directly against the reference frame."""
        h, w = frame.shape[:2]
        target_w = 640.0
        scale_f = target_w / float(w)
        target_h = int(h * scale_f)
        resized = cv2.resize(frame, (int(target_w), target_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_eq = self._clahe.apply(gray)

        # Build feature exclusion mask: HUD + padded person boxes
        mask = np.ones((target_h, int(target_w)), dtype=np.uint8) * 255
        hud_h = int(75 * scale_f)
        mask[:hud_h, :] = 0

        pad = int(18 * scale_f)
        for b in boxes:
            x1 = max(0, int(b[0] * scale_f) - pad)
            y1 = max(0, int(b[1] * scale_f) - pad)
            x2 = min(int(target_w), int(b[2] * scale_f) + pad)
            y2 = min(target_h, int(b[3] * scale_f) + pad)
            mask[y1:y2, x1:x2] = 0

        # Check mask coverage: if masked area > 65%, abort to avoid false homography
        unmasked_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if unmasked_ratio < 0.35:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason=f"Frame occluded ({int((1.0 - unmasked_ratio)*100)}% masked)",
                is_valid=False
            )

        # Detect features on live frame
        if self._detector_type == "SIFT" and self._sift is not None:
            live_kps, live_des = self._sift.detectAndCompute(gray_eq, mask)
        elif self._orb is not None:
            live_kps, live_des = self._orb.detectAndCompute(gray_eq, mask)
        else:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason="No feature detector available",
                is_valid=False
            )

        if live_des is None or len(live_kps) < self.min_inliers or self.ref_des is None:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason=f"Insufficient features detected ({len(live_kps) if live_kps else 0})",
                is_valid=False
            )

        # Match features against reference frame using k-NN with Lowe's ratio test
        norm_type = cv2.NORM_L2 if self._detector_type == "SIFT" else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm_type)
        try:
            matches = matcher.knnMatch(self.ref_des, live_des, k=2)
        except Exception as e:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason=f"Matching error: {e}",
                is_valid=False
            )

        good_matches = []
        ambiguous_matches = 0
        for m_tuple in matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                if m.distance > 0.85 * n.distance:
                    ambiguous_matches += 1

        if len(good_matches) < self.min_inliers:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason=f"Insufficient good matches ({len(good_matches)} < {self.min_inliers})",
                is_valid=False
            )

        # Check repetitive texture: if ratio of ambiguous matches is excessively high (>75%)
        if len(matches) > 30 and (ambiguous_matches / float(len(matches))) > 0.78:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason="Repetitive texture detected (tile/grid ambiguity)",
                is_valid=False
            )

        # Extract points in original (full resolution) coordinate space
        src_pts = np.float32([self.ref_kps[m.queryIdx].pt for m in good_matches]) / self.ref_scale_factor
        dst_pts = np.float32([live_kps[m.trainIdx].pt for m in good_matches]) / scale_f

        # Spatially weight points near baseline zones higher
        src_pts, dst_pts, good_matches = self._weight_points_near_zones(src_pts, dst_pts, good_matches)

        # RANSAC homography
        H, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.max_reproj_error,
            maxIters=2000
        )

        if H is None or inlier_mask is None:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason="RANSAC failed to find homography",
                is_valid=False
            )

        inliers = int(inlier_mask.sum())
        inlier_ratio = float(inliers) / float(len(good_matches)) if good_matches else 0.0

        # Perform comprehensive homography sanity checks
        valid, reason, metrics = self._sanity_check_homography(H, dst_pts, inlier_mask, w, h)
        if not valid:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                homography=H,
                inlier_count=inliers,
                inlier_ratio=inlier_ratio,
                scale=metrics.get("scale", 1.0),
                failure_reason=reason,
                is_valid=False
            )

        # Compute drift against baseline polygons
        drift_px, transformed = self._compute_drift_and_transformed_zones(H)
        if not transformed:
            return RegistrationResult(
                status=AlignmentStatus.ZONES_UNVERIFIED,
                failure_reason="Perspective transform failed to project polygons",
                is_valid=False
            )

        confidence = min(0.99, max(0.50, 0.60 + 0.35 * inlier_ratio))

        return RegistrationResult(
            status=AlignmentStatus.ALIGNED if drift_px < 5.0 else AlignmentStatus.DRIFTING,
            homography=H,
            inlier_count=inliers,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            drift_px=round(drift_px, 2),
            scale=round(metrics.get("scale", 1.0), 3),
            is_valid=True,
            transformed_zones=transformed,
            zone_version=self.zone_version + 1
        )

    def _weight_points_near_zones(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        matches: list
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prioritize feature points near safety zones to prevent off-plane drift."""
        # Find points within 150px of any baseline polygon
        weighted_indices = []
        regular_indices = []
        for i, pt in enumerate(src_pts):
            near = False
            for zn, zpts in self.baseline_zones.items():
                poly_arr = np.array(zpts, dtype=np.int32)
                dist = cv2.pointPolygonTest(poly_arr, (float(pt[0]), float(pt[1])), measureDist=True)
                if dist >= -150.0:  # inside or within 150px outside
                    near = True
                    break
            if near:
                weighted_indices.append(i)
            else:
                regular_indices.append(i)

        if len(weighted_indices) >= self.min_inliers:
            # Duplicate zone-proximate features so uniform RANSAC sampling gives them genuine sampling priority
            order = weighted_indices * 2 + regular_indices
            return src_pts[order], dst_pts[order], [matches[idx] for idx in order]
        return src_pts, dst_pts, matches

    def _sanity_check_homography(
        self,
        H: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray,
        w: int,
        h: int
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Homography sanity checks:
        1. Scale in [min_scale, max_scale]
        2. Small perspective terms
        3. Determinant away from zero
        4. 4-quadrant inlier distribution
        5. Repetitive texture collinearity
        6. Inlier count
        """
        # Scale & Determinant
        det_val = float(np.linalg.det(H))
        if det_val <= 0.40 or det_val >= 2.20:
            return False, f"Determinant out of bounds ({det_val:.3f})", {"scale": 1.0}

        scale_x = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        scale_y = math.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        scale = math.sqrt(abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]))

        if not (self.min_scale <= scale <= self.max_scale):
            return False, f"Scale out of bounds ({scale:.3f} not in [{self.min_scale}, {self.max_scale}])", {"scale": scale}
        if abs(scale_x - scale_y) > 0.25:
            return False, f"Non-uniform aspect distortion (|sx-sy|={abs(scale_x-scale_y):.3f})", {"scale": scale}

        # Perspective terms
        if abs(H[2, 0]) > 0.002 or abs(H[2, 1]) > 0.002:
            return False, f"Excessive perspective distortion ({abs(H[2, 0]):.5f}, {abs(H[2, 1]):.5f})", {"scale": scale}

        # Inlier count
        raw_inliers = dst_pts[inlier_mask.ravel() == 1]
        unique_inliers = np.unique(raw_inliers, axis=0) if len(raw_inliers) > 0 else raw_inliers
        if len(raw_inliers) < self.min_inliers:
            return False, f"Too few inliers ({len(raw_inliers)} < {self.min_inliers})", {"scale": scale}

        # Quadrant spread: Inliers must spread across at least 3 of 4 quadrants
        cx, cy = w / 2.0, h / 2.0
        q_counts = [0, 0, 0, 0]  # TL, TR, BL, BR
        for pt in unique_inliers:
            x, y = pt[0], pt[1]
            if x < cx and y < cy:
                q_counts[0] += 1
            elif x >= cx and y < cy:
                q_counts[1] += 1
            elif x < cx and y >= cy:
                q_counts[2] += 1
            else:
                q_counts[3] += 1

        active_quadrants = sum(1 for c in q_counts if c >= 2)
        if active_quadrants < 3:
            return False, f"Inliers clustered in {active_quadrants}/4 quadrants (must cover >= 3)", {"scale": scale}

        # False lock on repetitive pattern: check spatial variance
        std_x = float(np.std(unique_inliers[:, 0])) if len(unique_inliers) > 0 else 0.0
        std_y = float(np.std(unique_inliers[:, 1])) if len(unique_inliers) > 0 else 0.0
        if std_x < 25.0 or std_y < 25.0:
            return False, f"Inliers clustered on narrow line/patch (std_x={std_x:.1f}, std_y={std_y:.1f})", {"scale": scale}

        return True, "", {"scale": scale, "det": det_val}

    def _compute_drift_and_transformed_zones(
        self,
        H: np.ndarray
    ) -> Tuple[float, Dict[str, List[List[float]]]]:
        """Transform baseline polygons via H and compute mean corner drift in pixels against accepted alignment."""
        transformed: Dict[str, List[List[float]]] = {}
        total_drift = 0.0
        total_points = 0
        ref_alignment = getattr(self, "accepted_alignment_zones", self.baseline_zones)

        for name, poly in self.baseline_zones.items():
            pts_arr = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
            try:
                warped_raw = cv2.perspectiveTransform(pts_arr, H)
            except Exception:
                warped_raw = None

            if warped_raw is None:
                return 999.0, {}

            warped = warped_raw.reshape(-1, 2)

            # Compute drift from accepted in-memory alignment
            acc_pts = ref_alignment.get(name, poly)
            acc_arr = np.array(acc_pts, dtype=np.float32).reshape(-1, 2)
            if len(acc_arr) == len(warped):
                deltas = np.linalg.norm(warped - acc_arr, axis=1)
            else:
                deltas = np.linalg.norm(warped - pts_arr.reshape(-1, 2), axis=1)
            total_drift += float(np.sum(deltas))
            total_points += len(deltas)

            transformed[name] = warped.tolist()

        mean_drift = total_drift / float(max(1, total_points))
        return mean_drift, transformed

    def _apply_result_with_safety_limits(self, result: RegistrationResult, timestamp: float, generation: int = 0):
        """Apply safety limits, temporal smoothing, and atomic zone updates."""
        if generation != self._mode_generation:
            logger.debug("[AdaptiveZones] Discarding registration result from obsolete mode generation.")
            return

        if not result.is_valid or result.homography is None or result.transformed_zones is None:
            self._recovery_start_time = None
            self.consecutive_bad_estimates += 1
            self.consecutive_good_frames = 0
            if self.consecutive_bad_estimates >= self.max_consecutive_bad:
                prev_status = self.status
                self.status = AlignmentStatus.ZONES_UNVERIFIED
                self.requires_operator_confirmation = True
                if prev_status != AlignmentStatus.ZONES_UNVERIFIED:
                    self._log_audit_change(
                        event_type="ZONES_UNVERIFIED",
                        action=f"Camera alignment unverified after {self.consecutive_bad_estimates} failures: {result.failure_reason}",
                        drift=self.current_drift_px
                    )
            return

        # Check total drift limit against accepted alignment
        if result.drift_px > self.max_drift_px:
            self._recovery_start_time = None
            self.drift_breached = True
            self.status = AlignmentStatus.ZONES_UNVERIFIED
            self.requires_operator_confirmation = True
            self.proposed_homography = result.homography.copy()
            self.proposed_zones = copy.deepcopy(result.transformed_zones)
            logger.warning(
                f"[AdaptiveZones] Drift limit breached ({result.drift_px:.1f}px > {self.max_drift_px:.1f}px). "
                f"Halting automatic update. Operator confirmation required."
            )
            self._log_audit_change(
                event_type="DRIFT_LIMIT_BREACHED",
                action=f"Camera drift ({result.drift_px:.1f}px) exceeded safety limit ({self.max_drift_px:.1f}px). Operator confirmation required.",
                drift=result.drift_px
            )
            return

        # If operator confirmation is currently required, refresh proposals from newer valid registration
        if self.requires_operator_confirmation:
            if result.is_valid and result.transformed_zones:
                self.proposed_zones = copy.deepcopy(result.transformed_zones)
                self.proposed_homography = result.homography.copy() if result.homography is not None else None
            elif not result.is_valid:
                self.proposed_zones = None
                self.proposed_homography = None
            logger.debug("[AdaptiveZones] Operator confirmation required; updated proposals and skipping auto-update.")
            return

        # Camera returned to valid drift: refresh or clear stale proposals
        self.drift_breached = False
        self.proposed_zones = None
        self.proposed_homography = None

        # Initialize correction clock from first submitted frame timestamp if needed
        if not getattr(self, "_first_frame_ts_set", False):
            self.last_correction_time = timestamp
            self._first_frame_ts_set = True

        # Check max correction rate per second
        dt = max(0.01, timestamp - self.last_correction_time)
        max_allowed_step = self.max_correction_rate_px_per_sec * dt

        # Apply temporal smoothing to proposed polygon vertices
        smoothed_zones: Dict[str, List[List[float]]] = {}
        max_step_taken = 0.0

        for name, prop_pts in result.transformed_zones.items():
            curr_pts = self.current_zones.get(name, self.baseline_zones.get(name, prop_pts))
            smoothed_poly = []
            for i in range(len(prop_pts)):
                target_pt = prop_pts[i]
                current_pt = curr_pts[i] if i < len(curr_pts) else target_pt

                dx = target_pt[0] - current_pt[0]
                dy = target_pt[1] - current_pt[1]
                step_dist = math.hypot(dx, dy)

                if step_dist > max_allowed_step and step_dist > 0:
                    scale_step = max_allowed_step / step_dist
                    dx *= scale_step
                    dy *= scale_step

                # Exponential moving average
                new_x = current_pt[0] + self.smoothing_alpha * dx
                new_y = current_pt[1] + self.smoothing_alpha * dy
                smoothed_poly.append([float(new_x), float(new_y)])
                max_step_taken = max(max_step_taken, math.hypot(new_x - current_pt[0], new_y - current_pt[1]))

            smoothed_zones[name] = smoothed_poly

        # Re-validate all smoothed polygons before applying
        for zn, zpts in smoothed_zones.items():
            v, m = validate_polygon(zpts, name=zn)
            if not v:
                logger.error(f"[AdaptiveZones] Transformed zone '{zn}' failed validation: {m}. Update rejected.")
                self.consecutive_bad_estimates += 1
                self._recovery_start_time = None
                if self.consecutive_bad_estimates >= self.max_consecutive_bad:
                    self.status = AlignmentStatus.ZONES_UNVERIFIED
                    self.requires_operator_confirmation = True
                return

        # Atomic zone update in ZoneManager
        self.zone_version += 1
        success = self.zone_manager.update_zones(smoothed_zones, version=self.zone_version)
        if success:
            self.current_zones = copy.deepcopy(smoothed_zones)
            self.last_good_zones = copy.deepcopy(smoothed_zones)
            self.current_scale = result.scale
            self.last_correction_time = timestamp

            # Compute current drift from accepted alignment
            total_d = 0.0
            total_n = 0
            ref_alignment = getattr(self, "accepted_alignment_zones", self.baseline_zones)
            for name, pts in self.current_zones.items():
                base_pts = ref_alignment.get(name, self.baseline_zones.get(name))
                if base_pts and len(base_pts) == len(pts):
                    deltas = np.linalg.norm(np.array(pts) - np.array(base_pts), axis=1)
                    total_d += float(np.sum(deltas))
                    total_n += len(deltas)
            self.current_drift_px = total_d / float(max(1, total_n))

            # Hysteresis recovery using elapsed time
            self.consecutive_bad_estimates = 0
            self.consecutive_good_frames += 1
            if self._recovery_start_time is None:
                self._recovery_start_time = timestamp
            elif (timestamp - self._recovery_start_time) >= self.recovery_good_sec:
                if not self.drift_breached and self.status != AlignmentStatus.UNVERIFIED_REFERENCE and not self.requires_operator_confirmation:
                    self.status = AlignmentStatus.ALIGNED if self.current_drift_px < 5.0 else AlignmentStatus.DRIFTING
                    # Note: requires_operator_confirmation is kept separate and only cleared by operator in confirm_alignment()

            # Audit log corrections larger than 1.0 px (no flooding)
            if max_step_taken > 1.0:
                self._log_audit_change(
                    event_type="ZONE_ADAPTATION",
                    action=f"Adaptive zones updated to version {self.zone_version}. Drift: {self.current_drift_px:.1f}px, Inliers: {result.inlier_count}",
                    drift=self.current_drift_px
                )

    def confirm_alignment(self, is_valid_cb: Optional[Any] = None) -> Tuple[bool, str]:
        """Operator confirms proposed alignment from the dashboard.
        Accepted alignment lives strictly in memory and is logged to audit log.
        NEVER overwrites saved polygons or config.yaml on disk.
        """
        with self._lock:
            if is_valid_cb is not None and not is_valid_cb():
                return False, "Zone confirmation request expired or was cancelled before commit."
            self._mode_generation += 1
            # If reference was unverified at startup
            if self.status == AlignmentStatus.UNVERIFIED_REFERENCE:
                self.status = AlignmentStatus.ALIGNED
                self.requires_operator_confirmation = False
                self.accepted_alignment_zones = copy.deepcopy(self.current_zones)
                self._log_audit_change(
                    event_type="OPERATOR_CONFIRMED_REFERENCE",
                    action="Operator confirmed initial auto-captured reference frame. Zones verified.",
                    drift=self.current_drift_px
                )
                return True, "Reference frame and baseline zones confirmed."

            # If recovering from drift limit breach or low confidence
            if self.proposed_zones is not None:
                # Validate proposed zones
                for zn, zpts in self.proposed_zones.items():
                    v, m = validate_polygon(zpts, name=zn)
                    if not v:
                        return False, f"Proposed zone '{zn}' is geometrically invalid ({m}). Re-run calibration."

                self.zone_version += 1
                success = self.zone_manager.update_zones(self.proposed_zones, version=self.zone_version)
                if not success:
                    return False, "Failed to apply proposed zones in ZoneManager update_zones."
                self.current_zones = copy.deepcopy(self.proposed_zones)
                self.last_good_zones = copy.deepcopy(self.proposed_zones)
                self.accepted_alignment_zones = copy.deepcopy(self.proposed_zones)
                self.proposed_zones = None
                self.proposed_homography = None
            else:
                self.accepted_alignment_zones = copy.deepcopy(self.current_zones)

            self.status = AlignmentStatus.ALIGNED
            self.drift_breached = False
            self.requires_operator_confirmation = False
            self.consecutive_bad_estimates = 0
            self.consecutive_good_frames = 100
            self._recovery_start_time = None

            self._log_audit_change(
                event_type="OPERATOR_CONFIRMED_ALIGNMENT",
                action=f"Operator verified adaptive zone alignment. Zone version {self.zone_version}.",
                drift=self.current_drift_px
            )
            return True, f"Zone alignment accepted in memory (version {self.zone_version})."

    def _log_audit_change(self, event_type: str, action: str, drift: float):
        """Append state change or significant correction to audit log."""
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": round(time.time(), 3),
                "event_type": event_type,
                "action": action,
                "drift_px": round(drift, 2),
                "zone_version": self.zone_version,
                "status": self.status.value
            }
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"[AdaptiveZones] Failed to write audit log: {e}")

    def stop(self):
        """Clean shutdown of registration worker."""
        self._stop_event.set()
        self._job_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
