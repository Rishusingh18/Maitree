"""Camera Motion and Occlusion Estimator.
Estimates global scene motion via sparse optical flow on background features
(excluding person bounding boxes), and detects camera occlusion/blur via Laplacian variance.
Used to gate fall and deliberate jump classifications so camera shakes do not trigger false L3 alerts.
"""
from typing import Tuple, Optional, List
import numpy as np
import cv2


class CameraMotionDetector:
    """Estimates global camera motion using sparse Lucas-Kanade optical flow."""

    def __init__(
        self,
        motion_threshold_px: float = 5.0,
        blur_threshold: float = 30.0,
        max_corners: int = 120,
        quality_level: float = 0.02,
        min_distance: int = 15
    ):
        self.motion_threshold_px = motion_threshold_px
        self.blur_threshold = blur_threshold
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.frame_count: int = 0

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.frame_count = 0

    def estimate_motion(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[np.ndarray]] = None
    ) -> Tuple[bool, float, float, str]:
        """Estimate global camera motion.
        Returns:
            (is_moving_or_covered, median_motion_px, blur_score, reason)
        """
        if frame is None or frame.size == 0:
            return False, 0.0, 0.0, "Empty frame"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # 1. Blur / camera cover detection via Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.blur_threshold:
            return True, 0.0, blur_score, f"Camera covered or blurred (variance {blur_score:.1f} < {self.blur_threshold:.1f})"

        # 2. Mask out person bounding boxes to isolate background features
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if person_boxes:
            for b in person_boxes:
                # Add small padding to avoid box boundary artifacts
                x1 = max(0, int(b[0]) - 5)
                y1 = max(0, int(b[1]) - 5)
                x2 = min(gray.shape[1], int(b[2]) + 5)
                y2 = min(gray.shape[0], int(b[3]) + 5)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        # 3. Detect or refresh reference features
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 15 or self.frame_count % 30 == 0:
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask
            )
            self.prev_gray = gray
            self.prev_pts = corners
            return False, 0.0, blur_score, "Reference features refreshed"

        # 4. Track background features with sparse Lucas-Kanade optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=2
        )

        median_disp = 0.0
        reason = "Camera stable"
        is_moving = False

        if curr_pts is not None and status is not None:
            good_prev = self.prev_pts[status.ravel() == 1]
            good_curr = curr_pts[status.ravel() == 1]

            if len(good_curr) >= 6:
                diffs = (good_curr - good_prev).reshape(-1, 2)
                displacements = np.hypot(diffs[:, 0], diffs[:, 1])
                median_disp = float(np.median(displacements))

                if median_disp > self.motion_threshold_px:
                    is_moving = True
                    reason = f"Global camera motion ({median_disp:.1f} px > {self.motion_threshold_px:.1f} px threshold)"
            else:
                # Too few tracking points remaining; force refresh on next frame
                self.prev_pts = None

        self.prev_gray = gray
        # Filter surviving points for next frame
        if curr_pts is not None and status is not None:
            self.prev_pts = curr_pts[status.ravel() == 1].reshape(-1, 1, 2)
        else:
            self.prev_pts = None

        return is_moving, median_disp, blur_score, reason

