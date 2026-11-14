"""Perception base abstractions and detection containers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Standard COCO 17 Keypoints mapping
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

@dataclass
class Detection:
    """Represents a person detection in a single frame."""
    box: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[np.ndarray] = None  # shape (17, 3): x, y, conf
    track_id: Optional[int] = None
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Anchor point on ground using bounding box bottom edge."""
        x = (self.box[0] + self.box[2]) / 2.0
        y = self.box[3]
        return float(x), float(y)
    
    @property
    def foot_anchor(self) -> Tuple[float, float]:
        """Anchor point using ankles if confident, else bottom center."""
        if self.keypoints is not None and len(self.keypoints) >= 17:
            l_ankle = self.keypoints[15]
            r_ankle = self.keypoints[16]
            valid_ankles = []
            if l_ankle[2] > 0.35:
                valid_ankles.append((l_ankle[0], l_ankle[1]))
            if r_ankle[2] > 0.35:
                valid_ankles.append((r_ankle[0], r_ankle[1]))
            if valid_ankles:
                avg_x = sum(p[0] for p in valid_ankles) / len(valid_ankles)
                avg_y = sum(p[1] for p in valid_ankles) / len(valid_ankles)
                return float(avg_x), float(avg_y)
        return self.bottom_center

    @property
    def torso_center(self) -> Tuple[float, float]:
        """Center of mass approximate via shoulders and hips."""
        if self.keypoints is not None and len(self.keypoints) >= 13:
            pts = []
            for idx in [5, 6, 11, 12]:
                if self.keypoints[idx][2] > 0.3:
                    pts.append(self.keypoints[idx][:2])
            if pts:
                pts_arr = np.array(pts)
                return float(pts_arr[:, 0].mean()), float(pts_arr[:, 1].mean())
        x = (self.box[0] + self.box[2]) / 2.0
        y = (self.box[1] + self.box[3]) / 2.0
        return float(x), float(y)

    @property
    def torso_angle(self) -> float:
        """Angle of torso relative to horizontal (degrees). Upright = ~90 deg."""
        if self.keypoints is not None and len(self.keypoints) >= 13:
            sh_pts = [self.keypoints[i][:2] for i in [5, 6] if self.keypoints[i][2] > 0.3]
            hip_pts = [self.keypoints[i][:2] for i in [11, 12] if self.keypoints[i][2] > 0.3]
            if sh_pts and hip_pts:
                sh_mean = np.mean(sh_pts, axis=0)
                hip_mean = np.mean(hip_pts, axis=0)
                dx = sh_mean[0] - hip_mean[0]
                dy = sh_mean[1] - hip_mean[1] # Note: y is down in image coordinates
                angle = abs(np.degrees(np.arctan2(-dy, dx)))
                if angle > 90:
                    angle = 180 - angle
                return float(angle)
        # Fall back to bounding box aspect ratio: tall box = upright
        w = max(1.0, self.box[2] - self.box[0])
        h = max(1.0, self.box[3] - self.box[1])
        return 85.0 if h > w * 1.3 else 30.0


class BasePoseDetector(ABC):
    """Abstract interface for person pose detector models."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run pose detection on a single frame."""
        pass

