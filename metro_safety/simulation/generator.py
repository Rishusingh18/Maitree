"""High-fidelity synthetic scenario generator for metro platform CCTV.
Simulates realistic 2D kinematic skeletal trajectories and renders CCTV-like visual scenes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import json

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Facial
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12),                    # Torso sides
    (11, 12),                            # Hips
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16)                   # Right leg
]

@dataclass
class SyntheticPerson:
    """Represents a simulated person track over time."""
    track_id: int
    scenario_type: str  # "NORMAL_WAITING", "NORMAL_WALKING", "LINE_CROSSING", "ACCIDENTAL_FALL", "DELIBERATE_JUMP", "DISTRESS_GESTURE"
    start_pos: Tuple[float, float]
    frames_total: int
    gesture_type: Optional[str] = None   # "SIGNAL_FOR_HELP" or "WRISTS_CROSSED"

class ScenarioGenerator:
    """Generates synthetic video sequences and ground truth keypoint trajectories."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25.0):
        self.w = width
        self.h = height
        self.fps = fps
        # Default zone lines for rendering background
        self.platform_y = int(self.h * 0.40)      # y=192
        self.yellow_y = int(self.h * 0.58)        # y=278
        self.track_y = int(self.h * 0.68)         # y=326

    def render_background(self) -> np.ndarray:
        """Create a synthetic CCTV metro platform backdrop."""
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Upper wall/ceiling (Dark Gray)
        bg[0:self.platform_y, :] = (60, 60, 65)
        # Platform floor (Granite / Concrete gray)
        bg[self.platform_y:self.track_y, :] = (130, 130, 135)
        # Yellow safety tactile strip (Golden yellow)
        bg[self.yellow_y:self.track_y, :] = (20, 210, 240)
        # Platform edge granite border (Dark border)
        bg[self.track_y-4:self.track_y, :] = (40, 40, 40)
        # Track bed (Dark asphalt/gravel)
        bg[self.track_y:, :] = (35, 35, 40)
        # Rails (Silver parallel lines)
        rail1_y = self.track_y + int((self.h - self.track_y) * 0.35)
        rail2_y = self.track_y + int((self.h - self.track_y) * 0.70)
        cv2.line(bg, (0, rail1_y), (self.w, rail1_y), (180, 180, 190), 3)
        cv2.line(bg, (0, rail2_y), (self.w, rail2_y), (180, 180, 190), 3)
        # Sleepers (Cross ties)
        for x in range(0, self.w, 40):
            cv2.line(bg, (x, self.track_y + 10), (x, self.h - 10), (60, 75, 90), 4)

        # Station Signboard
        cv2.rectangle(bg, (int(self.w*0.35), 20), (int(self.w*0.65), 70), (180, 50, 0), -1)
        cv2.putText(bg, "DMRC PLATFORM 1", (int(self.w*0.37), 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return bg

    def generate_person_trajectory(
        self,
        scenario: str,
        num_frames: int = 75,
        gesture_variant: str = "SIGNAL_FOR_HELP"
    ) -> List[Dict]:
        """Generate frame-by-frame 17 COCO keypoints and bounding boxes."""
        frames_data = []

        # Base starting coordinates (standing safely on platform)
        curr_x = float(self.w * 0.45)
        curr_y = float(self.yellow_y - 45) # safe on platform
        height = 95.0
        width = 34.0

        for f in range(num_frames):
            # Kinematic dynamics per scenario
            if scenario == "NORMAL_WAITING":
                # Subtle breathing / small shifts
                x = curr_x + np.sin(f * 0.1) * 1.5
                y = curr_y + np.cos(f * 0.08) * 0.8
                torso_tilt = 88.0
                in_track = False

            elif scenario == "NORMAL_WALKING":
                # Moving parallel along platform safely
                curr_x += 1.8
                x = curr_x
                y = curr_y + np.sin(f * 0.3) * 1.2
                torso_tilt = 85.0
                in_track = False

            elif scenario == "LINE_CROSSING":
                # Strolling gradually forward across the yellow band (y: 278 -> 326), then stepping back
                if f < 25:
                    y = curr_y + (f / 25.0) * 65.0 # enters deep into yellow band (y -> 298)
                elif f < 50:
                    y = curr_y + 65.0              # lingers in yellow band
                else:
                    y = curr_y + 65.0 - ((f - 50) / 25.0) * 55.0 # steps back onto platform
                x = curr_x + f * 0.5
                torso_tilt = 86.0
                in_track = False

            elif scenario == "ACCIDENTAL_FALL":
                # Person stumbles at frame 25: rapid downward acceleration, torso rotates horizontal
                if f < 25:
                    x = curr_x + f * 0.2
                    y = curr_y + 15.0 # near edge
                    torso_tilt = 85.0
                else:
                    dt = f - 25
                    # Sudden vertical drop into track bed
                    drop = 0.5 * 1.4 * (dt ** 1.8)
                    y = min(float(self.h - 30), curr_y + 15.0 + drop)
                    x = curr_x + 5.0 + dt * 1.5
                    # Torso angle falls towards horizontal (90 -> 15 degrees)
                    torso_tilt = max(10.0, 85.0 - dt * 6.5)

            elif scenario == "DELIBERATE_JUMP":
                # Intentional run towards edge (frames 0-20), brief leap/descent (frame 20+), upright/crouch
                if f < 20:
                    x = curr_x + f * 1.2
                    y = curr_y + f * 1.8 # purposeful rapid approach to edge
                    torso_tilt = 82.0
                else:
                    dt = f - 20
                    x = curr_x + 24.0 + dt * 2.5
                    # Controlled leap descent into track
                    y = min(float(self.h - 35), curr_y + 36.0 + dt * 5.5)
                    torso_tilt = max(45.0, 75.0 - dt * 2.0) # Maintains more upright/crouch orientation

            elif scenario == "DISTRESS_GESTURE":
                x = curr_x
                y = curr_y
                torso_tilt = 88.0

            # Construct 17 COCO keypoints relative to (x, y) anchor (ankles)
            kpts = np.zeros((17, 3), dtype=np.float32)

            # Feet/Ankles at y
            kpts[15] = [x - 8, y, 0.90]       # left ankle
            kpts[16] = [x + 8, y, 0.90]       # right ankle

            # Knees
            knee_y = y - height * 0.25
            kpts[13] = [x - 8, knee_y, 0.90]  # left knee
            kpts[14] = [x + 8, knee_y, 0.90]  # right knee

            # Hips
            hip_y = y - height * 0.50
            kpts[11] = [x - 10, hip_y, 0.92]  # left hip
            kpts[12] = [x + 10, hip_y, 0.92]  # right hip

            # Shoulders (tilt computed based on torso_tilt)
            tilt_rad = np.radians(torso_tilt)
            torso_len = height * 0.32
            sh_x_off = torso_len * np.cos(tilt_rad)
            sh_y_off = torso_len * np.sin(tilt_rad)

            sh_center_x = x + (torso_len * np.cos(tilt_rad) if torso_tilt < 60 else 0)
            sh_center_y = hip_y - sh_y_off

            kpts[5] = [sh_center_x - 12, sh_center_y, 0.95] # left shoulder
            kpts[6] = [sh_center_x + 12, sh_center_y, 0.95] # right shoulder

            # Head
            head_y = sh_center_y - height * 0.18
            kpts[0] = [sh_center_x, head_y, 0.95]           # nose
            kpts[1] = [sh_center_x - 4, head_y - 2, 0.90]   # left eye
            kpts[2] = [sh_center_x + 4, head_y - 2, 0.90]   # right eye
            kpts[3] = [sh_center_x - 7, head_y - 1, 0.85]   # left ear
            kpts[4] = [sh_center_x + 7, head_y - 1, 0.85]   # right ear

            # Arms & Wrists dynamics
            if scenario == "DISTRESS_GESTURE" and f >= 20:
                if gesture_variant == "WRISTS_CROSSED":
                    # Arms raised crossed over head
                    kpts[7] = [sh_center_x - 15, sh_center_y - 15, 0.92] # L elbow
                    kpts[8] = [sh_center_x + 15, sh_center_y - 15, 0.92] # R elbow
                    kpts[9] = [sh_center_x + 3, head_y - 15, 0.95]       # L wrist crossed above head
                    kpts[10] = [sh_center_x - 3, head_y - 16, 0.95]      # R wrist crossed above head
                else:
                    # Signal for help: Right hand raised palm/fingers folding
                    kpts[7] = [sh_center_x - 12, hip_y - 10, 0.90]       # L elbow resting
                    kpts[9] = [sh_center_x - 12, hip_y + 10, 0.90]       # L wrist resting
                    kpts[8] = [sh_center_x + 18, sh_center_y - 10, 0.92] # R elbow raised
                    # R wrist high near face/head
                    kpts[10] = [sh_center_x + 22, head_y - 5, 0.96]      # R wrist raised
            else:
                # Normal natural arm swing
                arm_swing = np.sin(f * 0.3) * 6.0
                kpts[7] = [sh_center_x - 14, sh_center_y + 15 + arm_swing, 0.90]
                kpts[8] = [sh_center_x + 14, sh_center_y + 15 - arm_swing, 0.90]
                kpts[9] = [sh_center_x - 14, sh_center_y + 32 + arm_swing, 0.90]
                kpts[10] = [sh_center_x + 14, sh_center_y + 32 - arm_swing, 0.90]

            # Bounding box containing keypoints + padding
            min_x = max(0.0, float(np.min(kpts[:, 0]) - 8.0))
            max_x = min(float(self.w), float(np.max(kpts[:, 0]) + 8.0))
            min_y = max(0.0, float(np.min(kpts[:, 1]) - 10.0))
            max_y = min(float(self.h), float(np.max(kpts[:, 1]) + 5.0))
            box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

            frames_data.append({
                "frame_idx": f,
                "box": box,
                "keypoints": kpts,
                "confidence": 0.88,
                "ground_truth_event": scenario,
                "torso_tilt": torso_tilt
            })

        return frames_data

    def render_scenario_video(
        self,
        scenario_data: List[Dict],
        output_path: Path | str,
        add_noise: bool = False
    ):
        """Render realistic CCTV video frames and save to MP4."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (self.w, self.h))

        bg_template = self.render_background()

        for frame_dict in scenario_data:
            frame = bg_template.copy()
            kpts = frame_dict["keypoints"]
            box = frame_dict["box"].astype(int)

            # Draw body skeleton
            # Bones
            for p1_idx, p2_idx in COCO_PAIRS:
                p1 = kpts[p1_idx]
                p2 = kpts[p2_idx]
                if p1[2] > 0.4 and p2[2] > 0.4:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (245, 230, 210), 3)

            # Joints
            for pt in kpts:
                if pt[2] > 0.4:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 180, 255), -1)

            # Head circle
            nose = kpts[0]
            cv2.circle(frame, (int(nose[0]), int(nose[1])), 8, (230, 200, 180), -1)

            # Optional noise / blur / lighting degradation
            if add_noise:
                # Add gaussian noise
                noise = np.random.normal(0, 12, frame.shape).astype(np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)

        writer.release()

