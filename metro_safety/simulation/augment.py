"""Augmentation utilities for robustness testing (lighting, blur, crowd, occlusion)."""
import numpy as np
import cv2

def apply_brightness(frame: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """Simulate low lighting / night-time CCTV conditions."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_motion_blur(frame: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Simulate camera vibration or high-speed motion blur."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(frame, -1, kernel)

def apply_synthetic_occlusion(frame: np.ndarray, num_blocks: int = 2) -> np.ndarray:
    """Simulate platform pillars or luggage occluding the camera view."""
    res = frame.copy()
    h, w = frame.shape[:2]
    for _ in range(num_blocks):
        bx = np.random.randint(int(w * 0.2), int(w * 0.7))
        by = np.random.randint(int(h * 0.4), int(h * 0.8))
        bw = np.random.randint(30, 70)
        bh = np.random.randint(40, 100)
        # Draw dark pillar/obstacle
        cv2.rectangle(res, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
    return res

