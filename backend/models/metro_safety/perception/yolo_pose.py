"""Ultralytics YOLO Pose detector implementation."""
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from metro_safety.perception.base import BasePoseDetector, Detection

class YOLOPoseDetector(BasePoseDetector):
    """Ultralytics YOLO Pose estimation model wrapper."""

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        inference_size: int = 320
    ):
        self.model_name = model_name
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.device = device
        self.inference_size = inference_size
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        except Exception as e:
            print(f"[Warning] YOLO model load failed: {e}. Falling back to dummy mock detector.")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect person bounding boxes and 17 keypoints."""
        if self._model is None or frame is None or frame.size == 0:
            return []

        results = self._model(
            frame,
            imgsz=self.inference_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        keypoints = res.keypoints

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        kpts_data = None
        if keypoints is not None and keypoints.data is not None:
            kpts_data = keypoints.data.cpu().numpy()  # shape (N, 17, 3)

        for i in range(len(xyxy)):
            box = xyxy[i]
            conf = float(confs[i])
            kpts = kpts_data[i] if (kpts_data is not None and i < len(kpts_data)) else None
            detections.append(Detection(
                box=box,
                confidence=conf,
                keypoints=kpts
            ))

        return detections

