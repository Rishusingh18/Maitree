import base64
import io
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2
from ..config import settings

logger = logging.getLogger("MLService")

# Locks to protect Ultralytics predictor first-use setup and concurrent inference
_crossing_lock = threading.Lock()
_gesture_lock = threading.Lock()
_behaviour_lock = threading.Lock()

# Global model holders (initialized lazily)
_crossing_yolo = None
_gesture_yolo = None
_behaviour_yolo = None

_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

def _resolve_model_path(path: str) -> Optional[str]:
    """Resolve model path correctly without duplicating 'backend' component."""
    if os.path.exists(path):
        return path

    # Check relative to project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    p_from_root = os.path.join(root_dir, path)
    if os.path.exists(p_from_root):
        return p_from_root

    # If path starts with 'backend/' or 'backend\', strip it when inside backend directory
    if path.startswith("backend/") or path.startswith("backend\\"):
        sub_path = path[8:]
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        p_backend = os.path.join(backend_dir, sub_path)
        if os.path.exists(p_backend):
            return p_backend

    return None

def _get_model(path: str):
    """Load Ultralytics YOLO model or return None on failure."""
    resolved = _resolve_model_path(path)
    if resolved is None:
        logger.warning(f"Model path '{path}' not found on filesystem.")
        return None
    try:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model from {resolved}...")
        return YOLO(resolved)
    except Exception as e:
        logger.error(f"Could not load YOLO from {resolved}: {e}")
        return None

def _get_crossing_model():
    global _crossing_yolo
    with _crossing_lock:
        if _crossing_yolo is None:
            _crossing_yolo = _get_model(settings.CROSSING_MODEL_PATH)
    return _crossing_yolo

def _get_gesture_model():
    global _gesture_yolo
    with _gesture_lock:
        if _gesture_yolo is None:
            _gesture_yolo = _get_model(settings.GESTURE_MODEL_PATH)
    return _gesture_yolo

def _get_behaviour_model():
    global _behaviour_yolo
    with _behaviour_lock:
        if _behaviour_yolo is None:
            _behaviour_yolo = _get_model(settings.BEHAVIOUR_MODEL_PATH)
    return _behaviour_yolo

def _decode_image(b64_str: str) -> np.ndarray:
    """Decode base64 string to contiguous BGR numpy array as expected by YOLO."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
    rgb_arr = np.array(pil_img)
    # Convert RGB to contiguous BGR for OpenCV / YOLO input
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr_arr)

def _run_crossing_inference(img_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Run crossing / platform safety detection assessing every person box."""
    model = _get_crossing_model()
    if model is None:
        if getattr(settings, 'DEMO_MODE', False):
            return "safe_platform", 0.95, {"info": "Demo fallback - model checkpoint pending"}
        raise RuntimeError("Crossing safety model checkpoint is not available")

    with _crossing_lock:
        results = model(img_bgr, verbose=False)

    if not results:
        return "clear", 0.99, {"detections": 0}

    res = results[0]
    num_persons = len(res.boxes) if res.boxes is not None else 0
    details = {"persons_detected": num_persons}

    if num_persons == 0:
        return "safe_platform", 0.99, details

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.full(num_persons, 0.9)
    img_h = img_bgr.shape[0]

    # Assess every detected person box and return the highest risk indicated by any person
    warning_detected = False
    highest_risk_conf = 0.0

    for i, box in enumerate(boxes):
        conf_i = float(confs[i])
        foot_y = float(box[3])
        if foot_y > (img_h * 0.70):
            warning_detected = True
            if conf_i > highest_risk_conf:
                highest_risk_conf = conf_i

    if warning_detected:
        label = "warning_near_edge"
        confidence = highest_risk_conf if highest_risk_conf > 0 else 0.90
    else:
        label = "safe_platform"
        confidence = float(np.mean(confs)) if len(confs) > 0 else 0.90

    return label, confidence, details

def _run_gesture_inference(img_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Run gesture / distress signal detection evaluating each person's keypoints."""
    model = _get_gesture_model()
    if model is None:
        if getattr(settings, 'DEMO_MODE', False):
            return "no_distress", 0.95, {"info": "Demo fallback - gesture model pending"}
        raise RuntimeError("Gesture distress model checkpoint is not available")

    with _gesture_lock:
        results = model(img_bgr, verbose=False)

    if not results or results[0].keypoints is None:
        return "normal", 0.98, {"distress_detected": False, "keypoints_analyzed": 0}

    res = results[0]
    kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else []
    num_skeletons = len(kpts)

    label = "normal"
    confidence = 0.92

    # Evaluate each person's keypoints: 9=left wrist, 10=right wrist, 0=nose
    for person_kpts in kpts:
        if len(person_kpts) >= 11:
            left_wrist = person_kpts[9]
            right_wrist = person_kpts[10]
            nose = person_kpts[0]
            # Macro distress signal check: both hands raised above head/nose
            if left_wrist[1] > 0 and right_wrist[1] > 0 and nose[1] > 0:
                if left_wrist[1] < nose[1] and right_wrist[1] < nose[1]:
                    label = "distress_signal_detected"
                    confidence = 0.94
                    break

    return label, confidence, {"distress_detected": (label == "distress_signal_detected"), "keypoints_analyzed": num_skeletons}

def _run_behaviour_inference(img_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
    """Run behaviour / violence / theft detection mapping model's predicted class and score."""
    model = _get_behaviour_model()
    if model is None:
        if getattr(settings, 'DEMO_MODE', False):
            return "normal_activity", 0.90, {"info": "Demo fallback - behaviour model pending"}
        raise RuntimeError("Behaviour model checkpoint is not available")

    with _behaviour_lock:
        results = model(img_bgr, verbose=False)

    if not results:
        return "normal_activity", 0.95, {"detections": 0}

    res = results[0]

    # Map model's predicted class and score
    if hasattr(res, 'probs') and res.probs is not None:
        top1_idx = int(res.probs.top1)
        conf = float(res.probs.top1conf)
        class_name = res.names.get(top1_idx, "normal_activity") if hasattr(res, 'names') else "normal_activity"
        return class_name, conf, {"top1_class_id": top1_idx}
    elif hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()
        max_idx = int(np.argmax(confs))
        conf = float(confs[max_idx])
        class_id = int(clss[max_idx])
        class_name = res.names.get(class_id, "suspicious_behaviour") if hasattr(res, 'names') else "suspicious_behaviour"
        return class_name, conf, {"detected_objects": len(res.boxes), "top_class_id": class_id}
    else:
        return "normal_activity", 0.90, {"info": "No anomalous behaviour detected"}

async def predict_crossing(image_b64: str) -> Tuple[str, float, Dict[str, Any]]:
    """Async wrapper for crossing detection."""
    loop = asyncio.get_event_loop()
    img_bgr = _decode_image(image_b64)
    return await loop.run_in_executor(_executor, _run_crossing_inference, img_bgr)

async def predict_gesture(image_b64: str) -> Tuple[str, float, Dict[str, Any]]:
    """Async wrapper for gesture/distress detection."""
    loop = asyncio.get_event_loop()
    img_bgr = _decode_image(image_b64)
    return await loop.run_in_executor(_executor, _run_gesture_inference, img_bgr)

async def predict_behaviour(image_b64: str) -> Tuple[str, float, Dict[str, Any]]:
    """Async wrapper for behaviour/threat detection."""
    loop = asyncio.get_event_loop()
    img_bgr = _decode_image(image_b64)
    return await loop.run_in_executor(_executor, _run_behaviour_inference, img_bgr)
