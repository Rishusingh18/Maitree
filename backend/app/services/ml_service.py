import base64
import io
import torch
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import asyncio
from ..config import settings

# Global model holders (initialized lazily)
_crossing_model = None
_gesture_model = None

# Thread pool for blocking inference calls
_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

def _load_model(path: str):
    """Load a torch model from the given path.
    The model should be saved with ``torch.save(model.state_dict())`` and
    expects a matching architecture defined in the codebase.
    """
    # Placeholder simple model – replace with actual architecture if needed.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2),
        torch.nn.Softmax(dim=1),
    )
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # If a full model was saved instead of state_dict, fallback
        model = state_dict
    model.eval()
    return model

def _get_crossing_model():
    global _crossing_model
    if _crossing_model is None:
        _crossing_model = _load_model(settings.CROSSING_MODEL_PATH)
    return _crossing_model

def _get_gesture_model():
    global _gesture_model
    if _gesture_model is None:
        _gesture_model = _load_model(settings.GESTURE_MODEL_PATH)
    return _gesture_model

# Common preprocessing pipeline for image models
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization parameters may need adjusting per model
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _preprocess_image(b64_str: str) -> torch.Tensor:
    """Decode base64 image, apply transforms, and add batch dimension."""
    try:
        img_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")
    tensor = _transform(image).unsqueeze(0)  # Shape: (1, C, H, W)
    return tensor

def _run_inference(model, tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(tensor)
    # Assume model returns probabilities for two classes
    confidence, pred_idx = torch.max(outputs, dim=1)
    label = "positive" if pred_idx.item() == 1 else "negative"
    return label, confidence.item()

async def predict_crossing(image_b64: str):
    """Async wrapper for crossing detection inference."""
    loop = asyncio.get_event_loop()
    tensor = _preprocess_image(image_b64)
    model = _get_crossing_model()
    label, confidence = await loop.run_in_executor(_executor, _run_inference, model, tensor)
    return label, confidence

async def predict_gesture(image_b64: str):
    """Async wrapper for gesture/threat detection inference."""
    loop = asyncio.get_event_loop()
    tensor = _preprocess_image(image_b64)
    model = _get_gesture_model()
    label, confidence = await loop.run_in_executor(_executor, _run_inference, model, tensor)
    return label, confidence
