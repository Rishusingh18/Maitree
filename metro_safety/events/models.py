"""Safety Event data models and severity classifications."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np

class EventType(str, Enum):
    NORMAL = "NORMAL"
    LINE_CROSSING = "LINE_CROSSING"
    TRACK_INTRUSION = "TRACK_INTRUSION"
    ACCIDENTAL_FALL = "ACCIDENTAL_FALL"
    DELIBERATE_JUMP = "DELIBERATE_JUMP"
    DISTRESS_GESTURE = "DISTRESS_GESTURE"
    PERSON_DOWN = "PERSON_DOWN"
    LOITERING = "LOITERING"
    ABANDONED_OBJECT = "ABANDONED_OBJECT"
    POSSIBLE_SNATCH = "POSSIBLE_SNATCH"
    ANOMALY_POSE = "ANOMALY_POSE"

class SeverityLevel(str, Enum):
    NORMAL = "NORMAL"
    L1 = "L1"               # Local buzzer on platform
    L2 = "L2"               # Buzzer + Control Room alert & clip
    L3 = "L3"               # Emergency Stop / ETS Trigger + Control Room alarm
    DISTRESS = "DISTRESS"   # Silent alert to Control Room & Security staff only

@dataclass
class SafetyEvent:
    """Represents a validated safety event output from the classifier."""
    event_type: EventType
    confidence: float
    track_id: int
    timestamp: float
    zone: str
    train_state: str
    evidence_frames: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    camera_motion: bool = False
    motion_reason: str = ""
    zones_unverified: bool = False
    zone_version: int = 1

    def __post_init__(self):
        # Never report exactly 100% (1.0) confidence; strictly cap at 0.99 and calibrate
        try:
            val = float(self.confidence)
            if val > 1.0:
                val = val / 100.0
            self.confidence = min(0.99, max(0.01, val))
        except (ValueError, TypeError):
            self.confidence = 0.50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "confidence": round(float(self.confidence), 3),
            "track_id": self.track_id,
            "timestamp": round(float(self.timestamp), 3),
            "zone": self.zone,
            "train_state": self.train_state,
            "evidence_frames": self.evidence_frames,
            "metadata": self.metadata,
            "camera_motion": self.camera_motion,
            "motion_reason": self.motion_reason,
            "zones_unverified": self.zones_unverified,
            "zone_version": self.zone_version
        }

