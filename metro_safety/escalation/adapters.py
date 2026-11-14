"""Hardware adapter interfaces and mock simulators for Platform Safety System.
Includes:
- Audible Platform Buzzer
- Control Room Notifier (Console + Webhook + Log File)
- Emergency Trip Switch (ETS) with human-confirmation interlock
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import requests

from metro_safety.events.models import SafetyEvent, SeverityLevel

class BasePlatformBuzzer(ABC):
    """Abstract interface for local station platform audible buzzer."""

    @abstractmethod
    def activate(self, platform_id: str, pattern: str = "PULSED"):
        """Sound the local buzzer."""
        pass

    @abstractmethod
    def deactivate(self, platform_id: str):
        """Silence the buzzer."""
        pass

    @abstractmethod
    def is_active(self, platform_id: str) -> bool:
        pass


class SimulatedPlatformBuzzer(BasePlatformBuzzer):
    """Mock simulated platform buzzer logging state changes."""

    def __init__(self):
        self._states: Dict[str, bool] = {}

    def activate(self, platform_id: str, pattern: str = "PULSED"):
        self._states[platform_id] = True

    def deactivate(self, platform_id: str):
        self._states[platform_id] = False

    def is_active(self, platform_id: str) -> bool:
        return self._states.get(platform_id, False)


class BaseControlRoomNotifier(ABC):
    """Abstract interface for Control Room alarming and dispatch."""

    @abstractmethod
    def send_alert(self, event: SafetyEvent, severity: SeverityLevel, message: str) -> bool:
        pass


class SimulatedControlRoomNotifier(BaseControlRoomNotifier):
    """Dispatches alerts to console, local JSON audit log, and optional HTTP webhook."""

    def __init__(self, log_path: Path | str = "logs/safety_audit.log", webhook_url: Optional[str] = None):
        self.log_path = Path(log_path)
        self.webhook_url = webhook_url
        self.alert_history = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def send_alert(self, event: SafetyEvent, severity: SeverityLevel, message: str) -> bool:
        entry = {
            "timestamp": time.time(),
            "event_time": event.timestamp,
            "severity": severity.value,
            "event_type": event.event_type.value,
            "track_id": event.track_id,
            "confidence": event.confidence,
            "zone": event.zone,
            "train_state": event.train_state,
            "message": message,
            "evidence_frames": event.evidence_frames
        }
        self.alert_history.append(entry)

        # Append to persistent audit log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Optional simulated Webhook dispatch
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=entry, timeout=0.1)
            except Exception:
                pass  # Simulated webhook fallback

        return True


class BaseEmergencyTripSwitch(ABC):
    """Abstract interface for signaling interlocking Emergency Trip Switch (ETS) / Traction Power Cut."""

    @abstractmethod
    def trigger_stop(self, event: SafetyEvent, station_id: str, platform_id: str) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_tripped(self) -> bool:
        pass


class SimulatedEmergencyTripSwitch(BaseEmergencyTripSwitch):
    """Simulated ETS with human-confirmation interlock option."""

    def __init__(self, human_confirmation_required: bool = True, timeout_sec: float = 5.0):
        self.human_confirmation_required = human_confirmation_required
        self.timeout_sec = timeout_sec
        self.tripped = False
        self.pending_confirmation: Optional[Dict] = None

    def trigger_stop(self, event: SafetyEvent, station_id: str, platform_id: str) -> bool:
        payload = {
            "station_id": station_id,
            "platform_id": platform_id,
            "event": event.to_dict(),
            "timestamp": time.time()
        }

        if not self.human_confirmation_required:
            self.tripped = True
            return True
        else:
            # Stage for operator approval
            self.pending_confirmation = payload
            return False

    def confirm_by_operator(self) -> bool:
        """Station supervisor or control room operator confirms emergency trip."""
        if self.pending_confirmation:
            self.tripped = True
            self.pending_confirmation = None
            return True
        return False

    def reset(self):
        self.tripped = False
        self.pending_confirmation = None

    def is_tripped(self) -> bool:
        return self.tripped

