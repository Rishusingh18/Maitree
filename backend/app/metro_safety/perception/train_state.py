"""Train state provider interfaces and implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import logging
import time
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class TrainState(str, Enum):
    """Platform train presence / movement states."""
    ABSENT = "ABSENT"
    APPROACHING = "APPROACHING"
    STOPPED_DOORS_OPEN = "STOPPED_DOORS_OPEN"
    DEPARTING = "DEPARTING"
    PRESENT = "PRESENT"  # Backward compatibility alias for STOPPED_DOORS_OPEN

    @classmethod
    def from_str(cls, val: Any) -> "TrainState":
        """Parse train state from string with aliases."""
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        if s in ("ABSENT", "APPROACHING", "STOPPED_DOORS_OPEN", "DEPARTING", "PRESENT"):
            return cls(s)
        if s in ("STOPPED", "DOORS_OPEN", "DOCKED", "BOARDING"):
            return cls.STOPPED_DOORS_OPEN
        if s in ("LEAVING", "DEPART", "DEP"):
            return cls.DEPARTING
        if s in ("APPROACH", "ARRIVING", "APP"):
            return cls.APPROACHING
        if s in ("NONE", "EMPTY", "AWAY"):
            return cls.ABSENT
        raise ValueError(f"Unknown train state: {val}")


@dataclass
class TrainStateRecord:
    """Per-platform state tracking record with transition history."""
    state: TrainState
    timestamp: float
    platform_id: str = "PL_DOWN_01"
    previous_state: Optional[TrainState] = None
    transition_timestamp: float = 0.0
    source: str = "simulated"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainStateProvider(ABC):
    """Abstract train state provider interface."""

    def __init__(
        self,
        default_platform_id: str = "PL_DOWN_01",
        grace_period_sec: float = 2.0,
        fail_safe_timeout_sec: float = 10.0
    ):
        self.default_platform_id = default_platform_id
        self._last_platform_id = default_platform_id
        self.grace_period_sec = float(grace_period_sec)
        self.fail_safe_timeout_sec = float(fail_safe_timeout_sec)

    @abstractmethod
    def get_state(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None
    ) -> TrainState:
        """Returns the current train state for the platform."""
        pass

    @abstractmethod
    def get_state_record(
        self,
        platform_id: Optional[str] = None
    ) -> Optional[TrainStateRecord]:
        """Returns the full TrainStateRecord for the platform."""
        pass

    @abstractmethod
    def set_state(
        self,
        state: Union[TrainState, str],
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainStateRecord:
        """Sets train state for a platform."""
        pass

    def touch(self, platform_id: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        """Update last heartbeat timestamp without changing state (for fail-safe freshness)."""
        pass

    def is_geofence_active(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> bool:
        """Evaluates whether the yellow-line geofence is active for the given platform.

        Rules:
        - Fail-safe: if record is unknown, missing, or stale (> fail_safe_timeout_sec), return True.
        - Active in ABSENT, APPROACHING, DEPARTING.
        - Disabled only in STOPPED_DOORS_OPEN (and PRESENT).
        - On transition STOPPED_DOORS_OPEN -> DEPARTING:
            re-enable after grace_period_sec (default 2s).
        """
        t = float(timestamp) if timestamp is not None else time.time()
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        record = self.get_state_record(pid)
        if record is None:
            logger.debug(f"[TrainStateProvider] No record for platform '{pid}', fail-safe ACTIVE.")
            return True

        # Fail-safe freshness check
        if (t - record.timestamp) > self.fail_safe_timeout_sec:
            logger.warning(
                f"[TrainStateProvider] Fail-safe ACTIVE for platform '{pid}': "
                f"update stale by {t - record.timestamp:.2f}s (threshold {self.fail_safe_timeout_sec}s)"
            )
            return True

        state = record.state
        # STOPPED_DOORS_OPEN / PRESENT: geofence disabled
        if state in (TrainState.STOPPED_DOORS_OPEN, TrainState.PRESENT):
            return False

        # DEPARTING: check if transitioning from STOPPED_DOORS_OPEN / PRESENT
        if state == TrainState.DEPARTING:
            if record.previous_state in (TrainState.STOPPED_DOORS_OPEN, TrainState.PRESENT):
                dt = t - record.transition_timestamp
                if 0.0 <= dt < self.grace_period_sec:
                    return False  # Still in grace period
                return True
            return True

        # ABSENT, APPROACHING, or any other state: geofence active
        return True

    def is_update_stale(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> bool:
        """Check if platform state update is stale exceeding fail_safe_timeout_sec."""
        t = float(timestamp) if timestamp is not None else time.time()
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        record = self.get_state_record(pid)
        if record is None:
            return True
        return (t - record.timestamp) > self.fail_safe_timeout_sec


BaseTrainStateProvider = TrainStateProvider


class SimulatedProvider(TrainStateProvider):
    """Provider for programmatic simulation, MQTT ingestion, or manual UI control."""

    def __init__(
        self,
        default_platform_id: str = "PL_DOWN_01",
        initial_state: Union[TrainState, str] = TrainState.ABSENT,
        grace_period_sec: float = 2.0,
        fail_safe_timeout_sec: float = 10.0
    ):
        super().__init__(
            default_platform_id=default_platform_id,
            grace_period_sec=grace_period_sec,
            fail_safe_timeout_sec=fail_safe_timeout_sec
        )
        self._platforms: Dict[str, TrainStateRecord] = {}
        init_st = TrainState.from_str(initial_state) if isinstance(initial_state, str) else initial_state
        self.set_state(
            state=init_st,
            platform_id=self.default_platform_id,
            timestamp=time.time(),
            source="init"
        )
        self._mqtt_client = None

    def get_state_record(
        self,
        platform_id: Optional[str] = None
    ) -> Optional[TrainStateRecord]:
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        return self._platforms.get(pid)

    def get_state(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None
    ) -> TrainState:
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        rec = self._platforms.get(pid)
        if rec is None:
            return TrainState.ABSENT
        return rec.state

    def set_state(
        self,
        state: Union[TrainState, str],
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainStateRecord:
        import math
        if timestamp is not None:
            try:
                t_val = float(timestamp)
                if not math.isfinite(t_val):
                    t = time.time()
                else:
                    t = t_val
            except (ValueError, TypeError):
                t = time.time()
        else:
            t = time.time()

        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        self._last_platform_id = pid
        parsed_state = TrainState.from_str(state) if hasattr(TrainState, "from_str") else TrainState(state)

        prev_record = self._platforms.get(pid)
        is_external = source.startswith("mqtt") or source.startswith("external")
        if prev_record is not None and is_external and prev_record.source not in ("init", "vision_init") and not str(source).startswith("reset"):
            if t < prev_record.timestamp:
                logger.warning(
                    f"[TrainStateProvider] Rejecting stale observation for '{pid}': "
                    f"t={t:.3f} < stored={prev_record.timestamp:.3f} (source={source})"
                )
                return prev_record

        if prev_record is None:
            prev_state = None
            trans_ts = t
        elif prev_record.state != parsed_state:
            prev_state = prev_record.state
            trans_ts = t
        else:
            prev_state = prev_record.previous_state
            trans_ts = prev_record.transition_timestamp

        record = TrainStateRecord(
            state=parsed_state,
            timestamp=t,
            platform_id=pid,
            previous_state=prev_state,
            transition_timestamp=trans_ts,
            source=source,
            metadata=dict(metadata or {})
        )
        self._platforms[pid] = record

        if prev_state != parsed_state:
            logger.info(
                f"[TrainStateProvider] Platform '{pid}' transition: {prev_state} -> {parsed_state} "
                f"at {t:.3f} (source={source})"
            )
        else:
            logger.debug(
                f"[TrainStateProvider] Platform '{pid}' heartbeat state={parsed_state} at {t:.3f}"
            )

        return record

    def touch(self, platform_id: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        """Update last heartbeat timestamp without changing state."""
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        t = float(timestamp) if timestamp is not None else time.time()
        rec = self._platforms.get(pid)
        if rec:
            rec.timestamp = t

    def handle_mqtt_message(
        self,
        topic: str,
        payload: Union[str, bytes, Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> Optional[TrainStateRecord]:
        """Parses and applies an incoming MQTT message.

        Accepted topics:
        - "metro/train_state/<platform_id>"
        - "trains/<platform_id>/state"
        - "delhi_metro/patel_chowk/<platform_id>/train_state"
        - "metro/train_state" (uses default_platform_id)

        Payload formats:
        - JSON: {"state": "STOPPED_DOORS_OPEN", "platform_id": "PL_01", ...}
        - Plain text string: "STOPPED_DOORS_OPEN"
        """
        t = float(timestamp) if timestamp is not None else time.time()

        raw_payload = payload
        if isinstance(raw_payload, (bytes, bytearray)):
            try:
                raw_payload = raw_payload.decode("utf-8")
            except Exception:
                raw_payload = str(raw_payload)

        data: Dict[str, Any] = {}
        if isinstance(raw_payload, dict):
            data = raw_payload
        elif isinstance(raw_payload, str):
            trimmed = raw_payload.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                except Exception:
                    data = {"state": trimmed}
            else:
                data = {"state": trimmed}
        else:
            data = {"state": str(raw_payload)}

        platform_id = data.get("platform_id")
        if not platform_id and topic:
            parts = [p for p in topic.strip("/").split("/") if p]
            pl_part = next((p for p in parts if p.startswith("PL_")), None)
            if pl_part:
                platform_id = pl_part
            elif len(parts) >= 3 and parts[0] == "metro" and parts[1] == "train_state":
                platform_id = parts[2]
            elif len(parts) >= 3 and parts[0] == "trains" and parts[2] == "state":
                platform_id = parts[1]

        pid = platform_id or self.default_platform_id
        state_str = data.get("state") or data.get("train_state")
        if not state_str:
            logger.warning(f"[SimulatedProvider] MQTT message on topic '{topic}' missing state: {payload}")
            return None

        msg_ts = data.get("timestamp")
        eff_ts = t
        if msg_ts is not None:
            try:
                import math
                cand_ts = float(msg_ts)
                if math.isfinite(cand_ts) and 0.0 <= cand_ts <= (t + 300.0):
                    eff_ts = cand_ts
                else:
                    logger.warning(
                        f"[SimulatedProvider] Invalid or far-future MQTT timestamp {cand_ts} (receipt {t}), using receipt ts"
                    )
            except (ValueError, TypeError):
                eff_ts = t

        return self.set_state(
            state=state_str,
            platform_id=pid,
            timestamp=eff_ts,
            source=f"mqtt:{topic}",
            metadata={"topic": topic, "raw": data}
        )

    simulate_mqtt_message = handle_mqtt_message

    def connect_mqtt(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "metro/train_state/#",
        client_id: Optional[str] = None
    ):
        """Optional MQTT client integration using paho-mqtt."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.info("[SimulatedProvider] paho-mqtt not installed; running in local simulation mode.")
            return None

        try:
            cid = client_id or f"metro_train_sub_{int(time.time())}"
            client = mqtt.Client(client_id=cid)

            def on_connect(c, userdata, flags, rc):
                if rc == 0:
                    logger.info(f"[SimulatedProvider] MQTT connected to {broker}:{port}, subscribing to {topic}")
                    c.subscribe(topic)
                else:
                    logger.warning(f"[SimulatedProvider] MQTT connect failed with rc={rc}")

            def on_message(c, userdata, msg):
                try:
                    self.handle_mqtt_message(msg.topic, msg.payload.decode("utf-8"))
                except Exception as ex:
                    logger.warning(f"[SimulatedProvider] Error handling MQTT msg on {msg.topic}: {ex}")

            client.on_connect = on_connect
            client.on_message = on_message
            client.connect_async(broker, port, keepalive=60)
            client.loop_start()
            self._mqtt_client = client
            return client
        except Exception as e:
            logger.warning(f"[SimulatedProvider] Could not connect to MQTT broker {broker}:{port}: {e}")
            return None


SimulatedTrainStateProvider = SimulatedProvider


class VisionProvider(TrainStateProvider):
    """Vision-based train presence detector (stub implementation)."""

    def __init__(
        self,
        track_polygon: Optional[np.ndarray] = None,
        default_platform_id: str = "PL_DOWN_01",
        area_threshold: float = 0.25,
        grace_period_sec: float = 2.0,
        fail_safe_timeout_sec: float = 10.0
    ):
        super().__init__(
            default_platform_id=default_platform_id,
            grace_period_sec=grace_period_sec,
            fail_safe_timeout_sec=fail_safe_timeout_sec
        )
        self.track_polygon = track_polygon
        self.area_thresh = area_threshold
        self._platforms: Dict[str, TrainStateRecord] = {}
        self.set_state(
            state=TrainState.ABSENT,
            platform_id=self.default_platform_id,
            timestamp=time.time(),
            source="vision_init"
        )
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
        self.state_history = []
        self._bg_subtractors: Dict[str, Any] = {self.default_platform_id: self.bg_subtractor}
        self._state_histories: Dict[str, list] = {self.default_platform_id: self.state_history}

    def get_state_record(self, platform_id: Optional[str] = None) -> Optional[TrainStateRecord]:
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        return self._platforms.get(pid)

    def set_state(
        self,
        state: Union[TrainState, str],
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        source: str = "vision",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainStateRecord:
        import math
        if timestamp is not None:
            try:
                t_val = float(timestamp)
                if not math.isfinite(t_val):
                    t = time.time()
                else:
                    t = t_val
            except (ValueError, TypeError):
                t = time.time()
        else:
            t = time.time()

        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        self._last_platform_id = pid
        parsed_state = TrainState.from_str(state) if hasattr(TrainState, "from_str") else TrainState(state)

        prev_record = self._platforms.get(pid)
        is_external = source.startswith("mqtt") or source.startswith("external") or source == "vision_frame"
        if prev_record is not None and is_external and prev_record.source not in ("init", "vision_init") and not str(source).startswith("reset"):
            if t < prev_record.timestamp:
                logger.warning(
                    f"[VisionProvider] Rejecting stale observation for '{pid}': "
                    f"t={t:.3f} < stored={prev_record.timestamp:.3f} (source={source})"
                )
                return prev_record

        if prev_record is None:
            prev_state = None
            trans_ts = t
        elif prev_record.state != parsed_state:
            prev_state = prev_record.state
            trans_ts = t
        else:
            prev_state = prev_record.previous_state
            trans_ts = prev_record.transition_timestamp

        record = TrainStateRecord(
            state=parsed_state,
            timestamp=t,
            platform_id=pid,
            previous_state=prev_state,
            transition_timestamp=trans_ts,
            source=source,
            metadata=dict(metadata or {})
        )
        self._platforms[pid] = record
        if prev_state != parsed_state:
            logger.info(
                f"[VisionProvider] Platform '{pid}' transition: {prev_state} -> {parsed_state} at {t:.3f} (source={source})"
            )
        return record

    def touch(self, platform_id: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        t = float(timestamp) if timestamp is not None else time.time()
        rec = self._platforms.get(pid)
        if rec:
            rec.timestamp = t

    def get_state(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None
    ) -> TrainState:
        pid = platform_id or getattr(self, "_last_platform_id", None) or self.default_platform_id
        t = float(timestamp) if timestamp is not None else time.time()
        if frame is None:
            rec = self.get_state_record(pid)
            return rec.state if rec else TrainState.ABSENT

        # If track polygon is configured and frame is provided, compute vision heuristic
        if self.track_polygon is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [self.track_polygon], 255)
            track_area = max(1.0, float(np.sum(mask > 0)))

            rec = self.get_state_record(pid)
            is_stationary = bool(rec and rec.state in (TrainState.STOPPED_DOORS_OPEN, TrainState.PRESENT))
            lr = 0.0 if is_stationary else -1

            if pid not in self._bg_subtractors:
                self._bg_subtractors[pid] = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
            if pid not in self._state_histories:
                self._state_histories[pid] = []
            bg_sub = self._bg_subtractors[pid]
            hist = self._state_histories[pid]

            fg_mask = bg_sub.apply(gray, learningRate=lr)
            fg_track = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
            motion_ratio = float(np.sum(fg_track > 0)) / track_area

            if motion_ratio > self.area_thresh:
                curr_state = TrainState.STOPPED_DOORS_OPEN
            elif motion_ratio > (self.area_thresh * 0.4):
                curr_state = TrainState.APPROACHING
            else:
                curr_state = TrainState.ABSENT

            hist.append(curr_state)
            if len(hist) > 10:
                hist.pop(0)

            from collections import Counter
            counts = Counter(hist)
            voted = counts.most_common(1)[0][0]
            self.set_state(voted, platform_id=pid, timestamp=t, source="vision_frame")
            return voted

        rec = self.get_state_record(pid)
        return rec.state if rec else TrainState.ABSENT


VisionTrainStateProvider = VisionProvider


class ScheduleFileTrainStateProvider(TrainStateProvider):
    """Provider reading train arrival / departure timetable intervals."""

    def __init__(
        self,
        schedule_file: Union[Path, str],
        default_platform_id: str = "PL_DOWN_01",
        grace_period_sec: float = 2.0,
        fail_safe_timeout_sec: float = 10.0
    ):
        super().__init__(
            default_platform_id=default_platform_id,
            grace_period_sec=grace_period_sec,
            fail_safe_timeout_sec=fail_safe_timeout_sec
        )
        self.schedule_file = Path(schedule_file)
        self.intervals = []
        self._load_schedule()
        self._platforms: Dict[str, TrainStateRecord] = {}
        self.set_state(
            state=TrainState.ABSENT,
            platform_id=self.default_platform_id,
            timestamp=time.time(),
            source="schedule_init"
        )

    def _load_schedule(self):
        if not self.schedule_file.exists():
            return
        with open(self.schedule_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.intervals = data
            elif isinstance(data, dict):
                self.intervals = data.get("train_events", [])
            else:
                self.intervals = []

    def get_state_record(self, platform_id: Optional[str] = None) -> Optional[TrainStateRecord]:
        pid = platform_id or self.default_platform_id
        return self._platforms.get(pid)

    def set_state(
        self,
        state: Union[TrainState, str],
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        source: str = "schedule",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainStateRecord:
        t = float(timestamp) if timestamp is not None else time.time()
        pid = platform_id or self.default_platform_id
        parsed_state = TrainState.from_str(state) if hasattr(TrainState, "from_str") else TrainState(state)

        prev_record = self._platforms.get(pid)
        if prev_record is None:
            prev_state = None
            trans_ts = t
        elif prev_record.state != parsed_state:
            prev_state = prev_record.state
            trans_ts = t
        else:
            prev_state = prev_record.previous_state
            trans_ts = prev_record.transition_timestamp

        record = TrainStateRecord(
            state=parsed_state,
            timestamp=t,
            platform_id=pid,
            previous_state=prev_state,
            transition_timestamp=trans_ts,
            source=source,
            metadata=dict(metadata or {})
        )
        self._platforms[pid] = record
        return record

    def get_state(
        self,
        platform_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None
    ) -> TrainState:
        t = timestamp if timestamp is not None else time.time()
        pid = platform_id or self.default_platform_id
        resolved_state = TrainState.ABSENT

        for item in self.intervals:
            app = item.get("approaching")
            pres = item.get("present") or item.get("stopped_doors_open")
            dep = item.get("departing")
            if dep and dep[0] <= t <= dep[1]:
                resolved_state = TrainState.DEPARTING
                break
            if pres and pres[0] <= t <= pres[1]:
                resolved_state = TrainState.STOPPED_DOORS_OPEN
                break
            if app and app[0] <= t <= app[1]:
                resolved_state = TrainState.APPROACHING
                break

        self.set_state(resolved_state, platform_id=pid, timestamp=t, source="timetable")
        return resolved_state
