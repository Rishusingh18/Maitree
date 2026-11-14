"""Unified Anomaly and Theft Detection Engine.

Integrates:
1. PERSON_DOWN: Posture & horizontal axis analysis, energy, time-to-alert, churn recovery.
2. LOITERING: Spatial anchor radius, zone dwell limits, train-state suppression.
3. ABANDONED_OBJECT: Decimated bag detection, distance-to-person tracking, snapshot capture.
4. POSSIBLE_SNATCH: Bag-owner link, wrist snatch velocity, handover suppression, capped confidence.
5. STG_NF: Spatio-Temporal Graph Normalizing Flows pose sequence anomaly scoring.

Ground Rules:
- Anomaly events are WELFARE/REVIEW alerts for the operator.
- Max severity L2 (dashboard alert + overlooker).
- NEVER triggers simulated ETS, NEVER sounds platform L3 siren.
- While camera_motion=True or zones_unverified=True, events are flagged low-trust and capped.
- Decimated object detection to conserve CPU cycles.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import numpy as np

from metro_safety.events.models import EventType, SafetyEvent, SeverityLevel
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.anomaly.person_down import PersonDownDetector
from metro_safety.anomaly.loitering import LoiteringDetector
from metro_safety.anomaly.abandoned_object import AbandonedObjectDetector, TrackedObject
from metro_safety.anomaly.theft_snatch import TheftSnatchDetector
from metro_safety.anomaly.stg_nf_evaluator import STGNFEvaluator

logger = logging.getLogger("AnomalyEngine")


class AnomalyEngine:
    """Master orchestrator for all anomaly and welfare detection subsystems."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, object_detector: Optional[Any] = None):
        cfg = config or {}
        anomaly_cfg = cfg.get("anomaly", {})
        self.enabled = anomaly_cfg.get("enabled", True)

        # 1. Person Down
        pd_cfg = anomaly_cfg.get("person_down", {})
        self.pd_enabled = pd_cfg.get("enabled", True)
        self.person_down_detector = PersonDownDetector(
            prone_t1_sec=pd_cfg.get("prone_seconds_l1", 10.0),
            prone_t2_sec=pd_cfg.get("prone_seconds_l2", 30.0),
            horizontal_torso_angle_deg=pd_cfg.get("horizontal_torso_angle_deg", 35.0),
            max_head_hip_y_diff_px=pd_cfg.get("max_head_hip_y_diff_px", 45.0),
            min_aspect_ratio=pd_cfg.get("aspect_ratio_threshold", 1.15),
            max_motion_energy=pd_cfg.get("motion_energy_threshold", 4.0),
            track_reassoc_window_sec=pd_cfg.get("reassociation_window_sec", 3.0),
            ignore_bench_lying=pd_cfg.get("ignore_bench_lying", True)
        )

        # 2. Loitering / Dwell
        loit_cfg = anomaly_cfg.get("loitering", {})
        self.loitering_enabled = loit_cfg.get("enabled", True)
        self.loitering_detector = LoiteringDetector(
            radius_px=loit_cfg.get("anchor_radius_px", 70.0),
            dwell_t1_sec=loit_cfg.get("dwell_seconds_l1", 30.0),
            dwell_t2_sec=loit_cfg.get("dwell_seconds_l2", 60.0),
            train_suppression=loit_cfg.get("suppress_on_train", True)
        )

        # 3. Abandoned Object (Decimated bag detection)
        ao_cfg = anomaly_cfg.get("abandoned_object", {})
        self.ao_enabled = ao_cfg.get("enabled", True)
        self.abandoned_object_detector = AbandonedObjectDetector(
            detect_interval_frames=ao_cfg.get("detect_interval_frames", 10),
            unattended_radius_px=ao_cfg.get("unattended_radius_px", 130.0),
            unattended_sec=ao_cfg.get("unattended_seconds", 15.0),
            alert_startup_objects=ao_cfg.get("alert_startup_objects", False),
            confidence_thresh=ao_cfg.get("confidence_threshold", 0.20)
        )
        if object_detector is not None:
            self.abandoned_object_detector.yolo_model = object_detector

        # 4. Theft / Snatch Prototype
        snatch_cfg = anomaly_cfg.get("theft_snatch", {})
        self.snatch_enabled = snatch_cfg.get("enabled", True)
        self.theft_detector = TheftSnatchDetector(
            owner_proximity_px=snatch_cfg.get("owner_proximity_px", 65.0),
            owner_min_frames=snatch_cfg.get("owner_min_frames", 4),
            suspect_wrist_proximity_px=snatch_cfg.get("suspect_wrist_proximity_px", 55.0),
            snatch_accel_threshold_px=snatch_cfg.get("snatch_accel_threshold_px", 10.0),
            owner_follow_window_sec=snatch_cfg.get("owner_follow_window_sec", 1.5),
            handover_max_dist_px=snatch_cfg.get("handover_max_dist_px", 85.0),
            max_confidence=snatch_cfg.get("max_confidence", 0.70)
        )

        # 5. STG-NF Pose Sequence Anomaly
        stg_cfg = anomaly_cfg.get("stg_nf", {})
        self.stg_enabled = stg_cfg.get("enabled", True)
        self.stg_evaluator: Optional[STGNFEvaluator] = None
        if self.stg_enabled:
            ckpt_path = stg_cfg.get("checkpoint_path", "data/models/ShanghaiTech_85_9.tar")
            if Path(ckpt_path).exists():
                try:
                    self.stg_evaluator = STGNFEvaluator(
                        checkpoint_path=ckpt_path,
                        device="cpu",
                        anomaly_nll_thresh=stg_cfg.get("nll_threshold", 16.0),
                        window_len=stg_cfg.get("temporal_window", 24)
                    )
                    logger.info("STG-NF Anomaly model loaded successfully.")
                except Exception as e:
                    logger.warning(f"Failed to load STG-NF model from {ckpt_path}: {e}")
            else:
                logger.info(f"STG-NF checkpoint not found at {ckpt_path}; pose flow evaluator disabled.")

        # Dedup cooldown tracking: (key, event_type) -> last_alert_ts
        self._dedup_cache: Dict[Tuple[str, str], float] = {}
        self.dedup_cooldown_sec = anomaly_cfg.get("dedup_cooldown_sec", 15.0)

    def set_object_detector(self, detector: Any):
        """Assign or update YOLO detector for luggage classes."""
        self.abandoned_object_detector.yolo_model = detector

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        frame_idx: int,
        person_detections: List[Detection],
        zone_manager: ZoneManager,
        timestamp: Optional[float] = None,
        train_state: str = "ABSENT",
        camera_motion: bool = False,
        motion_reason: str = "",
        zones_unverified: bool = False,
        zone_version: int = 1,
        active_falls_map: Optional[Dict[int, str]] = None,
        synthetic_objects: Optional[List[Dict[str, Any]]] = None
    ) -> List[SafetyEvent]:
        """Process visual detections across all anomaly detection modules.

        Returns:
            List of generated SafetyEvents (all strictly welfare/review, max severity L2).
        """
        if not self.enabled:
            return []

        now = timestamp if timestamp is not None else time.time()
        emitted_events: List[SafetyEvent] = []

        # 1. Person Down Detection
        if self.pd_enabled:
            pd_events = self.person_down_detector.process_frame(
                frame_idx=frame_idx,
                timestamp=now,
                tracked_dets=person_detections,
                zone_manager=zone_manager,
                train_state=train_state,
                active_falls_map=active_falls_map
            )
            for ev in pd_events:
                ev.camera_motion = camera_motion
                ev.motion_reason = motion_reason
                ev.zones_unverified = zones_unverified
                ev.zone_version = zone_version
            emitted_events.extend(pd_events)

        # 2. Loitering Detection
        if self.loitering_enabled:
            loit_events = self.loitering_detector.process_frame(
                frame_idx=frame_idx,
                timestamp=now,
                tracked_dets=person_detections,
                zone_manager=zone_manager,
                train_state=train_state
            )
            for ev in loit_events:
                ev.camera_motion = camera_motion
                ev.motion_reason = motion_reason
                ev.zones_unverified = zones_unverified
                ev.zone_version = zone_version
            emitted_events.extend(loit_events)

        # 3. Abandoned Object Detection (YOLO bags on decimated frames)
        if self.ao_enabled:
            ao_events = self.abandoned_object_detector.process_frame(
                frame=frame,
                frame_idx=frame_idx,
                timestamp=now,
                tracked_persons=person_detections,
                zone_manager=zone_manager,
                train_state=train_state,
                synthetic_objects=synthetic_objects
            )
            for ev in ao_events:
                ev.camera_motion = camera_motion
                ev.motion_reason = motion_reason
                ev.zones_unverified = zones_unverified
                ev.zone_version = zone_version
            emitted_events.extend(ao_events)

        # 4. Theft / Snatch Prototype
        tracked_bags = list(self.abandoned_object_detector.tracked_objects.values())
        if self.snatch_enabled and tracked_bags:
            snatch_events = self.theft_detector.update(
                tracked_bags=tracked_bags,
                person_detections=person_detections,
                timestamp=now,
                camera_motion=camera_motion,
                motion_reason=motion_reason,
                zones_unverified=zones_unverified,
                zone_version=zone_version
            )
            emitted_events.extend(snatch_events)

        # 5. STG-NF Normalizing Flows Pose Anomaly
        if self.stg_enabled and self.stg_evaluator is not None:
            for det in person_detections:
                if det.track_id is not None and det.keypoints is not None:
                    self.stg_evaluator.update_track(det.track_id, det.keypoints)
                    is_anom, nll, conf = self.stg_evaluator.evaluate_track(det.track_id)
                    if is_anom:
                        meta = {
                            "anomaly_type": "ANOMALY_POSE",
                            "severity_cap": "L1",
                            "nll_score": round(float(nll), 2),
                            "disclaimer": "operator welfare review",
                            "low_trust": camera_motion or zones_unverified,
                            "low_trust_reason": motion_reason if camera_motion else ""
                        }
                        center = ((det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0)
                        zone = zone_manager.get_zone_for_point(center) or "platform"
                        ev = SafetyEvent(
                            event_type=EventType.ANOMALY_POSE,
                            confidence=min(0.75, conf),
                            track_id=det.track_id,
                            timestamp=now,
                            zone=zone,
                            train_state=train_state,
                            metadata=meta,
                            camera_motion=camera_motion,
                            motion_reason=motion_reason,
                            zones_unverified=zones_unverified,
                            zone_version=zone_version
                        )
                        emitted_events.append(ev)

        # Deduplicate and apply cooldowns
        final_events: List[SafetyEvent] = []
        for ev in emitted_events:
            ev.metadata["is_anomaly"] = True
            if "disclaimer" not in ev.metadata:
                ev.metadata["disclaimer"] = "operator welfare review"

            key = (str(ev.track_id), ev.event_type.value)
            last_ts = self._dedup_cache.get(key, 0.0)
            if (now - last_ts) >= self.dedup_cooldown_sec:
                self._dedup_cache[key] = now
                final_events.append(ev)

        return final_events
