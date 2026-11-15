"""Learned Temporal Classifier for FALL vs JUMP vs NORMAL vs CROSSING.
Extracts engineered kinematic feature sequences across a temporal window
and trains / predicts via Gradient Boosting / Random Forest.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from metro_safety.perception.base import Detection
from metro_safety.events.models import EventType, SafetyEvent
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "learned_model.pkl"

class FeatureExtractor:
    """Extracts normalized kinematic trajectory features from keypoint sequences."""

    @staticmethod
    def extract_sequence_features(detections: List[Detection], window_size: int = 15) -> np.ndarray:
        """Extract a rich fixed-length feature vector from window of detections."""
        if not detections:
            return np.zeros(32, dtype=np.float32)

        # Pad sequence if smaller than window
        dets = list(detections)
        while len(dets) < window_size:
            dets.insert(0, dets[0])
        dets = dets[-window_size:]

        # 1. Torso centers and Foot anchors
        torso_pts = np.array([d.torso_center for d in dets])    # shape (W, 2)
        foot_pts = np.array([d.foot_anchor for d in dets])      # shape (W, 2)
        angles = np.array([d.torso_angle for d in dets])        # shape (W,)

        # 2. Velocities and accelerations
        torso_vel = np.diff(torso_pts, axis=0)                  # shape (W-1, 2)
        torso_acc = np.diff(torso_vel, axis=0)                  # shape (W-2, 2)
        foot_vel = np.diff(foot_pts, axis=0)                    # shape (W-1, 2)

        # 3. Summary statistics
        mean_angle = float(np.mean(angles))
        min_angle = float(np.min(angles))
        angle_change = float(angles[-1] - angles[0])

        max_drop_vel = float(np.max(torso_vel[:, 1]))           # downward velocity (y is down)
        mean_drop_vel = float(np.mean(torso_vel[:, 1]))
        max_drop_acc = float(np.max(torso_acc[:, 1])) if len(torso_acc) > 0 else 0.0

        mean_fwd_vel = float(np.mean(foot_vel[:, 1]))           # approach towards edge
        max_fwd_vel = float(np.max(foot_vel[:, 1]))
        horiz_displacement = float(np.sum(np.abs(foot_vel[:, 0])))

        aspect_ratios = []
        for d in dets:
            w = max(1.0, d.box[2] - d.box[0])
            h = max(1.0, d.box[3] - d.box[1])
            aspect_ratios.append(h / w)
        mean_ar = float(np.mean(aspect_ratios))
        min_ar = float(np.min(aspect_ratios))

        # Recent vs older angle variance (loss of balance indicator)
        angle_var = float(np.var(angles[-6:]))

        feats = [
            mean_angle, min_angle, angle_change, angle_var,
            max_drop_vel, mean_drop_vel, max_drop_acc,
            mean_fwd_vel, max_fwd_vel, horiz_displacement,
            mean_ar, min_ar,
            torso_pts[-1, 1], foot_pts[-1, 1]  # absolute depth in frame
        ]

        # Pad or trim to exactly 16 features
        feats_arr = np.array(feats, dtype=np.float32)
        if len(feats_arr) < 32:
            feats_arr = np.pad(feats_arr, (0, 32 - len(feats_arr)), mode='constant')
        return feats_arr[:32]


class LearnedEventClassifier:
    """Trained temporal classifier comparing against rule-based state machine."""

    CLASSES = ["NORMAL", "LINE_CROSSING", "ACCIDENTAL_FALL", "DELIBERATE_JUMP"]

    def __init__(self, model_path: Optional[Path | str] = None, window_size: int = 15):
        self.model_path = Path(model_path) if model_path else MODEL_SAVE_PATH
        self.window_size = window_size
        self.model: Optional[GradientBoostingClassifier] = None
        self.history: Dict[int, List[Detection]] = {}
        self._load_or_train_model()

    def _load_or_train_model(self):
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.train_on_synthetic_data()

    def train_on_synthetic_data(self, samples_per_class: int = 100):
        """Train classifier on synthetic kinematic trajectories."""
        from metro_safety.simulation.generator import ScenarioGenerator

        gen = ScenarioGenerator()
        X_train = []
        y_train = []

        class_map = {
            "NORMAL": ["NORMAL_WAITING", "NORMAL_WALKING"],
            "LINE_CROSSING": ["LINE_CROSSING"],
            "ACCIDENTAL_FALL": ["ACCIDENTAL_FALL"],
            "DELIBERATE_JUMP": ["DELIBERATE_JUMP"]
        }

        for label_idx, cls_name in enumerate(self.CLASSES):
            scenarios = class_map.get(cls_name, ["NORMAL_WAITING"])
            for _ in range(samples_per_class):
                sc = np.random.choice(scenarios)
                # Randomize length
                n_frames = np.random.randint(30, 60)
                data = gen.generate_person_trajectory(sc, num_frames=n_frames)
                dets = [Detection(box=d["box"], confidence=d["confidence"], keypoints=d["keypoints"]) for d in data]

                # Extract features from windows around incident
                for start in range(5, max(6, len(dets) - self.window_size)):
                    window = dets[start : start + self.window_size]
                    feats = FeatureExtractor.extract_sequence_features(window, self.window_size)
                    # Add subtle jitter for robustness
                    feats += np.random.normal(0, 0.05, feats.shape)
                    X_train.append(feats)
                    y_train.append(label_idx)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        clf = GradientBoostingClassifier(n_estimators=75, learning_rate=0.1, max_depth=4, random_state=42)
        clf.fit(X_train, y_train)
        self.model = clf

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def process_frame(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_detections: List[Detection],
        zone_manager: ZoneManager,
        train_state: TrainState
    ) -> List[SafetyEvent]:
        """Predict events using learned temporal feature model."""
        if self.model is None:
            return []

        events = []
        active_ids = set()

        for det in tracked_detections:
            if det.track_id is None:
                continue

            # Strict zone semantics: ignore detections outside monitored zones or in 'ignore' zone
            foot = det.foot_anchor
            zone = zone_manager.get_zone_for_point(foot)
            if zone in ["outside", "ignore"]:
                continue

            tid = det.track_id
            active_ids.add(tid)

            if tid not in self.history:
                self.history[tid] = []
            self.history[tid].append(det)
            if len(self.history[tid]) > self.window_size * 2:
                self.history[tid].pop(0)

            if len(self.history[tid]) >= self.window_size:
                window = self.history[tid][-self.window_size:]
                feats = FeatureExtractor.extract_sequence_features(window, self.window_size)
                probs = self.model.predict_proba([feats])[0]
                pred_idx = int(np.argmax(probs))
                conf = min(0.99, float(probs[pred_idx]))
                predicted_label = self.CLASSES[pred_idx]

                if predicted_label != "NORMAL" and conf > 0.60:
                    ev_type = EventType(predicted_label)
                    foot = det.foot_anchor
                    zone = zone_manager.get_zone_for_point(foot)
                    ev = SafetyEvent(
                        event_type=ev_type,
                        confidence=conf,
                        track_id=tid,
                        timestamp=timestamp,
                        zone=zone,
                        train_state=train_state.value,
                        evidence_frames=[frame_idx],
                        metadata={"classifier": "learned_gbdt", "probs": {c: round(float(p), 3) for c, p in zip(self.CLASSES, probs)}}
                    )
                    events.append(ev)

        # Cleanup dead tracks
        for dead_id in list(self.history.keys()):
            if dead_id not in active_ids:
                del self.history[dead_id]

        return events

