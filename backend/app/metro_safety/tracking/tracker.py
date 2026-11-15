"""Multi-object tracker implementation (ByteTrack / IoU + Kalman Filter)."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from metro_safety.perception.base import Detection

def box_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter_area / float(boxA_area + boxB_area - inter_area)

class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    """Single object track state representation."""
    _count = 0

    def __init__(self, detection: Detection, track_id: Optional[int] = None):
        if track_id is not None:
            self.track_id = track_id
        else:
            STrack._count += 1
            self.track_id = STrack._count
        self.box = detection.box.copy()
        self.confidence = detection.confidence
        self.keypoints = detection.keypoints.copy() if detection.keypoints is not None else None
        self.state = TrackState.New
        self.time_since_update = 0
        self.history: List[Detection] = [detection]
        self.vx = 0.0
        self.vy = 0.0
        self.last_cx = (detection.box[0] + detection.box[2]) / 2.0
        self.last_cy = (detection.box[1] + detection.box[3]) / 2.0

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def predict(self):
        """Simple velocity-based prediction."""
        self.time_since_update += 1
        self.box[0] += self.vx
        self.box[1] += self.vy
        self.box[2] += self.vx
        self.box[3] += self.vy

    def update(self, detection: Detection):
        """Update track with matched detection."""
        # Compute velocity from successive measured positions normalized by elapsed frames
        curr_cx = (detection.box[0] + detection.box[2]) / 2.0
        curr_cy = (detection.box[1] + detection.box[3]) / 2.0
        dt = max(1, self.time_since_update)
        measured_vx = (curr_cx - self.last_cx) / float(dt)
        measured_vy = (curr_cy - self.last_cy) / float(dt)

        self.vx = 0.6 * self.vx + 0.4 * measured_vx
        self.vy = 0.6 * self.vy + 0.4 * measured_vy
        self.last_cx = curr_cx
        self.last_cy = curr_cy

        self.box = detection.box.copy()
        self.confidence = detection.confidence
        self.keypoints = detection.keypoints.copy() if detection.keypoints is not None else None
        self.state = TrackState.Tracked
        self.time_since_update = 0
        detection.track_id = self.track_id
        self.history.append(detection)
        if len(self.history) > 60:
            self.history.pop(0)

class ByteTracker:
    """ByteTrack-style multi-object tracker for CCTV monitoring."""

    def __init__(
        self,
        track_high_thresh: float = 0.45,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.4
    ):
        self.high_thresh = track_high_thresh
        self.low_thresh = track_low_thresh
        self.new_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self._track_id_counter = 0

    def _next_id(self) -> int:
        self._track_id_counter += 1
        return self._track_id_counter

    def reset(self):
        self.tracked_tracks = []
        self.lost_tracks = []
        self._track_id_counter = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracker with frame detections, assigning persistent track_ids."""
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # Stage 1: Associate high confidence detections with active tracks
        unmatched_tracks, unmatched_dets = self._associate(self.tracked_tracks, high_dets, self.match_thresh)

        # Stage 2: Associate remaining tracks with low confidence detections
        unmatched_tracks, unmatched_low = self._associate(unmatched_tracks, low_dets, self.match_thresh * 0.7)

        # Mark unmatched tracks as lost or removed
        new_lost: List[STrack] = []
        for t in unmatched_tracks:
            if t.time_since_update > self.track_buffer:
                t.state = TrackState.Removed
            else:
                t.state = TrackState.Lost
                new_lost.append(t)

        # Stage 3: Associate newly unmatched high-confidence detections with lost tracks
        unmatched_lost, remaining_dets = self._associate(self.lost_tracks, unmatched_dets, self.match_thresh)
        for t in self.lost_tracks:
            if t not in unmatched_lost and t.state == TrackState.Tracked:
                self.tracked_tracks.append(t)

        for t in unmatched_lost:
            if t.time_since_update > self.track_buffer:
                t.state = TrackState.Removed
            else:
                new_lost.append(t)

        # Initialize new tracks for remaining high confidence detections
        for det in remaining_dets:
            if det.confidence >= self.new_thresh:
                new_t = STrack(det, track_id=self._next_id())
                new_t.state = TrackState.Tracked
                det.track_id = new_t.track_id
                self.tracked_tracks.append(new_t)

        self.lost_tracks = [t for t in new_lost if t.state == TrackState.Lost]
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]

        # Return tracked detections
        output_detections: List[Detection] = []
        for t in self.tracked_tracks:
            latest_det = t.history[-1]
            latest_det.track_id = t.track_id
            output_detections.append(latest_det)

        return output_detections

    def _associate(
        self,
        tracks: List[STrack],
        dets: List[Detection],
        threshold: float
    ) -> Tuple[List[STrack], List[Detection]]:
        """IoU Hungarian / greedy association."""
        if not tracks or not dets:
            return tracks, dets

        # Build cost/iou matrix
        iou_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = box_iou(t.box, d.box)

        matched_tracks = set()
        matched_dets = set()

        # Greedy highest-IoU matching
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < threshold:
                break

            tr_idx, det_idx = max_idx
            tracks[tr_idx].update(dets[det_idx])
            matched_tracks.add(tr_idx)
            matched_dets.add(det_idx)
            iou_matrix[tr_idx, :] = -1.0
            iou_matrix[:, det_idx] = -1.0

        unmatched_tr = [tracks[i] for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dt = [dets[j] for j in range(len(dets)) if j not in matched_dets]
        return unmatched_tr, unmatched_dt
