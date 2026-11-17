import sys
sys.path.insert(0, ".")
import numpy as np
from metro_safety.events.gesture import DistressGestureDetector
from metro_safety.events.models import EventType
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState

zm = ZoneManager({"platform": [[0,0],[640,0],[640,480],[0,480]]})

def make_det(kpts):
    return Detection(box=np.array([280, 100, 360, 400]), confidence=0.9, keypoints=kpts, track_id=1)

# Case 1: Standard overhead cross (test_gesture.py)
det = DistressGestureDetector(min_persistence=3)
events1 = []
for f in range(4):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [320, 150, 0.9]  # nose
    kpts[5] = [300, 200, 0.9]  # L sh
    kpts[6] = [340, 200, 0.9]  # R sh
    kpts[9] = [325, 120, 0.9]  # L wrist crossed
    kpts[10] = [315, 120, 0.9] # R wrist crossed
    evs = det.process_frame(f, f*0.04, [make_det(kpts)], zm, TrainState.ABSENT)
    events1.extend(evs)
print("Case 1 (Overhead cross):", len(events1) > 0, events1[0].metadata["gesture_variant"] if events1 else None)

# Case 2: Chest X cross
det2 = DistressGestureDetector(min_persistence=3)
events2 = []
for f in range(4):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [320, 150, 0.9]  # nose
    kpts[5] = [300, 200, 0.9]  # L sh
    kpts[6] = [340, 200, 0.9]  # R sh
    kpts[7] = [290, 230, 0.9]  # L elbow
    kpts[8] = [350, 230, 0.9]  # R elbow
    kpts[9] = [330, 190, 0.9]  # L wrist crossed to right
    kpts[10] = [310, 190, 0.9] # R wrist crossed to left
    evs = det2.process_frame(f, f*0.04, [make_det(kpts)], zm, TrainState.ABSENT)
    events2.extend(evs)
print("Case 2 (Chest X cross):", len(events2) > 0, events2[0].metadata["gesture_variant"] if events2 else None)

# Case 3: Parallel hands raised (artificial cross) -> MUST BE REJECTED
det3 = DistressGestureDetector(min_persistence=3)
events3 = []
for f in range(4):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [320, 150, 0.9]  # nose
    kpts[5] = [300, 200, 0.9]  # L sh
    kpts[6] = [340, 200, 0.9]  # R sh
    kpts[9] = [310, 120, 0.9]  # L wrist uncrossed on left
    kpts[10] = [330, 120, 0.9] # R wrist uncrossed on right
    evs = det3.process_frame(f, f*0.04, [make_det(kpts)], zm, TrainState.ABSENT)
    events3.extend(evs)
print("Case 3 (Parallel hands - artificial cross):", len(events3) == 0, f"events={len(events3)}")

# Case 4: Occluded nose overhead cross
det4 = DistressGestureDetector(min_persistence=3)
events4 = []
for f in range(4):
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[0] = [0, 0, 0.0]      # nose occluded
    kpts[5] = [300, 200, 0.9]  # L sh
    kpts[6] = [340, 200, 0.9]  # R sh
    kpts[9] = [325, 120, 0.9]  # L wrist crossed
    kpts[10] = [315, 120, 0.9] # R wrist crossed
    evs = det4.process_frame(f, f*0.04, [make_det(kpts)], zm, TrainState.ABSENT)
    events4.extend(evs)
print("Case 4 (Occluded nose overhead cross):", len(events4) > 0, events4[0].metadata["gesture_variant"] if events4 else None)
