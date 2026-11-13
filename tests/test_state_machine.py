"""Unit tests for Event State Machine and Temporal Logic."""
import pytest
import numpy as np
from metro_safety.events.rule_classifier import RuleBasedEventClassifier
from metro_safety.events.models import EventType
from metro_safety.perception.base import Detection
from metro_safety.zones.geometry import ZoneManager
from metro_safety.perception.train_state import TrainState

@pytest.fixture
def setup_env():
    cfg = {
        "platform": [[0, 0], [640, 0], [640, 200], [0, 200]],
        "yellow_band": [[0, 200], [640, 200], [640, 250], [0, 250]],
        "track_bed": [[0, 250], [640, 250], [640, 480], [0, 480]]
    }
    zm = ZoneManager(cfg)
    clf = RuleBasedEventClassifier(line_crossing_thresh=3, track_intrusion_thresh=2)
    return zm, clf

def test_line_crossing_hysteresis(setup_env):
    zm, clf = setup_env
    # 2 frames in yellow band (below threshold of 3)
    for f in range(2):
        d = Detection(box=np.array([100, 210, 140, 240]), confidence=0.9, track_id=1)
        evs = clf.process_frame(f, f*0.04, [d], zm, TrainState.ABSENT)
        assert len(evs) == 0

    # 3rd frame (hits threshold)
    d = Detection(box=np.array([100, 210, 140, 240]), confidence=0.9, track_id=1)
    evs = clf.process_frame(2, 0.08, [d], zm, TrainState.ABSENT)
    assert len(evs) == 1
    assert evs[0].event_type == EventType.LINE_CROSSING

def test_track_intrusion_detection(setup_env):
    zm, clf = setup_env
    # 3 frames on track bed
    events = []
    for f in range(3):
        d = Detection(box=np.array([100, 300, 140, 360]), confidence=0.9, track_id=2)
        evs = clf.process_frame(f, f*0.04, [d], zm, TrainState.ABSENT)
        events.extend(evs)
    assert len(events) >= 1
    assert events[-1].event_type in [EventType.TRACK_INTRUSION, EventType.ACCIDENTAL_FALL]
