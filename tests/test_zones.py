"""Unit tests for Zone Geometry logic."""
import pytest
import numpy as np
from metro_safety.zones.geometry import Zone, ZoneManager

@pytest.fixture
def zone_manager():
    cfg = {
        "platform": [[0, 0], [640, 0], [640, 200], [0, 200]],
        "yellow_band": [[0, 200], [640, 200], [640, 250], [0, 250]],
        "track_bed": [[0, 250], [640, 250], [640, 480], [0, 480]]
    }
    return ZoneManager(cfg)

def test_point_in_platform(zone_manager):
    assert zone_manager.get_zone_for_point((300, 100)) == "platform"

def test_point_in_yellow_band(zone_manager):
    assert zone_manager.get_zone_for_point((300, 225)) == "yellow_band"

def test_point_in_track_bed(zone_manager):
    assert zone_manager.get_zone_for_point((300, 350)) == "track_bed"

def test_is_in_zone(zone_manager):
    assert zone_manager.is_in_zone((300, 225), "yellow_band") is True
    assert zone_manager.is_in_zone((300, 225), "platform") is False
