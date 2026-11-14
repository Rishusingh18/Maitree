"""Unit tests for Zone Geometry and Validation logic."""
import pytest
import numpy as np
from metro_safety.zones.geometry import (
    Zone, ZoneManager, validate_polygon,
    reorder_polygon_around_centroid, is_polygon_self_intersecting
)

@pytest.fixture
def zone_manager():
    cfg = {
        "platform": [[0, 0], [640, 0], [640, 200], [0, 200]],
        "yellow_band": [[0, 200], [640, 200], [640, 250], [0, 250]],
        "track_bed": [[0, 250], [640, 250], [640, 480], [0, 480]],
        "ignore": [[500, 50], [600, 50], [600, 150], [500, 150]]
    }
    return ZoneManager(cfg)

def test_point_in_platform(zone_manager):
    assert zone_manager.get_zone_for_point((300, 100)) == "platform"

def test_point_in_yellow_band(zone_manager):
    assert zone_manager.get_zone_for_point((300, 225)) == "yellow_band"

def test_point_in_track_bed(zone_manager):
    assert zone_manager.get_zone_for_point((300, 350)) == "track_bed"

def test_point_in_ignore_zone(zone_manager):
    assert zone_manager.get_zone_for_point((550, 100)) == "ignore"
    assert zone_manager.is_point_monitored((550, 100)) is False

def test_point_outside_all_zones(zone_manager):
    # Coordinate outside the defined polygons
    assert zone_manager.get_zone_for_point((1000, 1000)) == "outside"
    assert zone_manager.is_point_monitored((1000, 1000)) is False

def test_is_in_zone(zone_manager):
    assert zone_manager.is_in_zone((300, 225), "yellow_band") is True
    assert zone_manager.is_in_zone((300, 225), "platform") is False

def test_polygon_validation_rules():
    # 1. Fewer than 3 points
    valid, err = validate_polygon([[10, 10], [20, 20]], name="line")
    assert not valid
    assert "fewer than 3" in err

    # 2. Tiny area
    valid, err = validate_polygon([[0, 0], [1, 0], [0, 1]], min_area=100.0, name="tiny")
    assert not valid
    assert "too small" in err

    # 3. Bowtie self-intersecting polygon
    bowtie = [[0, 0], [100, 100], [100, 0], [0, 100]]
    valid, err = validate_polygon(bowtie, name="bowtie")
    assert not valid
    assert "self-intersecting" in err

    # 4. Auto-repair around centroid
    fixed = reorder_polygon_around_centroid(bowtie)
    valid_fixed, err_fixed = validate_polygon(fixed, name="fixed")
    assert valid_fixed
    assert err_fixed is None

def test_zone_manager_refuses_invalid_zone_silently():
    invalid_cfg = {
        "platform": [[0, 0], [640, 0], [640, 200], [0, 200]],
        "broken_zone": [[0, 0], [100, 100], [100, 0], [0, 100]] # self-intersecting
    }
    with pytest.raises(ValueError) as excinfo:
        ZoneManager(invalid_cfg, auto_fix=False)
    assert "self-intersecting" in str(excinfo.value)
