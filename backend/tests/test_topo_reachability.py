"""Tests for Tobler/Dijkstra reachability (topo_layout parity)."""

from __future__ import annotations

import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid
from app.models.mission import LatLon
from app.models.mission import BASE_STEP_SEC
from app.services.topo_reachability import (
    compute_reachability,
    compute_reachability_score,
    least_travel_time_hours,
    mission_max_hours,
    tobler_hiking_speed_kmh,
    travel_time_to_probability,
)

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def test_tobler_uphill_slower_than_flat():
    flat = float(tobler_hiking_speed_kmh(0.0))
    uphill = float(tobler_hiking_speed_kmh(0.3))
    assert uphill < flat


def test_dijkstra_reachability_from_center():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    elevation = np.zeros((32, 32), dtype=np.float64)
    center = 16
    travel = least_travel_time_hours(elevation, 50.0, center, center, max_hours=2.0)
    assert travel[center, center] == 0.0
    assert np.isfinite(travel[center + 1, center])
    assert travel[center + 1, center] > 0.0


def test_reachability_probability_normalized():
    grid = create_empty_grid(HAIFA, 50.0, 16)
    elevation = np.zeros((16, 16), dtype=np.float64)
    reach = compute_reachability(grid, elevation, 8, 8, max_hours=1.0)
    assert reach.sum() == pytest.approx(1.0)
    assert reach[8, 8] > reach[0, 0]


def test_steep_terrain_reduces_reachability():
    grid = create_empty_grid(HAIFA, 50.0, 16)
    flat = np.zeros((16, 16), dtype=np.float64)
    steep = flat.copy()
    steep[:, 8:] = np.linspace(0, 400, 8)[None, :]

    reach_flat = compute_reachability(grid, flat, 8, 8, max_hours=0.5)
    reach_steep = compute_reachability(grid, steep, 8, 8, max_hours=0.5)

    east_flat = reach_flat[8, 12]
    east_steep = reach_steep[8, 12]
    assert east_steep <= east_flat


def test_reachability_score_linear_at_lkp():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    elevation = np.zeros((32, 32), dtype=np.float64)
    center = 16
    score = compute_reachability_score(grid, elevation, center, center, max_hours=0.1)
    assert score[center, center] == pytest.approx(1.0)
    assert score[center + 5, center] > score[center + 10, center]
    assert score[0, 0] == pytest.approx(0.0)
    assert score.sum() > 1.0  # not normalized to a probability measure


def test_mission_max_hours_grows_every_tick():
    h0 = mission_max_hours(tick_count=0, step_sec=BASE_STEP_SEC)
    h1 = mission_max_hours(tick_count=1, step_sec=BASE_STEP_SEC)
    h2 = mission_max_hours(tick_count=2, step_sec=BASE_STEP_SEC)
    assert h0 == pytest.approx(BASE_STEP_SEC / 3600.0)
    assert h1 == pytest.approx(2 * BASE_STEP_SEC / 3600.0)
    assert h0 < h1 < h2


def test_reachability_score_expands_as_horizon_grows():
    grid = create_empty_grid(HAIFA, 50.0, 128)
    elevation = np.zeros((128, 128), dtype=np.float64)
    center = 64
    s0 = compute_reachability_score(
        grid, elevation, center, center, mission_max_hours(tick_count=0, step_sec=BASE_STEP_SEC)
    )
    s1 = compute_reachability_score(
        grid, elevation, center, center, mission_max_hours(tick_count=5, step_sec=BASE_STEP_SEC)
    )
    assert (s1 > 0.01).sum() > (s0 > 0.01).sum()
