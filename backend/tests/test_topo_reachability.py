"""Tests for Tobler/Dijkstra reachability (topo_layout parity)."""

from __future__ import annotations

import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid
from app.models.mission import LatLon
from app.services.topo_reachability import (
    compute_reachability,
    least_travel_time_hours,
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
