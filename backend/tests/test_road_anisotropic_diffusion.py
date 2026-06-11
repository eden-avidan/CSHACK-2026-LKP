"""Tests for continuous cost-surface diffusion."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.engine.cost_surface_diffusion import (
    build_terrain_cost_map,
    cost_surface_diffusion,
    l2_neighbor_distance,
    transition_weight_l2_cost,
)
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.road_magnetism import RoadMagnetismLayer
from app.engine.road_anisotropic_diffusion import pairwise_transition_weight
from app.engine.transition_context import TransitionContext
from app.models.mission import LatLon
from app.services.particle_types import EnvForcing

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def test_l2_neighbor_distance():
    assert l2_neighbor_distance(0, 1) == pytest.approx(1.0)
    assert l2_neighbor_distance(1, 1) == pytest.approx(1.414, rel=1e-3)


def test_terrain_cost_map_values():
    is_road = np.array([[True, False], [False, False]])
    slope = np.zeros((2, 2))
    is_land = np.ones((2, 2), dtype=bool)
    cost = build_terrain_cost_map(is_road, slope, is_land)
    assert cost[0, 0] == settings.cost_road
    assert cost[0, 1] == settings.cost_offroad


def test_transition_weight_inversely_proportional_to_cost_and_l2():
    cost_b = np.array([[settings.cost_offroad]], dtype=np.float64)
    road = np.array([[False]], dtype=bool)
    w_card = transition_weight_l2_cost(1.0, cost_b, road, road)[0, 0]
    w_diag = transition_weight_l2_cost(np.sqrt(2), cost_b, road, road)[0, 0]
    assert w_card > w_diag

    cheap = np.array([[settings.cost_road]], dtype=np.float64)
    road_b = np.array([[True]], dtype=bool)
    w_road = transition_weight_l2_cost(1.0, cheap, road, road_b)[0, 0]
    assert w_road > w_card


def test_trail_magnetism_boosts_offroad_to_road():
    cost_road = np.array([[settings.cost_road]], dtype=np.float64)
    cost_off = np.array([[settings.cost_offroad]], dtype=np.float64)
    off = np.array([[False]], dtype=bool)
    on = np.array([[True]], dtype=bool)
    to_road = transition_weight_l2_cost(1.0, cost_road, off, on)[0, 0]
    to_off = transition_weight_l2_cost(1.0, cost_off, off, off)[0, 0]
    assert to_road > to_off


def test_cost_surface_diffusion_road_fingers_with_forest_bleed():
    size = 16
    row = 8
    is_road = np.zeros((size, size), dtype=bool)
    is_road[row, :] = True
    slope = np.zeros((size, size))
    is_land = np.ones((size, size), dtype=bool)
    cost = build_terrain_cost_map(is_road, slope, is_land)

    p = np.zeros((size, size), dtype=np.float64)
    p[row, 8] = 1.0

    out = cost_surface_diffusion(p, cost, is_road, steps=8)
    along = out[row, 12]
    off = out[row - 3, 12]
    forest_near = out[row - 1, 12]
    assert along > off
    assert forest_near > 0.0
    assert along > forest_near


def test_steep_slope_increases_terrain_cost():
    size = 8
    is_road = np.zeros((size, size), dtype=bool)
    slope = np.zeros((size, size))
    slope[3:6, 3:6] = np.radians(40.0)
    is_land = np.ones((size, size), dtype=bool)
    cost = build_terrain_cost_map(is_road, slope, is_land)
    assert cost[4, 4] >= settings.cost_steep_slope
    assert cost[4, 4] > settings.cost_offroad


def test_road_layer_cost_surface_integration():
    size = 16
    fields = NodeFields.zeros(size)
    row = 8
    fields.is_road[row, :] = True
    fields.road_proximity[row, :] = 1.0
    fields.is_land.fill(True)

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    matrix.probabilities.fill(0.0)
    matrix.probabilities[row, row] = 1.0

    ctx = TransitionContext(
        matrix=matrix,
        node_fields=fields,
        probabilities=matrix.probabilities,
        dt_sec=60.0,
        tick_count=0,
        env=EnvForcing(),
        size=size,
        resolution_m=50.0,
    )
    out = RoadMagnetismLayer().apply_field(ctx, weight=1.0)
    assert out[row, row + 4] > out[row - 3, row + 4]
    assert out[row - 1, row + 2] > 0.0


def test_pairwise_transition_weight_legacy_ordering():
    assert pairwise_transition_weight(True, True) > pairwise_transition_weight(True, False)
    assert pairwise_transition_weight(False, True) > pairwise_transition_weight(False, False)
