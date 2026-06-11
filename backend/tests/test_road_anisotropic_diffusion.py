"""Tests for cost-weighted trail diffusion."""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.road_magnetism import RoadMagnetismLayer
from app.engine.road_anisotropic_diffusion import (
    anisotropic_road_diffusion,
    pairwise_transition_weight,
)
from app.engine.transition_context import TransitionContext
from app.models.mission import LatLon
from app.services.particle_types import EnvForcing

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def test_pairwise_transition_weights():
    assert pairwise_transition_weight(True, True) == settings.road_to_road_transition
    assert pairwise_transition_weight(True, False) == settings.road_to_offroad_transition
    assert pairwise_transition_weight(False, True) == settings.offroad_to_road_transition
    assert pairwise_transition_weight(False, False) == settings.offroad_to_offroad_transition
    assert settings.road_to_road_transition > settings.offroad_to_road_transition
    assert settings.offroad_to_road_transition > settings.offroad_to_offroad_transition
    assert settings.road_to_offroad_transition < settings.offroad_to_offroad_transition


def test_anisotropic_diffusion_prefers_trail_over_forest():
    size = 16
    is_road = np.zeros((size, size), dtype=bool)
    row = 8
    is_road[row, :] = True

    p = np.zeros((size, size), dtype=np.float64)
    p[row, 2] = 1.0

    out = anisotropic_road_diffusion(p, is_road, steps=6)
    along_trail = out[row, 6]
    off_trail = out[row - 3, 6]
    assert along_trail > off_trail


def test_road_layer_winds_along_horizontal_trail():
    size = 16
    fields = NodeFields.zeros(size)
    row = 8
    fields.is_road[row, :] = True
    fields.road_proximity[row, :] = 1.0
    fields.is_land.fill(True)

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    matrix.probabilities.fill(0.0)
    matrix.probabilities[row, 2] = 1.0

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
    assert out[row, 7] > out[row - 3, 7]
