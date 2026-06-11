"""Grid Matrix engine tests — interactive field layer pipeline."""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.grid_engine import GridEngine
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.registry import get_active_layers
from app.engine.layers.road_magnetism import RoadMagnetismLayer
from app.engine.layers.topography import TopographyLayer
from app.models.layers import LayerFlags
from app.models.mission import LatLon
from app.services.particle_types import EnvForcing

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def test_grid_matrix_t0_initialization():
    matrix = GridMatrix.create(HAIFA, size=32, resolution_m=50.0)
    assert matrix.lkp_row == 16
    assert matrix.lkp_col == 16
    assert matrix.probabilities.sum() == 1.0
    assert matrix.probabilities[16, 16] == 1.0
    assert matrix.probabilities[0, 0] == 0.0


def test_topography_spreads_impulse_to_reachability_field():
    size = 32
    fields = NodeFields.zeros(size)
    row, col = size // 2, size // 2
    fields.reachability_score[row, col] = 1.0
    fields.reachability_score[row, col + 5] = 0.6
    fields.reachability_score[row, col + 15] = 0.0
    fields.is_land.fill(True)

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    engine = GridEngine()
    out = engine.apply_layers(matrix, LayerFlags(topography=True, roads=False), env=EnvForcing())

    assert out[row, col] == 1.0
    assert out[row, col + 5] == 0.6
    assert out[row, col + 15] == 0.0
    assert out.sum() > 1.0  # not forced to sum to 1


def test_topography_zeros_water():
    size = 16
    fields = NodeFields.zeros(size)
    fields.reachability_score[:, :] = 0.5
    fields.is_land[8, 8] = False

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    out = GridEngine().apply_layers(matrix, LayerFlags(topography=True), env=EnvForcing())
    assert out[8, 8] == 0.0


def test_roads_boosts_near_road_cells():
    size = 16
    fields = NodeFields.zeros(size)
    fields.road_proximity[8, 8] = 1.0
    fields.road_proximity[8, 9] = 0.0
    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    matrix.probabilities.fill(0.0)
    matrix.probabilities[8, 8] = 1.0
    matrix.probabilities[8, 9] = 1.0

    out = GridEngine().apply_layers(
        matrix,
        LayerFlags(topography=False, roads=True),
        env=EnvForcing(),
    )
    boost = 1.0 + settings.road_kde_bonus
    assert out[8, 8] == 1.0 * boost
    assert out[8, 9] == 1.0


def test_layers_chain_topography_then_roads():
    size = 16
    fields = NodeFields.zeros(size)
    fields.reachability_score[8, 8] = 1.0
    fields.reachability_score[8, 9] = 0.5
    fields.road_proximity[8, 9] = 1.0
    fields.is_land.fill(True)

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    out = GridEngine().apply_layers(
        matrix,
        LayerFlags(topography=True, roads=True),
        env=EnvForcing(),
    )
    boost = 1.0 + settings.road_kde_bonus
    assert out[8, 8] == 1.0
    assert out[8, 9] == 0.5 * boost


def test_layer_registry_loads_topography_by_default():
    layers = get_active_layers(LayerFlags())
    ids = {layer.layer_id for layer, _ in layers}
    assert "topography" in ids


def test_grid_matrix_dimensions_match_config():
    matrix = GridMatrix.create(HAIFA, settings.grid_size, settings.grid_resolution_m)
    span = settings.grid_size * settings.grid_resolution_m
    assert matrix.size == settings.grid_size
    assert matrix.total_area_m == span * span


def test_road_layer_apply_field_direct():
    size = 16
    fields = NodeFields.zeros(size)
    fields.road_proximity[8, 8] = 0.8
    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    from app.engine.transition_context import TransitionContext

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
    assert out[8, 8] == 1.0 + settings.road_kde_bonus * 0.8


def test_topography_layer_apply_field_direct():
    size = 16
    fields = NodeFields.zeros(size)
    fields.reachability_score[8, 8] = 1.0
    fields.reachability_score[8, 10] = 0.4
    fields.is_land.fill(True)
    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    from app.engine.transition_context import TransitionContext

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
    out = TopographyLayer().apply_field(ctx, weight=1.0)
    assert out[8, 8] == 1.0
    assert out[8, 10] == 0.4
