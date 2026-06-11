"""Grid Matrix engine tests."""

from __future__ import annotations

from app.core.config import settings
from app.engine.grid_engine import GridEngine
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.road_magnetism import RoadMagnetismLayer
from app.engine.layers.registry import get_active_layers
from app.engine.neighbors import NEIGHBOR_OFFSETS
from app.engine.transition_context import TransitionContext
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


def test_grid_engine_conserves_probability_mass():
    matrix = GridMatrix.create(HAIFA, size=32, resolution_m=50.0)
    engine = GridEngine()
    layers = LayerFlags(topography=False, roads=False, weather=False, subject_injured=False)
    out = engine.tick(matrix, layers, dt_sec=60.0, tick_count=1, env=EnvForcing())
    assert out.sum() == 1.0
    assert out.max() > 0.0


def test_road_magnetism_biases_along_tangent():
    size = 16
    fields = NodeFields.zeros(size)
    row, col = 8, 8
    fields.road_proximity[row, col] = 1.0
    fields.road_tangent_e[row, col] = 1.0
    fields.road_tangent_n[row, col] = 0.0

    matrix = GridMatrix.create(HAIFA, size=size, resolution_m=50.0, node_fields=fields)
    transition_ctx = TransitionContext(
        matrix=matrix,
        node_fields=fields,
        probabilities=matrix.probabilities,
        dt_sec=60.0,
        tick_count=0,
        env=EnvForcing(),
        size=size,
        resolution_m=50.0,
    )
    layer = RoadMagnetismLayer()
    weights = layer.transition_weights(transition_ctx, row, col, weight=1.0)
    east_idx = NEIGHBOR_OFFSETS.index((0, 1))
    west_idx = NEIGHBOR_OFFSETS.index((0, -1))
    assert weights[east_idx] > weights[west_idx]


def test_layer_registry_loads_topography_by_default():
    layers = get_active_layers(LayerFlags())
    ids = {layer.layer_id for layer, _ in layers}
    assert "topography" in ids


def test_grid_matrix_dimensions_match_config():
    matrix = GridMatrix.create(HAIFA, settings.grid_size, settings.grid_resolution_m)
    span = settings.grid_size * settings.grid_resolution_m
    assert matrix.size == settings.grid_size
    assert matrix.total_area_m == span * span
