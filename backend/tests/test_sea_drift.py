"""Tests for the sea-drift layer and the engine's sea-mode land masking.

These exercise the engine layer directly (app.engine.*), avoiding the legacy
app.layers.* import chain that has a pre-existing Python 3.11 dataclass issue.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.config import settings
from app.engine.grid_engine import GridEngine
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.sea_drift import SeaDriftLayer
from app.engine.neighbors import NEIGHBOR_OFFSETS
from app.engine.transition_context import TransitionContext
from app.models.layers import LayerFlags
from app.models.mission import LatLon
from app.services.particle_types import EnvForcing

try:
    from app.engine.node_builder import _populate_sea_current
except ImportError:
    _populate_sea_current = None  # type: ignore[misc, assignment]

# Open-water LKP (Mediterranean, west of Haifa) — far enough offshore that the
# whole default grid is sea.
OPEN_SEA = LatLon(lat=32.80, lon=34.50)


def _centroid(probs: np.ndarray) -> tuple[float, float]:
    total = probs.sum()
    rows = np.arange(probs.shape[0])[:, None]
    cols = np.arange(probs.shape[1])[None, :]
    return float((probs * rows).sum() / total), float((probs * cols).sum() / total)


def _water_matrix(size: int = 32, res: float = 50.0) -> GridMatrix:
    """A grid whose cells are ALL water (is_land = False everywhere)."""
    fields = NodeFields.zeros(size)
    fields.is_land = np.zeros((size, size), dtype=bool)
    matrix = GridMatrix.create(OPEN_SEA, size=size, resolution_m=res, node_fields=fields)
    return matrix


def test_sea_current_populated_on_water_cells(monkeypatch):
    if _populate_sea_current is None:
        pytest.skip("node_builder unavailable")
    monkeypatch.setattr(settings, "sea_drift_speed_mps", 0.5)
    monkeypatch.setattr(settings, "sea_drift_heading_deg", 90.0)  # due east
    fields = NodeFields.zeros(4)
    fields.is_land = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    _populate_sea_current(fields)
    assert fields.current_u[2, 2] == pytest.approx(0.5)
    assert fields.current_v[2, 2] == pytest.approx(0.0)
    assert fields.current_u[0, 0] == pytest.approx(0.0)
    assert fields.current_v[0, 0] == pytest.approx(0.0)


# --- the layer's transition vector ------------------------------------------


def test_sea_drift_biases_toward_heading(monkeypatch):
    monkeypatch.setattr(settings, "sea_drift_heading_deg", 90.0)  # due east
    size = 16
    matrix = _water_matrix(size=size)
    ctx = TransitionContext(
        matrix=matrix,
        node_fields=matrix.node_fields,
        probabilities=matrix.probabilities,
        dt_sec=60.0,
        tick_count=0,
        env=EnvForcing(),
        size=size,
        resolution_m=matrix.resolution_m,
    )
    weights = SeaDriftLayer().transition_weights(ctx, 8, 8, weight=1.0)

    east_idx = NEIGHBOR_OFFSETS.index((0, 1))
    west_idx = NEIGHBOR_OFFSETS.index((0, -1))
    assert weights[east_idx] > 0
    assert weights[west_idx] == 0.0  # against-drift neighbors get no bias
    assert weights[east_idx] > weights[NEIGHBOR_OFFSETS.index((-1, 0))]  # > pure north


def test_sea_drift_zero_when_weight_zero():
    size = 16
    matrix = _water_matrix(size=size)
    ctx = TransitionContext(
        matrix=matrix,
        node_fields=matrix.node_fields,
        probabilities=matrix.probabilities,
        dt_sec=60.0,
        tick_count=0,
        env=EnvForcing(),
        size=size,
        resolution_m=matrix.resolution_m,
    )
    weights = SeaDriftLayer().transition_weights(ctx, 8, 8, weight=0.0)
    assert np.all(weights == 0.0)


# --- end-to-end through the engine ------------------------------------------


def _sea_flags() -> LayerFlags:
    return LayerFlags(
        topography=False, roads=False, subject_injured=False,
        weather=False, sea_drift=True,
    )


def test_engine_drifts_cloud_in_heading_direction(monkeypatch):
    monkeypatch.setattr(settings, "sea_drift_heading_deg", 90.0)  # east
    size = 41
    matrix = _water_matrix(size=size)
    engine = GridEngine()

    before = _centroid(matrix.probabilities)
    probs = matrix.probabilities
    for _ in range(8):
        matrix.probabilities = probs
        probs = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())
    after = _centroid(probs)

    # East = columns increasing, rows roughly unchanged.
    assert after[1] > before[1] + 1.0, "cloud should drift east (cols increase)"
    assert abs(after[0] - before[0]) < 1.0, "no significant north/south drift for due-east heading"
    assert probs.sum() == 1.0


def test_engine_drift_heading_north(monkeypatch):
    monkeypatch.setattr(settings, "sea_drift_heading_deg", 0.0)  # north
    size = 41
    matrix = _water_matrix(size=size)
    engine = GridEngine()

    before = _centroid(matrix.probabilities)
    probs = matrix.probabilities
    for _ in range(8):
        matrix.probabilities = probs
        probs = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())
    after = _centroid(probs)

    # North = rows decreasing.
    assert after[0] < before[0] - 1.0, "cloud should drift north (rows decrease)"
    assert abs(after[1] - before[1]) < 1.0
    assert probs.sum() == 1.0


def test_sea_mode_keeps_mass_on_water_zeros_land():
    """With sea_drift active, land cells are absorbed (zeroed); on a land
    search the same cells would instead retain the mass."""
    size = 21
    fields = NodeFields.zeros(size)
    # Make the entire east half land, west half water.
    fields.is_land = np.zeros((size, size), dtype=bool)
    fields.is_land[:, size // 2:] = True
    matrix = GridMatrix.create(OPEN_SEA, size=size, resolution_m=50.0, node_fields=fields)

    engine = GridEngine()
    # Heading east pushes mass toward the land half; it must be absorbed.
    out = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())

    land_mass = out[:, size // 2:].sum()
    assert land_mass == 0.0, "no probability may remain on land in sea mode"
    assert out.sum() == 1.0


def test_mass_conserved_on_open_sea_over_many_ticks():
    size = 41
    matrix = _water_matrix(size=size)
    engine = GridEngine()
    probs = matrix.probabilities
    for _ in range(25):
        matrix.probabilities = probs
        probs = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())
        assert math.isclose(probs.sum(), 1.0, rel_tol=1e-9), "mass must stay normalized"
