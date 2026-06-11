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


def _water_matrix(size: int = 32, res: float = 50.0, *, u: float | None = None, v: float | None = None) -> GridMatrix:
    """A grid whose cells are ALL water (is_land = False everywhere)."""
    fields = NodeFields.zeros(size)
    fields.is_land = np.zeros((size, size), dtype=bool)
    if u is not None and v is not None:
        fields.current_u[:] = u
        fields.current_v[:] = v
    else:
        from app.services.marine_current import MarineCurrent

        _populate_sea_current(fields, MarineCurrent(
            u_east_mps=settings.sea_drift_speed_mps,
            v_north_mps=0.0,
            speed_mps=settings.sea_drift_speed_mps,
            direction_deg=90.0,
            source="test",
        ))
    matrix = GridMatrix.create(OPEN_SEA, size=size, resolution_m=res, node_fields=fields)
    return matrix


def test_sea_current_populated_on_water_cells(monkeypatch):
    if _populate_sea_current is None:
        pytest.skip("node_builder unavailable")
    from app.services.marine_current import MarineCurrent

    marine = MarineCurrent(
        u_east_mps=0.5,
        v_north_mps=0.0,
        speed_mps=0.5,
        direction_deg=90.0,
        source="test",
    )
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
    _populate_sea_current(fields, marine)
    assert fields.current_u[2, 2] == pytest.approx(0.5)
    assert fields.current_v[2, 2] == pytest.approx(0.0)
    assert fields.current_u[0, 0] == pytest.approx(0.0)
    assert fields.current_v[0, 0] == pytest.approx(0.0)


# --- the layer's transition vector ------------------------------------------


def test_sea_drift_biases_toward_heading(monkeypatch):
    size = 16
    matrix = _water_matrix(size=size, u=0.5, v=0.0)
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
        topography=False, roads=False, personality=False,
        weather=False, sea_drift=True,
    )


def test_engine_drifts_cloud_in_heading_direction(monkeypatch):
    monkeypatch.setattr(settings, "marine_drift_steps", 2)
    size = 41
    matrix = _water_matrix(size=size, u=0.5, v=0.0)
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
    assert probs.sum() == pytest.approx(1.0)


def test_engine_drift_heading_north(monkeypatch):
    monkeypatch.setattr(settings, "marine_drift_steps", 2)
    size = 41
    matrix = _water_matrix(size=size, u=0.0, v=0.5)
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
    assert probs.sum() == pytest.approx(1.0)


def test_sea_mode_keeps_mass_on_water_zeros_land():
    """With sea_drift active, land cells are absorbed (zeroed); on a land
    search the same cells would instead retain the mass."""
    size = 21
    fields = NodeFields.zeros(size)
    # Make the entire east half land, west half water.
    fields.is_land = np.zeros((size, size), dtype=bool)
    fields.is_land[:, (size // 2) + 1 :] = True
    matrix = GridMatrix.create(OPEN_SEA, size=size, resolution_m=50.0, node_fields=fields)

    engine = GridEngine()
    # Heading east pushes mass toward the land half; it must be absorbed.
    out = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())

    land_mass = out[:, (size // 2) + 1 :].sum()
    assert land_mass == 0.0, "no probability may remain on land in sea mode"
    assert out.sum() == pytest.approx(1.0)


def test_mass_conserved_on_open_sea_over_many_ticks():
    size = 41
    matrix = _water_matrix(size=size)
    engine = GridEngine()
    probs = matrix.probabilities
    for _ in range(25):
        matrix.probabilities = probs
        probs = engine.tick(matrix, _sea_flags(), dt_sec=60.0, tick_count=1, env=EnvForcing())
        assert math.isclose(probs.sum(), 1.0, rel_tol=1e-9), "mass must stay normalized"
