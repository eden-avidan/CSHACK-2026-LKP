"""Tests for toggleable layers, MPP, and land mask."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid
from app.models.layers import LayerFlags
from app.models.mission import LatLon
from app.services.env_ingestion import TerrainContext
from app.services.mission_store import MissionStore
from app.services.particle_filter import (
    Particles,
    compute_mpp,
    get_mock_env,
    initialize_particles,
    predict_step,
    zero_env,
)

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def _terrain(rows: int, cols: int, *, land: bool = True) -> TerrainContext:
    is_land = np.full((rows, cols), land, dtype=bool)
    road_prox = np.zeros((rows, cols))
    road_prox[rows // 2, cols // 2] = 1.0
    return TerrainContext(
        elevation=np.zeros((rows, cols)),
        slope=np.zeros((rows, cols)),
        aspect_n=np.zeros((rows, cols)),
        aspect_e=np.ones((rows, cols)),
        road_proximity=road_prox,
        is_land=is_land,
        road_tangent_e=np.ones((rows, cols)),
        road_tangent_n=np.zeros((rows, cols)),
    )


def test_weather_off_zero_wind():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    n = 1000
    particles = Particles(
        eastings=np.full(n, grid.crs.origin_e),
        northings=np.full(n, grid.crs.origin_n),
        v_n=np.zeros(n),
        v_e=np.zeros(n),
        weights=np.full(n, 1.0 / n),
    )
    calm = predict_step(
        particles, zero_env(), dt=120.0, grid=grid, layers=LayerFlags(weather=False)
    )
    windy = predict_step(
        particles, get_mock_env(), dt=120.0, grid=grid, layers=LayerFlags(weather=True)
    )
    assert np.mean(windy.v_n) > np.mean(calm.v_n) + 0.3
    assert np.mean(windy.v_e) > np.mean(calm.v_e) + 0.3


def test_topography_land_mask():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    terrain = _terrain(grid.rows, grid.cols, land=False)
    # Mark center cell as land so particle can be nudged there
    terrain.is_land[16, 16] = True
    particles = Particles(
        eastings=np.array([grid.crs.origin_e]),
        northings=np.array([grid.crs.origin_n]),
        v_n=np.array([5.0]),
        v_e=np.array([0.0]),
        weights=np.array([1.0]),
    )
    layers = LayerFlags(topography=True, weather=False)
    out = predict_step(particles, zero_env(), dt=1.0, terrain=terrain, grid=grid, layers=layers)
    rows, cols = 16, 16
    res = grid.metadata.resolution_m
    half = (grid.rows * res) / 2.0
    target_e = grid.crs.origin_e - half + (cols + 0.5) * res
    target_n = grid.crs.origin_n + half - (rows + 0.5) * res
    # Particle should move toward land cell from water
    assert abs(out.eastings[0] - target_e) < abs(particles.eastings[0] - target_e) or out.v_e[0] < particles.v_e[0]


def test_road_snap_within_50m():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    terrain = _terrain(grid.rows, grid.cols)
    terrain.road_proximity[:, :] = 1.0
    terrain.road_tangent_e[:, :] = 1.0
    terrain.road_tangent_n[:, :] = 0.0
    particles = Particles(
        eastings=np.full(500, grid.crs.origin_e),
        northings=np.full(500, grid.crs.origin_n),
        v_n=np.zeros(500),
        v_e=np.full(500, -2.0),
        weights=np.full(500, 1.0 / 500),
    )
    layers = LayerFlags(roads=True, weather=False, topography=False)
    out = predict_step(particles, zero_env(), dt=1.0, terrain=terrain, grid=grid, layers=layers)
    # Road snap should pull east velocity toward positive tangent (east)
    assert np.mean(out.v_e) > np.mean(particles.v_e)


def test_personality_reduces_variance():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 200, 50.0)
    base = predict_step(
        particles, zero_env(), dt=5.0, grid=grid, layers=LayerFlags(personality=False)
    )
    injured = predict_step(
        particles, zero_env(), dt=5.0, grid=grid, layers=LayerFlags(personality=True)
    )
    base_spread = np.std(base.eastings - particles.eastings)
    injured_spread = np.std(injured.eastings - particles.eastings)
    assert injured_spread < base_spread


def test_compute_mpp_centroid():
    grid = create_empty_grid(HAIFA, 50.0, 16)
    probs = np.zeros((16, 16))
    probs[8, 8] = 0.9
    probs[8, 9] = 0.1
    mpp = compute_mpp(grid, probs)
    center_lat, center_lon = grid.metadata.origin.lat, grid.metadata.origin.lon
    assert abs(mpp.lat - center_lat) < 0.05
    assert abs(mpp.lon - center_lon) < 0.05


def test_update_layers():
    async def run() -> None:
        store = MissionStore()
        size = 128
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(size, size)
            state = await store.create(HAIFA)
            assert state.layers.weather is False
            await store.update_layers(state.mission_id, {"weather": True, "roads": True})
            assert state.layers.weather is True
            assert state.layers.roads is True

    asyncio.run(run())


def test_tick_emits_engine_tick():
    async def run() -> None:
        store = MissionStore()
        size = 128
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(size, size)
            state = await store.create(HAIFA)
            mpp_before = state.mpp
            assert mpp_before is not None
            result = await store.tick(state.mission_id)
            assert result.engine_tick is not None
            assert result.engine_tick.tick_count == 1
            assert result.engine_tick.lkp_coords.lat == HAIFA.lat
            assert len(result.engine_tick.particle_matrix) > 0
            assert result.full_refresh is True
            assert abs(state.grid.metadata.origin.lat - HAIFA.lat) < 1e-5
            assert abs(state.grid.metadata.origin.lon - HAIFA.lon) < 1e-5

    asyncio.run(run())


def test_create_applies_layers():
    async def run() -> None:
        store = MissionStore()
        size = 128
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(size, size)
            state = await store.create(HAIFA, layers={"weather": True, "roads": True})
            assert state.layers.weather is True
            assert state.layers.roads is True

    asyncio.run(run())


def test_momentum_preserves_direction_over_steps():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 800, 30.0)
    particles = Particles(
        eastings=particles.eastings,
        northings=particles.northings,
        v_n=np.full(800, 0.8),
        v_e=np.full(800, 0.3),
        weights=particles.weights,
    )
    layers = LayerFlags(topography=False, weather=False, roads=False)
    dt = 60.0

    headings: list[float] = []
    for _ in range(8):
        prev_e, prev_n = particles.eastings.copy(), particles.northings.copy()
        particles = predict_step(particles, zero_env(), dt=dt, grid=grid, layers=layers)
        de = float(np.mean(particles.eastings - prev_e))
        dn = float(np.mean(particles.northings - prev_n))
        headings.append(float(np.arctan2(de, dn)))

    spread = float(np.std(headings))
    assert spread < 0.35


def test_radial_fade_softens_corners_not_center():
    from app.services.particle_types import apply_radial_fade

    uniform = np.ones((64, 64), dtype=np.float64) / (64 * 64)
    faded = apply_radial_fade(uniform, fade_end=0.88)
    assert faded.sum() == pytest.approx(1.0)
    assert faded[32, 32] > faded[0, 0]
    assert faded[0, 32] < faded[32, 32]
