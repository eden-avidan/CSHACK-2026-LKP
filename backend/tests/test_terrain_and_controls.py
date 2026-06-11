"""Tests for terrain physics, road bias, pause behavior, and grid boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid, grid_utm_bounds
from app.models.layers import LayerFlags
from app.models.mission import LatLon
from app.services.env_ingestion import TerrainContext, build_terrain_context
from app.services.mission_store import MissionStore
from app.services.topo_reachability import classify_terrain_from_elevation
from app.services.particle_filter import (
    apply_grid_bounds,
    get_mock_env,
    initialize_particles,
    predict_step,
    rasterize_kde,
)

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def test_terrain_classification_steep_and_valley():
    rows, cols = 32, 32
    elevation = np.zeros((rows, cols), dtype=np.float64)
    elevation[8:24, 8:24] = np.linspace(0, 200, 16 * 16).reshape(16, 16)
    terrain = classify_terrain_from_elevation(elevation, resolution_m=50.0)
    assert terrain.steep_mask.shape == (rows, cols)
    assert terrain.valley_mask.shape == (rows, cols)


def test_road_kde_bias():
    grid = create_empty_grid(HAIFA, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 200, 100.0)
    road = np.zeros((grid.rows, grid.cols))
    road[16, 16] = 1.0
    probs_flat = rasterize_kde(
        particles,
        grid.rows,
        grid.cols,
        grid.metadata.resolution_m,
        grid.crs.origin_e,
        grid.crs.origin_n,
    )
    probs_road = rasterize_kde(
        particles,
        grid.rows,
        grid.cols,
        grid.metadata.resolution_m,
        grid.crs.origin_e,
        grid.crs.origin_n,
        road_proximity=road,
        roads_layer=True,
    )
    assert np.isclose(probs_road.sum(), 1.0)
    assert probs_road[16, 16] > probs_flat[16, 16]


def test_pause_skips_tick():
    async def run() -> None:
        store = MissionStore()
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_terrain:
            size = 128
            mock_terrain.return_value = TerrainContext(
                elevation=np.zeros((size, size)),
                slope=np.zeros((size, size)),
                aspect_n=np.zeros((size, size)),
                aspect_e=np.zeros((size, size)),
                road_proximity=np.zeros((size, size)),
                is_land=np.ones((size, size), dtype=bool),
                road_tangent_e=np.zeros((size, size)),
                road_tangent_n=np.zeros((size, size)),
            )
            state = await store.create(HAIFA)
            assert state.tick_count == 0
            await store.pause(state.mission_id)
            result = await store.tick(state.mission_id)
            assert result.deltas == []
            assert state.tick_count == 0

    asyncio.run(run())


def test_particles_stay_in_bounds():
    grid = create_empty_grid(HAIFA, 50.0, 128)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 500, 50.0)
    env = get_mock_env()
    layers = LayerFlags(topography=False, weather=True)

    for _ in range(100):
        particles = predict_step(particles, env, dt=5.0, grid=grid, layers=layers)

    road_prox = np.zeros((grid.rows, grid.cols))
    probs = rasterize_kde(
        particles,
        grid.rows,
        grid.cols,
        grid.metadata.resolution_m,
        grid.crs.origin_e,
        grid.crs.origin_n,
        road_proximity=road_prox,
    )
    edge_mass = probs[0, :].sum() + probs[-1, :].sum() + probs[:, 0].sum() + probs[:, -1].sum()
    assert edge_mass < 0.08


def test_build_terrain_context_fallback():
    async def run() -> None:
        grid = create_empty_grid(HAIFA, 50.0, 16)
        with patch("app.services.env_ingestion.fetch_elevations", new_callable=AsyncMock) as mock_elev:
            mock_elev.return_value = None
            with patch("app.services.env_ingestion.fetch_osm_roads", new_callable=AsyncMock) as mock_roads:
                mock_roads.return_value = []
                ctx = await build_terrain_context(grid)
                assert ctx.slope.shape == (16, 16)
                assert np.allclose(ctx.road_proximity, 0.0)

    asyncio.run(run())
