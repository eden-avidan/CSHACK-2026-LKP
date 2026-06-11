"""Tests for live/offline mission modes and layer validation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.models.layers import LayerFlags
from app.models.mission import BASE_STEP_SEC, LIVE_UPDATE_INTERVAL_SEC, LatLon, MissionMode
from app.services.env_ingestion import TerrainContext
from app.services.mission_store import MissionStore

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def _terrain(size: int) -> TerrainContext:
    return TerrainContext(
        elevation=np.zeros((size, size)),
        slope=np.zeros((size, size)),
        aspect_n=np.zeros((size, size)),
        aspect_e=np.zeros((size, size)),
        road_proximity=np.zeros((size, size)),
        is_land=np.ones((size, size), dtype=bool),
        road_tangent_e=np.zeros((size, size)),
        road_tangent_n=np.zeros((size, size)),
        reachability=None,
    )


def test_live_mode_pace_derived_timing():
    async def run() -> None:
        store = MissionStore()
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(HAIFA, mode=MissionMode.LIVE, pace=2.0)
            assert state.mode == MissionMode.LIVE
            assert state.simulation_running is True
            assert state.step_sec == pytest.approx(BASE_STEP_SEC * 2.0)
            assert state.update_interval_sec == LIVE_UPDATE_INTERVAL_SEC
            assert state.pace == 2.0

    asyncio.run(run())


def test_offline_mode_batch_ticks():
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(
                HAIFA,
                mode=MissionMode.OFFLINE,
                lkp_timestamp=two_hours_ago,
            )
            assert state.mode == MissionMode.OFFLINE
            assert state.simulation_running is True
            assert state.tick_count >= 1

    asyncio.run(run())


def test_offline_mode_uses_pace_for_live_ticks_after_seed():
    async def run() -> None:
        store = MissionStore()
        two_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=2)
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(
                HAIFA,
                mode=MissionMode.OFFLINE,
                lkp_timestamp=two_minutes_ago,
                pace=3.0,
            )
            assert state.pace == pytest.approx(3.0)
            assert state.step_sec == pytest.approx(BASE_STEP_SEC * 3.0)

            await store.update_pace(state.mission_id, pace=0.5)
            assert state.pace == pytest.approx(0.5)
            assert state.step_sec == pytest.approx(BASE_STEP_SEC * 0.5)

    asyncio.run(run())


def test_offline_mode_continues_after_seed_batch():
    async def run() -> None:
        store = MissionStore()
        two_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=2)
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(
                HAIFA,
                mode=MissionMode.OFFLINE,
                lkp_timestamp=two_minutes_ago,
            )
            seeded_tick_count = state.tick_count
            await store.tick(state.mission_id)
            assert state.tick_count == seeded_tick_count + 1

    asyncio.run(run())


def test_update_layers_forces_topography_when_all_off():
    async def run() -> None:
        store = MissionStore()
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(HAIFA, layers={"topography": False, "roads": False})
            await store.update_layers(
                state.mission_id,
                {
                    "topography": False,
                    "roads": False,
                    "personality": False,
                    "weather": False,
                },
            )
            assert state.layers.topography is True

    asyncio.run(run())


def test_tick_preserves_probability_mass():
    async def run() -> None:
        store = MissionStore()
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(HAIFA)
            origin = state.grid.metadata.origin
            before = float(state.grid.probabilities.sum())
            assert before > 0.0
            await store.tick(state.mission_id)
            after = float(state.grid.probabilities.sum())
            assert after > 0.0
            assert float(state.grid.probabilities.max()) > 1e-6
            assert state.grid.metadata.origin.lat == pytest.approx(origin.lat, abs=1e-6)
            assert state.grid.metadata.origin.lon == pytest.approx(origin.lon, abs=1e-6)

    asyncio.run(run())


def test_tick_accumulates_history_near_lkp():
    async def run() -> None:
        store = MissionStore()
        with patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock:
            mock.return_value = _terrain(128)
            state = await store.create(HAIFA, layers={"topography": False, "roads": False, "weather": False})
            lkp_row = state.grid.rows // 2
            lkp_col = state.grid.cols // 2
            initial_lkp_mass = float(state.grid.probabilities[lkp_row, lkp_col])
            for _ in range(8):
                await store.tick(state.mission_id)
            lkp_mass_after = float(state.grid.probabilities[lkp_row, lkp_col])
            assert lkp_mass_after > 0
            assert lkp_mass_after < initial_lkp_mass * 2.5

    asyncio.run(run())


def test_layer_flags_apply_update_min_one():
    flags = LayerFlags(topography=False, roads=False, personality=False, weather=False)
    flags.apply_update({"weather": False})
    assert flags.topography is True
