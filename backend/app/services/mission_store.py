from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, create_empty_grid, extract_field_for_grid
from app.models.heatmap import HeatmapCellDelta
from app.models.layers import EngineTickMessage, LayerFlags
from app.models.mission import LatLon, MissionStatus
from app.services.env_ingestion import TerrainContext, build_terrain_context
from app.services.particle_filter import (
    Particles,
    compute_mpp,
    downsample_particles,
    get_mock_env,
    initialize_particles,
    predict_step,
    rasterize_kde,
    resample_if_needed,
    zero_env,
)
from app.services.negative_search import apply_negative_search


@dataclass
class TickResult:
    deltas: list[HeatmapCellDelta]
    engine_tick: Optional[EngineTickMessage]
    full_refresh: bool = False


@dataclass
class MissionState:
    mission_id: UUID
    lkp: LatLon
    status: MissionStatus
    created_at: datetime
    particles: Particles
    grid: ProbabilityGrid
    terrain_grid: ProbabilityGrid
    tick_count: int = 0
    step_sec: float = 1.0
    update_interval_sec: float = 1.0
    simulation_running: bool = True
    terrain: Optional[TerrainContext] = None
    layers: LayerFlags = field(default_factory=LayerFlags)
    mpp: Optional[LatLon] = None
    tick_task: Optional[asyncio.Task] = field(default=None, repr=False)
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class MissionStore:
    def __init__(self) -> None:
        self._missions: dict[UUID, MissionState] = {}

    def get(self, mission_id: UUID) -> MissionState | None:
        return self._missions.get(mission_id)

    async def create(
        self,
        lkp: LatLon,
        sigma_0_m: Optional[float] = None,
        step_sec: float = 1.0,
        update_interval_sec: float = 1.0,
        layers: Optional[dict[str, bool]] = None,
    ) -> MissionState:
        mission_id = uuid4()
        sigma = sigma_0_m or settings.sigma_0_m
        terrain_grid = create_empty_grid(lkp, settings.grid_resolution_m, settings.grid_size)
        terrain = await build_terrain_context(terrain_grid)
        grid = create_empty_grid(lkp, settings.grid_resolution_m, settings.grid_size)
        layer_flags = LayerFlags()
        if layers:
            layer_flags.apply_update(layers)
        particles = initialize_particles(
            grid.crs.origin_e,
            grid.crs.origin_n,
            settings.particle_count,
            sigma,
        )
        road_prox = terrain.road_proximity if terrain else None
        grid.probabilities = rasterize_kde(
            particles,
            grid.rows,
            grid.cols,
            grid.metadata.resolution_m,
            grid.crs.origin_e,
            grid.crs.origin_n,
            road_proximity=road_prox,
            roads_layer=layer_flags.roads,
        )
        mpp = compute_mpp(grid, grid.probabilities)

        state = MissionState(
            mission_id=mission_id,
            lkp=lkp,
            status=MissionStatus.SEARCHING,
            created_at=datetime.now(timezone.utc),
            particles=particles,
            grid=grid,
            terrain_grid=terrain_grid,
            step_sec=step_sec,
            update_interval_sec=update_interval_sec,
            terrain=terrain,
            layers=layer_flags,
            mpp=mpp,
        )
        self._missions[mission_id] = state
        return state

    async def tick(self, mission_id: UUID) -> TickResult:
        state = self._require(mission_id)
        if not state.simulation_running:
            return TickResult(deltas=[], engine_tick=None)
        async with state._lock:
            step_center = state.mpp if state.mpp else state.lkp
            state.grid = create_empty_grid(
                step_center, settings.grid_resolution_m, settings.grid_size
            )

            env = get_mock_env() if state.layers.weather else zero_env()
            state.particles = predict_step(
                state.particles,
                env,
                dt=state.step_sec,
                terrain=state.terrain,
                grid=state.grid,
                terrain_grid=state.terrain_grid,
                layers=state.layers,
            )
            state.particles = resample_if_needed(state.particles)
            road_prox = None
            if state.terrain is not None:
                road_prox = extract_field_for_grid(
                    state.grid, state.terrain_grid, state.terrain.road_proximity
                )
            state.grid.probabilities = rasterize_kde(
                state.particles,
                state.grid.rows,
                state.grid.cols,
                state.grid.metadata.resolution_m,
                state.grid.crs.origin_e,
                state.grid.crs.origin_n,
                road_proximity=road_prox,
                roads_layer=state.layers.roads,
            )
            state.tick_count += 1
            state.mpp = compute_mpp(state.grid, state.grid.probabilities)
            engine_tick = EngineTickMessage(
                event="engine_tick",
                tick_count=state.tick_count,
                lkp_coords=state.lkp,
                mpp_coords=state.mpp,
                layers=state.layers.as_dict(),
                particle_matrix=downsample_particles(state.particles, state.grid),
            )
            return TickResult(deltas=[], engine_tick=engine_tick, full_refresh=True)

    async def update_layers(self, mission_id: UUID, layers: dict[str, bool]) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            state.layers.apply_update(layers)
        return state

    async def pause(self, mission_id: UUID) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            state.simulation_running = False
        return state

    async def resume(self, mission_id: UUID) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            state.simulation_running = True
        return state

    async def delete(self, mission_id: UUID) -> None:
        state = self.get(mission_id)
        if not state:
            raise KeyError(f"Mission {mission_id} not found")
        async with state._lock:
            state.simulation_running = False
            for q in list(state.subscribers):
                await q.put({"type": "mission_closed"})
            state.subscribers.clear()
        del self._missions[mission_id]

    async def negative_search(
        self, mission_id: UUID, polygon: dict, pod: float
    ) -> list[HeatmapCellDelta]:
        state = self._require(mission_id)
        async with state._lock:
            old = state.grid.probabilities.copy()
            apply_negative_search(state.grid, polygon, pod)
            state.mpp = compute_mpp(state.grid, state.grid.probabilities)
            return _compute_delta(old, state.grid.probabilities, threshold=1e-10)

    async def update_pace(
        self,
        mission_id: UUID,
        step_sec: Optional[float] = None,
        update_interval_sec: Optional[float] = None,
    ) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            if step_sec is not None:
                state.step_sec = step_sec
            if update_interval_sec is not None:
                state.update_interval_sec = update_interval_sec
        return state

    def subscribe(self, mission_id: UUID) -> asyncio.Queue:
        state = self._require(mission_id)
        q: asyncio.Queue = asyncio.Queue()
        state.subscribers.append(q)
        return q

    def unsubscribe(self, mission_id: UUID, q: asyncio.Queue) -> None:
        state = self.get(mission_id)
        if state and q in state.subscribers:
            state.subscribers.remove(q)

    async def broadcast(self, mission_id: UUID, message: dict) -> None:
        state = self.get(mission_id)
        if not state:
            return
        for q in list(state.subscribers):
            await q.put(message)

    def build_engine_tick(self, mission_id: UUID) -> Optional[EngineTickMessage]:
        state = self.get(mission_id)
        if not state or state.mpp is None:
            return None
        return EngineTickMessage(
            event="engine_tick",
            tick_count=state.tick_count,
            lkp_coords=state.lkp,
            mpp_coords=state.mpp,
            layers=state.layers.as_dict(),
            particle_matrix=downsample_particles(state.particles, state.grid),
        )

    def _require(self, mission_id: UUID) -> MissionState:
        state = self.get(mission_id)
        if not state:
            raise KeyError(f"Mission {mission_id} not found")
        return state


def _compute_delta(
    old: "np.ndarray",
    new: "np.ndarray",
    threshold: float = 1e-7,
) -> list[HeatmapCellDelta]:
    import numpy as np

    deltas: list[HeatmapCellDelta] = []
    diff = np.abs(new - old)
    rows, cols = np.where(diff > threshold)
    for row, col in zip(rows.tolist(), cols.tolist()):
        deltas.append(HeatmapCellDelta(row=int(row), col=int(col), probability=float(new[row, col])))
    return deltas


mission_store = MissionStore()
