from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

from app.core.config import settings
from app.engine.grid_engine import GridEngine
from app.engine.grid_matrix import GridMatrix
from app.engine.grid_utils import compute_mpp, downsample_grid_peaks
from app.engine.layers.registry import ensure_min_one_dict, ensure_min_one_layer
from app.engine.node_builder import build_node_fields, env_for_layers
from app.geospatial.grid import ProbabilityGrid, create_empty_grid
from app.models.heatmap import HeatmapCellDelta
from app.models.layers import EngineTickMessage, LayerFlags
from app.models.mission import (
    BASE_STEP_SEC,
    LIVE_UPDATE_INTERVAL_SEC,
    LatLon,
    MissionMode,
    MissionStatus,
)
from app.services.env_ingestion import TerrainContext, build_terrain_context
from app.services.negative_search import apply_negative_search
from app.services.topo_reachability import (
    apply_reachability_to_grid,
    compute_reachability,
    lkp_to_grid_cell,
    mission_max_hours,
)
from app.engine.layers.registry import get_layer_weight


def _pace_to_timing(pace: float) -> tuple[float, float]:
    return BASE_STEP_SEC * pace, LIVE_UPDATE_INTERVAL_SEC


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
    mode: MissionMode
    created_at: datetime
    lkp_timestamp: Optional[datetime]
    grid_matrix: GridMatrix
    terrain_grid: ProbabilityGrid
    tick_count: int = 0
    pace: float = 1.0
    step_sec: float = BASE_STEP_SEC
    update_interval_sec: float = LIVE_UPDATE_INTERVAL_SEC
    simulation_running: bool = True
    terrain: Optional[TerrainContext] = None
    layers: LayerFlags = field(default_factory=LayerFlags)
    mpp: Optional[LatLon] = None
    tick_task: Optional[asyncio.Task] = field(default=None, repr=False)
    subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def grid(self) -> ProbabilityGrid:
        return self.grid_matrix.grid


class MissionStore:
    def __init__(self) -> None:
        self._missions: dict[UUID, MissionState] = {}
        self._engine = GridEngine()

    def get(self, mission_id: UUID) -> MissionState | None:
        return self._missions.get(mission_id)

    async def create(
        self,
        lkp: LatLon,
        sigma_0_m: Optional[float] = None,
        mode: MissionMode = MissionMode.LIVE,
        lkp_timestamp: Optional[datetime] = None,
        pace: float = 1.0,
        step_sec: Optional[float] = None,
        update_interval_sec: Optional[float] = None,
        layers: Optional[dict[str, bool]] = None,
    ) -> MissionState:
        del sigma_0_m  # grid engine uses t=0 impulse at LKP center
        mission_id = uuid4()
        size = settings.grid_size
        resolution = settings.grid_resolution_m
        terrain_grid = create_empty_grid(lkp, resolution, size)
        terrain = await build_terrain_context(terrain_grid)
        layer_flags = LayerFlags()
        if layers:
            filtered = ensure_min_one_dict(layers)
            layer_flags.apply_update(filtered)
        ensure_min_one_layer(layer_flags)

        if mode == MissionMode.LIVE:
            if step_sec is not None and update_interval_sec is not None:
                resolved_step = step_sec
                resolved_interval = update_interval_sec
            else:
                resolved_step, resolved_interval = _pace_to_timing(pace)
            simulation_running = True
        else:
            resolved_step, resolved_interval = _pace_to_timing(1.0)
            simulation_running = False

        node_fields = build_node_fields(
            terrain, size, weather_enabled=layer_flags.weather
        )
        grid_matrix = GridMatrix.create(lkp, size, resolution, node_fields)

        temp_state = MissionState(
            mission_id=uuid4(),
            lkp=lkp,
            status=MissionStatus.SEARCHING,
            mode=mode,
            created_at=datetime.now(timezone.utc),
            lkp_timestamp=lkp_timestamp,
            grid_matrix=grid_matrix,
            terrain_grid=terrain_grid,
            terrain=terrain,
            layers=layer_flags,
        )
        self._update_reachability(temp_state)
        grid_matrix.probabilities = self._finalize_probabilities(
            temp_state, grid_matrix.probabilities
        )
        grid_matrix.sync_to_grid()
        mpp = compute_mpp(grid_matrix.grid, grid_matrix.probabilities)

        lkp_ts = lkp_timestamp
        if lkp_ts is not None and lkp_ts.tzinfo is None:
            lkp_ts = lkp_ts.replace(tzinfo=timezone.utc)

        state = MissionState(
            mission_id=mission_id,
            lkp=lkp,
            status=MissionStatus.SEARCHING,
            mode=mode,
            created_at=datetime.now(timezone.utc),
            lkp_timestamp=lkp_ts,
            grid_matrix=grid_matrix,
            terrain_grid=terrain_grid,
            pace=pace if mode == MissionMode.LIVE else 1.0,
            step_sec=resolved_step,
            update_interval_sec=resolved_interval,
            simulation_running=simulation_running,
            terrain=terrain,
            layers=layer_flags,
            mpp=mpp,
        )
        self._missions[mission_id] = state

        if mode == MissionMode.OFFLINE and lkp_ts is not None:
            await self._run_offline_batch(state)

        return state

    async def _run_offline_batch(self, state: MissionState) -> None:
        if state.lkp_timestamp is None:
            return
        now = datetime.now(timezone.utc)
        elapsed_sec = max(0.0, (now - state.lkp_timestamp).total_seconds())
        n_ticks = max(1, int(elapsed_sec / state.step_sec))
        async with state._lock:
            for _ in range(n_ticks):
                await self._tick_unlocked(state)

    async def _tick_unlocked(self, state: MissionState) -> None:
        prior_probs = state.grid_matrix.probabilities.copy()
        self._update_reachability(state)

        env = env_for_layers(state.layers.weather)
        current_probs = self._engine.tick(
            state.grid_matrix,
            state.layers,
            dt_sec=state.step_sec,
            tick_count=state.tick_count,
            env=env,
        )
        current_probs = self._finalize_probabilities(state, current_probs)
        blended = self._blend_history(prior_probs, current_probs)
        state.grid_matrix.probabilities = blended
        state.grid_matrix.sync_to_grid()
        state.tick_count += 1
        state.mpp = compute_mpp(state.grid_matrix.grid, blended)

    def _blend_history(self, prior: np.ndarray, current: np.ndarray) -> np.ndarray:
        decay = settings.heatmap_history_decay
        if prior.sum() <= 0 or decay <= 0:
            return current
        if decay >= 1.0:
            return prior
        blended = decay * prior + (1.0 - decay) * current
        total = blended.sum()
        if total > 0:
            blended /= total
        return blended

    def _update_reachability(self, state: MissionState) -> None:
        if not state.layers.topography or state.terrain is None:
            return
        now = datetime.now(timezone.utc)
        lkp_ts = state.lkp_timestamp or state.created_at
        max_h = mission_max_hours(
            tick_count=state.tick_count,
            step_sec=state.step_sec,
            lkp_timestamp=lkp_ts,
            now=now,
        )
        start_row, start_col = lkp_to_grid_cell(
            state.terrain_grid,
            state.terrain_grid.crs.origin_e,
            state.terrain_grid.crs.origin_n,
        )
        reach = compute_reachability(
            state.terrain_grid,
            state.terrain.elevation,
            start_row,
            start_col,
            max_h,
        )
        state.terrain.reachability = reach
        state.grid_matrix.node_fields.reachability = reach.astype(np.float64, copy=True)

    def _finalize_probabilities(
        self, state: MissionState, probs: np.ndarray
    ) -> np.ndarray:
        if (
            state.layers.topography
            and state.terrain is not None
            and state.terrain.reachability is not None
        ):
            weight = get_layer_weight(state.layers, "topography")
            return apply_reachability_to_grid(
                probs,
                state.grid_matrix.grid,
                state.terrain_grid,
                state.terrain.reachability,
                weight,
            )
        return probs

    async def tick(self, mission_id: UUID) -> TickResult:
        state = self._require(mission_id)
        if not state.simulation_running and state.mode == MissionMode.LIVE:
            return TickResult(deltas=[], engine_tick=None)
        async with state._lock:
            await self._tick_unlocked(state)
            engine_tick = EngineTickMessage(
                event="engine_tick",
                tick_count=state.tick_count,
                lkp_coords=state.lkp,
                mpp_coords=state.mpp,
                layers=state.layers.as_dict(),
                particle_matrix=downsample_grid_peaks(
                    state.grid_matrix.probabilities, state.grid_matrix.grid
                ),
            )
            return TickResult(deltas=[], engine_tick=engine_tick, full_refresh=True)

    async def update_layers(self, mission_id: UUID, layers: dict[str, bool]) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            filtered = ensure_min_one_dict(layers)
            state.layers.apply_update(filtered)
        return state

    async def pause(self, mission_id: UUID) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            state.simulation_running = False
        return state

    async def resume(self, mission_id: UUID) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            if state.mode == MissionMode.LIVE:
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
            old = state.grid_matrix.probabilities.copy()
            apply_negative_search(state.grid_matrix.grid, polygon, pod)
            state.grid_matrix.sync_from_grid()
            state.mpp = compute_mpp(state.grid_matrix.grid, state.grid_matrix.probabilities)
            return _compute_delta(old, state.grid_matrix.probabilities, threshold=1e-10)

    async def update_pace(
        self,
        mission_id: UUID,
        pace: Optional[float] = None,
        step_sec: Optional[float] = None,
        update_interval_sec: Optional[float] = None,
    ) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            if state.mode != MissionMode.LIVE:
                return state
            if pace is not None:
                state.pace = pace
                state.step_sec, state.update_interval_sec = _pace_to_timing(pace)
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
            particle_matrix=downsample_grid_peaks(
                state.grid_matrix.probabilities, state.grid_matrix.grid
            ),
        )

    def _require(self, mission_id: UUID) -> MissionState:
        state = self.get(mission_id)
        if not state:
            raise KeyError(f"Mission {mission_id} not found")
        return state


def _compute_delta(
    old: np.ndarray,
    new: np.ndarray,
    threshold: float = 1e-7,
) -> list[HeatmapCellDelta]:
    deltas: list[HeatmapCellDelta] = []
    diff = np.abs(new - old)
    rows, cols = np.where(diff > threshold)
    for row, col in zip(rows.tolist(), cols.tolist()):
        deltas.append(
            HeatmapCellDelta(row=int(row), col=int(col), probability=float(new[row, col]))
        )
    return deltas


mission_store = MissionStore()
