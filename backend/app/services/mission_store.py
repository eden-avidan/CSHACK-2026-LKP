from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

from app.core.config import settings
from app.engine.grid_engine import GridEngine
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.grid_utils import compute_mpp, downsample_grid_peaks
from app.engine.layers.registry import ensure_min_one_dict, ensure_min_one_layer
from app.engine.node_builder import build_node_fields, copy_node_fields, env_for_layers
from app.geospatial.grid import ProbabilityGrid, create_empty_grid
from app.models.heatmap import HeatmapCellDelta
from app.models.detection import DetectionEventMessage, DroneTrackItem, DroneTrackMessage
from app.models.layers import EngineTickMessage, LayerFlags
from app.models.personality import PersonalityProfile
from app.models.mission import (
    BASE_STEP_SEC,
    LatLon,
    MissionMode,
    MissionStatus,
    live_update_interval_sec,
)
from app.services.drone_detection import (
    DetectionRecord,
    cells_within_radius,
    get_default_detection_jsonl_path,
    get_drone_sortie_paths,
    load_detection_records,
    load_detection_records_cached,
    map_detection_to_grid_cell,
    subsample_records,
)
from app.services.env_ingestion import TerrainContext, build_terrain_context
from app.services.negative_search import apply_negative_search
from app.services.marine_current import MarineCurrent, fetch_marine_current
from app.services.path_optimizer import DroneRoute, optimize_drone_route
from app.services.topo_reachability import (
    compute_reachability,
    compute_reachability_score,
    lkp_to_grid_cell,
    mission_max_hours,
)


def _pace_to_timing(pace: float) -> tuple[float, float]:
    return BASE_STEP_SEC * pace, live_update_interval_sec()


def _lkp_in_sea(terrain: Optional[TerrainContext], size: int) -> bool:
    """True when the LKP cell (grid center) is water rather than land."""
    if terrain is None:
        return False
    r = c = size // 2
    try:
        return not bool(terrain.is_land[r, c])
    except (IndexError, TypeError):
        return False


@dataclass
class TickResult:
    deltas: list[HeatmapCellDelta]
    engine_tick: Optional[EngineTickMessage]
    full_refresh: bool = False
    detection_events: list[DetectionEventMessage] = field(default_factory=list)
    drone_track: Optional[DroneTrackMessage] = None


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
    simulation_start_timestamp: Optional[datetime] = None
    tick_count: int = 0
    drone_start_tick: int = 0
    pace: float = 1.0
    step_sec: float = BASE_STEP_SEC
    update_interval_sec: float = field(default_factory=live_update_interval_sec)
    simulation_running: bool = True
    terrain: Optional[TerrainContext] = None
    initial_node_fields: Optional[NodeFields] = None
    marine_current: Optional[MarineCurrent] = None
    layers: LayerFlags = field(default_factory=LayerFlags)
    personality: PersonalityProfile | None = None
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
        simulation_start_timestamp: Optional[datetime] = None,
        pace: float = 1.0,
        step_sec: Optional[float] = None,
        update_interval_sec: Optional[float] = None,
        layers: Optional[dict[str, bool]] = None,
        personality: PersonalityProfile | None = None,
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

        # If the LKP is on water, switch to a sea-drift-only model: the subject
        # is adrift, so the land-based layers don't apply and the spread is
        # governed solely by the constant drift current.
        if _lkp_in_sea(terrain, size):
            layer_flags = LayerFlags(
                topography=False,
                roads=False,
                personality=False,
                weather=False,
                sea_drift=True,
            )

        marine_current = await fetch_marine_current(lkp.lat, lkp.lon)

        resolved_personality: PersonalityProfile | None = None
        if layer_flags.personality:
            resolved_personality = personality or PersonalityProfile()

        if mode == MissionMode.LIVE:
            if step_sec is not None and update_interval_sec is not None:
                resolved_step = step_sec
                resolved_interval = update_interval_sec
            else:
                resolved_step, resolved_interval = _pace_to_timing(pace)
            simulation_running = True
        else:
            if step_sec is not None and update_interval_sec is not None:
                resolved_step = step_sec
                resolved_interval = update_interval_sec
            else:
                resolved_step, resolved_interval = _pace_to_timing(pace)
            simulation_running = True

        node_fields = build_node_fields(
            terrain,
            size,
            weather_enabled=layer_flags.weather,
            marine_current=marine_current,
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
            personality=resolved_personality,
        )
        self._update_reachability(temp_state)
        initial_node_fields = copy_node_fields(grid_matrix.node_fields)
        grid_matrix.probabilities = self._recompute_from_impulse(temp_state)
        grid_matrix.sync_to_grid()
        mpp = compute_mpp(grid_matrix.grid, grid_matrix.probabilities)

        lkp_ts = lkp_timestamp
        if lkp_ts is not None and lkp_ts.tzinfo is None:
            lkp_ts = lkp_ts.replace(tzinfo=timezone.utc)
        sim_start_ts = simulation_start_timestamp
        if sim_start_ts is not None and sim_start_ts.tzinfo is None:
            sim_start_ts = sim_start_ts.replace(tzinfo=timezone.utc)
        if (
            mode == MissionMode.OFFLINE
            and lkp_ts is not None
            and sim_start_ts is not None
            and sim_start_ts < lkp_ts
        ):
            raise ValueError("simulation_start_timestamp must be on or after lkp_timestamp")

        state = MissionState(
            mission_id=mission_id,
            lkp=lkp,
            status=MissionStatus.SEARCHING,
            mode=mode,
            created_at=datetime.now(timezone.utc),
            lkp_timestamp=lkp_ts,
            simulation_start_timestamp=sim_start_ts,
            grid_matrix=grid_matrix,
            terrain_grid=terrain_grid,
            pace=pace,
            step_sec=resolved_step,
            update_interval_sec=resolved_interval,
            simulation_running=simulation_running,
            terrain=terrain,
            initial_node_fields=initial_node_fields,
            marine_current=marine_current,
            layers=layer_flags,
            mpp=mpp,
            personality=resolved_personality,
        )
        self._missions[mission_id] = state

        if mode == MissionMode.OFFLINE and lkp_ts is not None and sim_start_ts is not None:
            await self._run_offline_batch(state)

        return state

    @staticmethod
    def simulated_datetime(state: MissionState) -> datetime | None:
        """Wall-clock time in the scenario: LKP time + simulated ticks."""
        if state.lkp_timestamp is None:
            return None
        base = state.lkp_timestamp
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return base + timedelta(seconds=state.tick_count * state.step_sec)

    async def _run_offline_batch(self, state: MissionState) -> None:
        if state.lkp_timestamp is None or state.simulation_start_timestamp is None:
            return
        elapsed_sec = max(
            0.0,
            (state.simulation_start_timestamp - state.lkp_timestamp).total_seconds(),
        )
        n_ticks = int(elapsed_sec / state.step_sec)
        async with state._lock:
            for _ in range(n_ticks):
                await self._tick_unlocked(state, advance_drone=False)
            state.drone_start_tick = state.tick_count

    def _recompute_from_impulse(self, state: MissionState) -> np.ndarray:
        """Reset to LKP impulse and apply the active layer stack (no normalization)."""
        state.grid_matrix.initialize_t0()
        env = env_for_layers(state.layers.weather)
        return self._engine.apply_layers(
            state.grid_matrix,
            state.layers,
            dt_sec=state.step_sec,
            tick_count=state.tick_count,
            env=env,
            personality=state.personality if state.layers.personality else None,
        )

    async def _tick_unlocked(
        self, state: MissionState, advance_drone: bool = True
    ) -> tuple[list[DetectionEventMessage], Optional[DroneTrackMessage]]:
        prior_probs = state.grid_matrix.probabilities.copy()
        state.tick_count += 1
        self._update_reachability(state)

        current_probs = self._recompute_from_impulse(state)
        blended = self._blend_history(prior_probs, current_probs)
        state.grid_matrix.probabilities = blended
        if advance_drone:
            detection_events, drone_track = self._update_drone_coverage(state)
            self._apply_clean_suppression(state)
        else:
            detection_events, drone_track = [], None
        state.grid_matrix.sync_to_grid()
        state.mpp = compute_mpp(state.grid_matrix.grid, state.grid_matrix.probabilities)
        return detection_events, drone_track

    @staticmethod
    def _drone_elapsed_sec(state: MissionState) -> float:
        return max(0.0, (state.tick_count - state.drone_start_tick) * state.step_sec)

    @staticmethod
    def _drone_window_start_sec(state: MissionState) -> float:
        return max(0.0, (state.tick_count - state.drone_start_tick - 1) * state.step_sec)

    @staticmethod
    def _drone_clock_start(state: MissionState) -> datetime:
        if state.mode == MissionMode.OFFLINE and state.simulation_start_timestamp is not None:
            return state.simulation_start_timestamp
        return state.lkp_timestamp or state.created_at

    def _blend_history(self, prior: np.ndarray, current: np.ndarray) -> np.ndarray:
        decay = settings.heatmap_history_decay
        if prior.sum() <= 0 or decay <= 0:
            return current
        if decay >= 1.0:
            return prior
        return decay * prior + (1.0 - decay) * current

    def _update_drone_coverage(
        self, state: MissionState
    ) -> tuple[list[DetectionEventMessage], Optional[DroneTrackMessage]]:
        """Advance drone coverage for this tick.

        Decays the per-cell "clean" score once, then folds in two sources:
          1. The real detection feed (absolute timestamps vs the mission clock) —
             drives person-found detection events and clears found cells.
          2. The sequential drone sorties (timestamps interpreted *relative to
             mission start*) — each drone walks its path as ticks advance, one
             after another, marking swept cells clean (1.0), emitting detection
             events when the "found" drone spots the person, and yielding the
             live position(s) + flown path(s).
        """
        clean = state.grid_matrix.node_fields.searched_clean
        clean *= settings.drone_clean_decay

        events = self._mark_detection_feed(state, clean)
        sortie_events, track = self._mark_drone_sorties(state, clean)
        events.extend(sortie_events)
        return events, track

    def _mark_detection_feed(
        self, state: MissionState, clean: np.ndarray
    ) -> list[DetectionEventMessage]:
        """Real detection feed, matched on absolute time, emits person-found events."""
        clock_start = self._drone_clock_start(state)
        tick_start = clock_start + timedelta(seconds=self._drone_window_start_sec(state))
        tick_end = tick_start + timedelta(seconds=state.step_sec)
        records = load_detection_records(get_default_detection_jsonl_path())
        events: list[DetectionEventMessage] = []

        for record in records:
            if not (tick_start <= record.timestamp < tick_end):
                continue
            position = self._record_position(record)
            if record.person:
                events.append(
                    DetectionEventMessage(
                        mission_id=state.mission_id,
                        timestamp=record.timestamp,
                        confidence=record.confidence,
                        confidence_percent=record.confidence_percent,
                        frame=record.frame,
                        bbox=record.bbox,
                        position=position,
                    )
                )
            self._mark_clean_cell(state, clean, record, position)
        return events

    def _sortie_plan(self, state: MissionState) -> list[dict]:
        """Per-sortie timeline relative to mission start (launch delay + gaps).

        Each entry: ``asset_id``, ``found`` (sortie contains a person), the
        subsampled ``records``, the file ``base`` time, the mission-elapsed
        ``start`` second the drone launches at, and the flight ``duration``.
        """
        delay = settings.drone_track_launch_delay_sec
        gap = settings.drone_sortie_gap_sec
        max_points = settings.drone_sortie_max_points
        offsets = settings.drone_sortie_north_offset_list
        plan: list[dict] = []
        start = delay
        for idx, path in enumerate(get_drone_sortie_paths()):
            records = subsample_records(load_detection_records_cached(path), max_points)
            if not records:
                continue
            base = records[0].timestamp
            duration = (records[-1].timestamp - base).total_seconds()
            north_m = offsets[idx] if idx < len(offsets) else 0.0
            plan.append(
                {
                    "asset_id": f"drone-{idx + 1}",
                    "found": any(r.person for r in records),
                    "records": records,
                    "base": base,
                    "start": start,
                    "duration": duration,
                    # Per-sortie north shift, applied to every revealed lat.
                    "lat_offset_deg": north_m / 111_320.0,
                }
            )
            start += duration + gap
        return plan

    def _mark_drone_sorties(
        self, state: MissionState, clean: np.ndarray
    ) -> tuple[list[DetectionEventMessage], Optional[DroneTrackMessage]]:
        """Advance every sortie: mark this tick's swept cells, emit detection
        events for person-found points, and snapshot the revealed track(s)."""
        plan = self._sortie_plan(state)
        if not plan:
            return [], None

        win_end = self._drone_elapsed_sec(state)
        win_start = self._drone_window_start_sec(state)
        events: list[DetectionEventMessage] = []

        for sortie in plan:
            start, base = sortie["start"], sortie["base"]
            lat_off = sortie["lat_offset_deg"]
            for record in sortie["records"]:
                reveal = start + (record.timestamp - base).total_seconds()
                if not (win_start <= reveal < win_end):
                    continue
                position = self._record_position(record, lat_off)
                self._mark_clean_cell(state, clean, record, position)
                if record.person:
                    events.append(
                        DetectionEventMessage(
                            mission_id=state.mission_id,
                            asset_id=sortie["asset_id"],
                            timestamp=record.timestamp,
                            confidence=record.confidence,
                            confidence_percent=record.confidence_percent,
                            frame=record.frame,
                            bbox=record.bbox,
                            position=position,
                        )
                    )

        return events, self._sortie_track_snapshot(state, plan)

    def _sortie_track_snapshot(
        self, state: MissionState, plan: list[dict]
    ) -> Optional[DroneTrackMessage]:
        """Cumulative path + current position of every drone revealed so far."""
        now = self._drone_elapsed_sec(state)
        items: list[DroneTrackItem] = []
        for sortie in plan:
            start, base, duration = sortie["start"], sortie["base"], sortie["duration"]
            lat_off = sortie["lat_offset_deg"]
            if start >= now:
                continue  # not launched yet
            revealed = [
                r
                for r in sortie["records"]
                if r.latitude is not None
                and r.longitude is not None
                and start + (r.timestamp - base).total_seconds() < now
            ]
            if not revealed:
                continue
            path = [[float(r.longitude), float(r.latitude) + lat_off] for r in revealed]
            last = revealed[-1]
            items.append(
                DroneTrackItem(
                    asset_id=sortie["asset_id"],
                    found=bool(sortie["found"]),
                    active=start <= now < start + duration,
                    position=LatLon(lat=float(last.latitude) + lat_off, lon=float(last.longitude)),
                    path=path,
                )
            )

        if not items:
            return None
        active = next((it for it in reversed(items) if it.active), items[-1])
        return DroneTrackMessage(
            mission_id=state.mission_id,
            timestamp=datetime.now(timezone.utc),
            asset_id=active.asset_id,
            position=active.position,
            path=active.path,
            drones=items,
        )

    @staticmethod
    def _record_position(
        record: DetectionRecord, lat_offset_deg: float = 0.0
    ) -> Optional[LatLon]:
        if record.latitude is None or record.longitude is None:
            return None
        try:
            return LatLon(lat=record.latitude + lat_offset_deg, lon=record.longitude)
        except ValueError:
            return None

    def _mark_clean_cell(
        self,
        state: MissionState,
        clean: np.ndarray,
        record: DetectionRecord,
        position: Optional[LatLon],
    ) -> None:
        if position is None:
            return
        # Pure 2D overflight match: the cell(s) the drone flew over are cleared
        # regardless of the drone's flight altitude. The drone's altitude is an
        # absolute flight height (e.g. ~505 m ASL) while node altitude is the
        # ground DEM elevation, so they must never be compared to decide 2D
        # coverage — doing so silently dropped every real-data clear.
        #
        # A "no person" fix clears its whole camera footprint (a disc of radius
        # drone_coverage_radius_m), so a sweep suppresses a realistic area. A
        # positive detection is point-like and only touches its own cell.
        if record.person:
            try:
                cells = [
                    map_detection_to_grid_cell(
                        state.grid_matrix.grid, position.lat, position.lon
                    )
                ]
            except ValueError:
                return
            value = 0.0
        else:
            try:
                cells = cells_within_radius(
                    state.grid_matrix.grid,
                    position.lat,
                    position.lon,
                    settings.drone_coverage_radius_m,
                )
            except ValueError:
                return
            value = 1.0
        for row, col in cells:
            clean[row, col] = value

    def build_drone_track(self, mission_id: UUID) -> Optional[DroneTrackMessage]:
        """Current flown path(s) + position(s) without advancing the sim (for connect)."""
        state = self.get(mission_id)
        if not state:
            return None
        if self._drone_elapsed_sec(state) <= 0:
            return None
        return self._sortie_track_snapshot(state, self._sortie_plan(state))

    def _apply_clean_suppression(self, state: MissionState) -> None:
        """Lower probability in cleared cells: prob *= (1 - strength*clean), renorm."""
        clean = state.grid_matrix.node_fields.searched_clean
        if float(clean.max()) <= 0.0:
            return
        strength = settings.drone_clean_suppression_strength
        factor = 1.0 - strength * np.clip(clean, 0.0, 1.0)
        probs = state.grid_matrix.probabilities * factor
        total = float(probs.sum())
        if total <= 0.0:
            return
        state.grid_matrix.probabilities = probs / total

    def _update_reachability(self, state: MissionState) -> None:
        if not state.layers.topography or state.terrain is None:
            return
        if state.mode == MissionMode.OFFLINE:
            max_h = max(
                state.step_sec / 3600.0,
                ((state.tick_count + 1) * state.step_sec) / 3600.0,
            )
        else:
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
        score = compute_reachability_score(
            state.terrain_grid,
            state.terrain.elevation,
            start_row,
            start_col,
            max_h,
        )
        state.terrain.reachability = reach
        state.terrain.reachability_score = score
        state.grid_matrix.node_fields.reachability = reach.astype(np.float64, copy=True)
        state.grid_matrix.node_fields.reachability_score = score.astype(np.float64, copy=True)

    async def tick(self, mission_id: UUID) -> TickResult:
        state = self._require(mission_id)
        if not state.simulation_running:
            return TickResult(deltas=[], engine_tick=None)
        async with state._lock:
            detection_events, drone_track = await self._tick_unlocked(state)
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
            return TickResult(
                deltas=[],
                engine_tick=engine_tick,
                full_refresh=True,
                detection_events=detection_events,
                drone_track=drone_track,
            )

    async def update_layers(self, mission_id: UUID, layers: dict[str, bool]) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
            filtered = ensure_min_one_dict(layers)
            state.layers.apply_update(filtered)
            self._update_reachability(state)
            probs = self._recompute_from_impulse(state)
            state.grid_matrix.probabilities = probs
            state.grid_matrix.sync_to_grid()
            state.mpp = compute_mpp(state.grid_matrix.grid, probs)
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
            old = state.grid_matrix.probabilities.copy()
            apply_negative_search(state.grid_matrix.grid, polygon, pod)
            state.grid_matrix.sync_from_grid()
            state.mpp = compute_mpp(state.grid_matrix.grid, state.grid_matrix.probabilities)
            return _compute_delta(old, state.grid_matrix.probabilities, threshold=1e-10)

    async def drone_route(self, mission_id: UUID) -> DroneRoute:
        state = self._require(mission_id)
        async with state._lock:
            return optimize_drone_route(state.grid)

    async def update_pace(
        self,
        mission_id: UUID,
        pace: Optional[float] = None,
        step_sec: Optional[float] = None,
        update_interval_sec: Optional[float] = None,
    ) -> MissionState:
        state = self._require(mission_id)
        async with state._lock:
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
