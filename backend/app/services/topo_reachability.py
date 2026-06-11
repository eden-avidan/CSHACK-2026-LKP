"""Tobler hiking + Dijkstra reachability (ported from topo_layout/mobility.py)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal

from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_utm

ProbabilityMethod = Literal["linear", "exponential"]


@dataclass
class TerrainInfluenceConfig:
    steep_weight: float = 0.7
    cliff_like_weight: float = 0.2
    valley_weight: float = 1.15
    ridge_weight: float = 0.9


@dataclass
class TerrainClassification:
    steep_mask: np.ndarray
    cliff_like_mask: np.ndarray
    valley_mask: np.ndarray
    ridge_mask: np.ndarray


def tobler_hiking_speed_kmh(signed_slope_grade: np.ndarray | float) -> np.ndarray | float:
    """Tobler hiking function: speed in km/h from signed slope grade (rise/run)."""
    return 6.0 * np.exp(-3.5 * np.abs(np.asarray(signed_slope_grade) + 0.05))


def classify_terrain_from_elevation(
    elevation: np.ndarray,
    resolution_m: float,
    *,
    steep_threshold_deg: float | None = None,
    cliff_threshold_deg: float | None = None,
    neighborhood_size: int | None = None,
    ridge_threshold_m: float | None = None,
    valley_threshold_m: float | None = None,
) -> TerrainClassification:
    steep_threshold_deg = steep_threshold_deg or settings.topo_steep_threshold_deg
    cliff_threshold_deg = cliff_threshold_deg or settings.topo_cliff_threshold_deg
    neighborhood_size = neighborhood_size or settings.topo_neighborhood_size
    ridge_threshold_m = ridge_threshold_m or settings.topo_ridge_threshold_m
    valley_threshold_m = valley_threshold_m or settings.topo_valley_threshold_m

    gy, gx = np.gradient(elevation.astype(np.float64), resolution_m, resolution_m)
    slope_degrees = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    steep_mask = slope_degrees >= steep_threshold_deg
    cliff_like_mask = slope_degrees >= cliff_threshold_deg

    pad = neighborhood_size // 2
    padded = np.pad(elevation, pad_width=pad, mode="edge")
    windows = sliding_window_view(padded, (neighborhood_size, neighborhood_size))
    neighborhood_sum = windows.sum(axis=(-1, -2), dtype=np.float64) - elevation
    neighbor_count = (neighborhood_size * neighborhood_size) - 1
    tpi = elevation - neighborhood_sum / neighbor_count

    return TerrainClassification(
        steep_mask=steep_mask,
        cliff_like_mask=cliff_like_mask,
        valley_mask=tpi <= -valley_threshold_m,
        ridge_mask=tpi >= ridge_threshold_m,
    )


def terrain_probability_weight(
    terrain: TerrainClassification,
    config: TerrainInfluenceConfig | None = None,
) -> np.ndarray:
    config = config or TerrainInfluenceConfig(
        steep_weight=settings.topo_steep_weight,
        cliff_like_weight=settings.topo_cliff_like_weight,
        valley_weight=settings.topo_valley_weight,
        ridge_weight=settings.topo_ridge_weight,
    )
    weight = np.ones(terrain.steep_mask.shape, dtype=np.float64)
    weight *= np.where(terrain.steep_mask, config.steep_weight, 1.0)
    weight *= np.where(terrain.cliff_like_mask, config.cliff_like_weight, 1.0)
    weight *= np.where(terrain.valley_mask, config.valley_weight, 1.0)
    weight *= np.where(terrain.ridge_mask, config.ridge_weight, 1.0)
    return weight


def lkp_to_grid_cell(grid: ProbabilityGrid, lkp_e: float, lkp_n: float) -> tuple[int, int]:
    res = grid.metadata.resolution_m
    half = (grid.rows * res) / 2.0
    col = int((lkp_e - (grid.crs.origin_e - half)) / res)
    row = int(((grid.crs.origin_n + half) - lkp_n) / res)
    row = min(grid.rows - 1, max(0, row))
    col = min(grid.cols - 1, max(0, col))
    return row, col


def least_travel_time_hours(
    elevation: np.ndarray,
    resolution_m: float,
    start_row: int,
    start_col: int,
    max_hours: float,
) -> np.ndarray:
    height, width = elevation.shape
    travel_time = np.full((height, width), np.inf, dtype=np.float64)
    travel_time[start_row, start_col] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, start_row, start_col)]

    neighbor_offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    elev = elevation.astype(np.float64, copy=False)

    while heap:
        current_time, row, col = heapq.heappop(heap)
        if current_time > travel_time[row, col]:
            continue
        if current_time > max_hours:
            continue

        for delta_row, delta_col in neighbor_offsets:
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue

            horizontal_distance_m = float(np.hypot(delta_col * resolution_m, delta_row * resolution_m))
            elevation_change_m = float(elev[next_row, next_col] - elev[row, col])
            signed_slope_grade = elevation_change_m / horizontal_distance_m
            speed_kmh = float(tobler_hiking_speed_kmh(signed_slope_grade))
            speed_kmh = max(speed_kmh, 1e-6)
            step_time_hours = (horizontal_distance_m / 1000.0) / speed_kmh
            next_time = current_time + step_time_hours

            if next_time < travel_time[next_row, next_col] and next_time <= max_hours:
                travel_time[next_row, next_col] = next_time
                heapq.heappush(heap, (next_time, next_row, next_col))

    return travel_time


def travel_time_to_probability(
    travel_time_hours: np.ndarray,
    max_hours: float,
    method: ProbabilityMethod | None = None,
    terrain_weight: np.ndarray | None = None,
) -> np.ndarray:
    method = method or settings.topo_probability_method  # type: ignore[assignment]
    reachable = np.isfinite(travel_time_hours) & (travel_time_hours <= max_hours)
    probability = np.zeros_like(travel_time_hours, dtype=np.float64)

    if method == "linear":
        probability[reachable] = (max_hours - travel_time_hours[reachable]) / max_hours
    elif method == "exponential":
        decay_hours = max_hours / 3.0
        probability[reachable] = np.exp(-travel_time_hours[reachable] / decay_hours)
    else:
        raise ValueError(f"Unsupported probability method: {method}")

    if terrain_weight is not None:
        probability[reachable] *= terrain_weight[reachable]

    total = float(probability.sum())
    if total > 0:
        probability /= total
    return probability


def compute_reachability_score(
    terrain_grid: ProbabilityGrid,
    elevation: np.ndarray,
    start_row: int,
    start_col: int,
    max_hours: float,
) -> np.ndarray:
    """Linear 0..1 walking reachability for maps: 1 at LKP, 0 beyond horizon.

    Unlike ``compute_reachability`` this is NOT normalized to sum to 1, so the
    overlay shows a smooth falloff disk instead of a flat probability share.
    """
    if max_hours <= 0:
        max_hours = 0.1
    travel_time = least_travel_time_hours(
        elevation,
        terrain_grid.metadata.resolution_m,
        start_row,
        start_col,
        max_hours,
    )
    reachable = np.isfinite(travel_time) & (travel_time <= max_hours)
    score = np.zeros_like(travel_time, dtype=np.float64)
    score[reachable] = (max_hours - travel_time[reachable]) / max_hours
    return score


def compute_reachability(
    terrain_grid: ProbabilityGrid,
    elevation: np.ndarray,
    start_row: int,
    start_col: int,
    max_hours: float,
    *,
    use_terrain_influence: bool = True,
) -> np.ndarray:
    if max_hours <= 0:
        max_hours = 0.1

    travel_time = least_travel_time_hours(
        elevation,
        terrain_grid.metadata.resolution_m,
        start_row,
        start_col,
        max_hours,
    )

    terrain_weight = None
    if use_terrain_influence:
        classification = classify_terrain_from_elevation(
            elevation, terrain_grid.metadata.resolution_m
        )
        terrain_weight = terrain_probability_weight(classification)

    return travel_time_to_probability(
        travel_time,
        max_hours=max_hours,
        terrain_weight=terrain_weight,
    )


def apply_reachability_to_grid(
    particle_probs: np.ndarray,
    display_grid: ProbabilityGrid,
    terrain_grid: ProbabilityGrid,
    reachability: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Blend particle KDE with Tobler/Dijkstra reachability prior."""
    from app.geospatial.grid import extract_field_for_grid

    if weight <= 0:
        return particle_probs

    reach_on_display = extract_field_for_grid(display_grid, terrain_grid, reachability)
    floor_frac = settings.topo_reachability_floor_frac
    if floor_frac > 0:
        reach_peak = float(reach_on_display.max())
        if reach_peak > 0:
            reach_on_display = np.maximum(reach_on_display, reach_peak * floor_frac)
    reach_on_display = np.maximum(reach_on_display, 1e-12)

    combined = particle_probs * ((1.0 - weight) + weight * reach_on_display)

    total = combined.sum()
    if total > 0:
        combined /= total
    return combined


def mission_max_hours(
    *,
    tick_count: int,
    step_sec: float,
    lkp_timestamp: datetime | None = None,
    now: datetime | None = None,
) -> float:
    """Compute search horizon in hours from elapsed real + simulated time."""
    from datetime import timezone as tz

    simulated_hours = (tick_count * step_sec) / 3600.0
    elapsed_hours = 0.0
    if lkp_timestamp is not None and now is not None:
        lkp_ts = lkp_timestamp
        if lkp_ts.tzinfo is None:
            lkp_ts = lkp_ts.replace(tzinfo=tz.utc)
        now_ts = now
        if now_ts.tzinfo is None:
            now_ts = now_ts.replace(tzinfo=tz.utc)
        elapsed_hours = max(0.0, (now_ts - lkp_ts).total_seconds() / 3600.0)
    return max(0.25, simulated_hours + elapsed_hours)
