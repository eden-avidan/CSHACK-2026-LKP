from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon, grid_utm_bounds
from app.layers.pipeline import run_kde_pipeline, run_predict_pipeline
from app.models.layers import LayerFlags
from app.models.mission import LatLon
from app.services.particle_types import EnvForcing, Particles, apply_edge_fade

if TYPE_CHECKING:
    from app.services.env_ingestion import TerrainContext


def get_mock_env() -> EnvForcing:
    return EnvForcing(u_w=4.0, v_w=2.5, u_c=0.0, v_c=0.0)


def zero_env() -> EnvForcing:
    return EnvForcing(u_w=0.0, v_w=0.0, u_c=0.0, v_c=0.0)


def initialize_particles(
    origin_e: float,
    origin_n: float,
    n_particles: int,
    sigma_0_m: float,
) -> Particles:
    rng = np.random.default_rng()
    eastings = origin_e + rng.normal(0, sigma_0_m, n_particles)
    northings = origin_n + rng.normal(0, sigma_0_m, n_particles)
    v_n = rng.normal(0, settings.sigma_v, n_particles)
    v_e = rng.normal(0, settings.sigma_v, n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    return Particles(eastings, northings, v_n, v_e, weights)


def predict_step(
    particles: Particles,
    env: EnvForcing,
    dt: float = 1.0,
    terrain: Optional["TerrainContext"] = None,
    grid: Optional[ProbabilityGrid] = None,
    terrain_grid: Optional[ProbabilityGrid] = None,
    layers: Optional[LayerFlags] = None,
) -> Particles:
    layer_flags = layers or LayerFlags()
    return run_predict_pipeline(
        particles, env, dt, terrain, grid, terrain_grid, layer_flags
    )


def apply_grid_bounds(particles: Particles, grid: ProbabilityGrid) -> Particles:
    min_e, min_n, max_e, max_n = grid_utm_bounds(grid)
    span_e = max_e - min_e
    span_n = max_n - min_n
    margin_e = span_e * settings.boundary_soft_margin_frac
    margin_n = span_n * settings.boundary_soft_margin_frac
    damp = settings.boundary_reflect_damping

    eastings = particles.eastings.copy()
    northings = particles.northings.copy()
    v_e = particles.v_e.copy()
    v_n = particles.v_n.copy()

    over_e = eastings > max_e
    eastings[over_e] = 2 * max_e - eastings[over_e]
    v_e[over_e] *= -damp

    under_e = eastings < min_e
    eastings[under_e] = 2 * min_e - eastings[under_e]
    v_e[under_e] *= -damp

    over_n = northings > max_n
    northings[over_n] = 2 * max_n - northings[over_n]
    v_n[over_n] *= -damp

    under_n = northings < min_n
    northings[under_n] = 2 * min_n - northings[under_n]
    v_n[under_n] *= -damp

    near_max_e = eastings > (max_e - margin_e)
    eastings[near_max_e] -= (eastings[near_max_e] - (max_e - margin_e)) * 0.15
    v_e[near_max_e] *= 0.9

    near_min_e = eastings < (min_e + margin_e)
    eastings[near_min_e] += ((min_e + margin_e) - eastings[near_min_e]) * 0.15
    v_e[near_min_e] *= 0.9

    near_max_n = northings > (max_n - margin_n)
    northings[near_max_n] -= (northings[near_max_n] - (max_n - margin_n)) * 0.15
    v_n[near_max_n] *= 0.9

    near_min_n = northings < (min_n + margin_n)
    northings[near_min_n] += ((min_n + margin_n) - northings[near_min_n]) * 0.15
    v_n[near_min_n] *= 0.9

    return Particles(eastings, northings, v_n, v_e, particles.weights.copy())


def effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / np.sum(weights**2)


def systematic_resample(particles: Particles) -> Particles:
    n = particles.count
    weights = particles.weights
    rng = np.random.default_rng()
    u0 = rng.uniform(0, 1.0 / n)
    positions = u0 + np.arange(n) / n
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, positions)
    indices = np.clip(indices, 0, n - 1)

    return Particles(
        eastings=particles.eastings[indices],
        northings=particles.northings[indices],
        v_n=particles.v_n[indices],
        v_e=particles.v_e[indices],
        weights=np.full(n, 1.0 / n),
    )


def resample_if_needed(particles: Particles) -> Particles:
    n_eff = effective_sample_size(particles.weights)
    if n_eff < particles.count / 2:
        return systematic_resample(particles)
    return particles


def _apply_edge_fade(grid: np.ndarray, fade_cells: int) -> np.ndarray:
    return apply_edge_fade(grid, fade_cells)


def rasterize_kde(
    particles: Particles,
    grid_rows: int,
    grid_cols: int,
    resolution_m: float,
    origin_e: float,
    origin_n: float,
    road_proximity: Optional[np.ndarray] = None,
    roads_layer: bool = False,
    edge_fade_cells: Optional[int] = None,
    layers: Optional[LayerFlags] = None,
) -> np.ndarray:
    layer_flags = layers or LayerFlags()
    if roads_layer and not layer_flags.roads:
        layer_flags = LayerFlags(
            topography=layer_flags.topography,
            roads=True,
            subject_injured=layer_flags.subject_injured,
            weather=layer_flags.weather,
        )
    return run_kde_pipeline(
        particles,
        grid_rows,
        grid_cols,
        resolution_m,
        origin_e,
        origin_n,
        road_proximity,
        layer_flags,
        edge_fade_cells,
    )


def compute_mpp(grid: ProbabilityGrid, probabilities: np.ndarray) -> LatLon:
    """Most Probable Position: weighted centroid of top 5% probability cells."""
    margin = max(1, settings.kde_edge_fade_cells // 2)
    masked = probabilities.copy()
    if margin * 2 < masked.shape[0] and margin * 2 < masked.shape[1]:
        masked[:margin, :] = 0.0
        masked[-margin:, :] = 0.0
        masked[:, :margin] = 0.0
        masked[:, -margin:] = 0.0

    flat = masked.ravel()
    nonzero = flat[flat > 1e-12]
    if nonzero.size == 0:
        flat = probabilities.ravel()
        nonzero = flat[flat > 1e-12]
        if nonzero.size == 0:
            return LatLon(lat=grid.metadata.origin.lat, lon=grid.metadata.origin.lon)
        masked = probabilities

    threshold = float(np.percentile(nonzero, 95))
    rows, cols = np.where(masked >= threshold)
    if rows.size == 0:
        peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
        rows, cols = np.array([peak[0]]), np.array([peak[1]])

    weights = masked[rows, cols]
    w_sum = weights.sum()
    if w_sum <= 0:
        return LatLon(lat=grid.metadata.origin.lat, lon=grid.metadata.origin.lon)

    lat_acc = 0.0
    lon_acc = 0.0
    for row, col, w in zip(rows, cols, weights):
        lat, lon = cell_centroid_latlon(grid, int(row), int(col))
        lat_acc += lat * w
        lon_acc += lon * w

    return LatLon(lat=lat_acc / w_sum, lon=lon_acc / w_sum)


def downsample_particles(
    particles: Particles,
    grid: ProbabilityGrid,
    limit: Optional[int] = None,
) -> list[list[float]]:
    cap = limit or settings.engine_tick_particle_limit
    n = particles.count
    if n == 0:
        return []

    indices = np.argsort(particles.weights)[::-1][:cap]
    matrix: list[list[float]] = []
    for i in indices:
        lat, lon = grid.crs.to_wgs84(float(particles.eastings[i]), float(particles.northings[i]))
        matrix.append([lat, lon, float(particles.weights[i])])
    return matrix
