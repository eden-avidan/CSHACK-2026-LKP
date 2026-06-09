from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon, grid_utm_bounds, particle_cell_indices
from app.models.layers import LayerFlags
from app.models.mission import LatLon

if TYPE_CHECKING:
    from app.services.env_ingestion import TerrainContext


@dataclass
class EnvForcing:
    u_w: float = 2.0  # north wind m/s
    v_w: float = 1.0  # east wind m/s
    u_c: float = 0.0
    v_c: float = 0.0


def get_mock_env() -> EnvForcing:
    return EnvForcing(u_w=4.0, v_w=2.5, u_c=0.0, v_c=0.0)


def zero_env() -> EnvForcing:
    return EnvForcing(u_w=0.0, v_w=0.0, u_c=0.0, v_c=0.0)


@dataclass
class Particles:
    eastings: np.ndarray
    northings: np.ndarray
    v_n: np.ndarray
    v_e: np.ndarray
    weights: np.ndarray

    @property
    def count(self) -> int:
        return len(self.weights)


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


def _apply_anisotropic_mobility(
    de: np.ndarray,
    dn: np.ndarray,
    aspect_e: np.ndarray,
    aspect_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parallel = de * aspect_e + dn * aspect_n
    de_par = parallel * aspect_e
    dn_par = parallel * aspect_n
    de_perp = de - de_par
    dn_perp = dn - dn_par

    uphill = parallel < 0
    downhill = parallel > 0
    scale = np.ones_like(parallel)
    scale[uphill] = settings.uphill_factor
    scale[downhill] = settings.downhill_factor

    de_out = de_par * scale + de_perp
    dn_out = dn_par * scale + dn_perp
    return de_out, dn_out


def _apply_land_mask(
    particles: Particles,
    grid: ProbabilityGrid,
    terrain: "TerrainContext",
    eastings: np.ndarray,
    northings: np.ndarray,
    v_e: np.ndarray,
    v_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = particle_cell_indices(grid, eastings, northings)
    on_water = ~terrain.is_land[rows, cols]
    if not np.any(on_water):
        return eastings, northings, v_e, v_n

    # Nudge water particles toward nearest land cell in grid
    land_rows, land_cols = np.where(terrain.is_land)
    if land_rows.size == 0:
        return eastings, northings, v_e, v_n

    res = grid.metadata.resolution_m
    half = (grid.rows * res) / 2.0
    origin_e = grid.crs.origin_e
    origin_n = grid.crs.origin_n

    for idx in np.where(on_water)[0]:
        r, c = rows[idx], cols[idx]
        dists = (land_rows - r) ** 2 + (land_cols - c) ** 2
        nearest = int(np.argmin(dists))
        lr, lc = int(land_rows[nearest]), int(land_cols[nearest])
        target_e = origin_e - half + (lc + 0.5) * res
        target_n = origin_n + half - (lr + 0.5) * res
        eastings[idx] = 0.85 * eastings[idx] + 0.15 * target_e
        northings[idx] = 0.85 * northings[idx] + 0.15 * target_n
        v_e[idx] *= 0.2
        v_n[idx] *= 0.2

    return eastings, northings, v_e, v_n


def _momentum_scales(dt: float, sigma_v: float, sigma_x: float) -> tuple[float, float, float, float]:
    """Return (alpha_dt, velocity_noise_std, position_noise_std, dt_eff) for step dt."""
    dt_eff = max(float(dt), 0.1)
    ref_dt = settings.momentum_reference_dt_sec
    tau = settings.momentum_tau_sec
    alpha_dt = float(np.exp(-dt_eff / tau))
    time_scale = np.sqrt(dt_eff / ref_dt)
    return alpha_dt, sigma_v * time_scale, sigma_x * time_scale, dt_eff


def predict_step(
    particles: Particles,
    env: EnvForcing,
    dt: float = 1.0,
    terrain: Optional["TerrainContext"] = None,
    grid: Optional[ProbabilityGrid] = None,
    terrain_grid: Optional[ProbabilityGrid] = None,
    layers: Optional[LayerFlags] = None,
) -> Particles:
    rng = np.random.default_rng()
    layer_flags = layers or LayerFlags()
    sample_grid = terrain_grid or grid
    sigma_v = settings.sigma_v
    sigma_x = settings.sigma_x

    if layer_flags.topography:
        sigma_v *= 0.7
        sigma_x *= 0.7
    if layer_flags.subject_injured:
        sigma_v *= 0.5
        sigma_x *= 0.5

    alpha_dt, sigma_v_eff, pos_noise_std, dt_eff = _momentum_scales(dt, sigma_v, sigma_x)

    n = particles.count
    wind_n = env.u_w + env.u_c if layer_flags.weather else 0.0
    wind_e = env.v_w + env.v_c if layer_flags.weather else 0.0

    v_n = alpha_dt * particles.v_n + (1.0 - alpha_dt) * wind_n + sigma_v_eff * rng.standard_normal(n)
    v_e = alpha_dt * particles.v_e + (1.0 - alpha_dt) * wind_e + sigma_v_eff * rng.standard_normal(n)

    if layer_flags.subject_injured:
        factor = settings.injured_velocity_factor
        v_n *= factor
        v_e *= factor

    if layer_flags.topography and terrain is not None and sample_grid is not None:
        rows, cols = particle_cell_indices(sample_grid, particles.eastings, particles.northings)
        slope = terrain.slope[rows, cols]
        aspect_e = terrain.aspect_e[rows, cols]
        aspect_n = terrain.aspect_n[rows, cols]
        beta = settings.terrain_beta
        v_e = v_e + beta * aspect_e * slope
        v_n = v_n + beta * aspect_n * slope

    if layer_flags.roads and terrain is not None and sample_grid is not None:
        rows, cols = particle_cell_indices(sample_grid, particles.eastings, particles.northings)
        prox = terrain.road_proximity[rows, cols]
        snap_threshold = np.exp(
            -settings.road_snap_radius_m / settings.road_proximity_decay_m
        )
        near_road = prox >= snap_threshold
        if np.any(near_road):
            te = terrain.road_tangent_e[rows, cols]
            tn = terrain.road_tangent_n[rows, cols]
            strength = settings.road_snap_strength * prox
            v_dot_t = v_e * te + v_n * tn
            v_e = np.where(near_road, v_e + strength * (v_dot_t * te - v_e), v_e)
            v_n = np.where(near_road, v_n + strength * (v_dot_t * tn - v_n), v_n)

    de = v_e * dt_eff + pos_noise_std * rng.standard_normal(n)
    dn = v_n * dt_eff + pos_noise_std * rng.standard_normal(n)

    if layer_flags.roads and terrain is not None and sample_grid is not None:
        rows, cols = particle_cell_indices(sample_grid, particles.eastings, particles.northings)
        prox = terrain.road_proximity[rows, cols]
        snap_threshold = np.exp(
            -settings.road_snap_radius_m / settings.road_proximity_decay_m
        )
        near_road = prox >= snap_threshold
        if np.any(near_road):
            te = terrain.road_tangent_e[rows, cols]
            tn = terrain.road_tangent_n[rows, cols]
            pull = settings.road_displacement_pull * prox * pos_noise_std
            de = np.where(near_road, de + pull * te, de)
            dn = np.where(near_road, dn + pull * tn, dn)

    if layer_flags.topography and terrain is not None and sample_grid is not None:
        rows, cols = particle_cell_indices(sample_grid, particles.eastings, particles.northings)
        aspect_e = terrain.aspect_e[rows, cols]
        aspect_n = terrain.aspect_n[rows, cols]
        de, dn = _apply_anisotropic_mobility(de, dn, aspect_e, aspect_n)

    eastings = particles.eastings + de
    northings = particles.northings + dn

    if layer_flags.topography and terrain is not None and sample_grid is not None:
        eastings, northings, v_e, v_n = _apply_land_mask(
            particles, sample_grid, terrain, eastings, northings, v_e, v_n
        )

    return Particles(eastings, northings, v_n, v_e, particles.weights.copy())


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
    if fade_cells <= 0:
        return grid
    rows, cols = grid.shape
    r_idx = np.arange(rows, dtype=np.float64)[:, None]
    c_idx = np.arange(cols, dtype=np.float64)[None, :]
    dist = np.minimum(
        np.minimum(r_idx, rows - 1 - r_idx),
        np.minimum(c_idx, cols - 1 - c_idx),
    )
    t = np.clip(dist / fade_cells, 0.0, 1.0)
    fade = t * t * (3.0 - 2.0 * t)
    out = grid * fade
    total = out.sum()
    if total > 0:
        out /= total
    return out


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
) -> np.ndarray:
    half_rows = (grid_rows * resolution_m) / 2.0
    half_cols = (grid_cols * resolution_m) / 2.0
    road_lambda = settings.road_kde_bonus

    grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    h = resolution_m * 0.4
    h_sq = h * h

    for i in range(particles.count):
        col_f = (particles.eastings[i] - (origin_e - half_cols)) / resolution_m
        row_f = ((origin_n + half_rows) - particles.northings[i]) / resolution_m
        w = particles.weights[i]

        radius_cells = 4
        row_min = max(0, int(np.floor(row_f - radius_cells)))
        row_max = min(grid_rows, int(np.ceil(row_f + radius_cells)) + 1)
        col_min = max(0, int(np.floor(col_f - radius_cells)))
        col_max = min(grid_cols, int(np.ceil(col_f + radius_cells)) + 1)

        for row in range(row_min, row_max):
            for col in range(col_min, col_max):
                dr = row - row_f
                dc = col - col_f
                dist_sq = (dr * resolution_m) ** 2 + (dc * resolution_m) ** 2
                kernel = np.exp(-0.5 * dist_sq / h_sq)
                road_factor = 1.0
                if roads_layer and road_proximity is not None:
                    road_factor = 1.0 + road_lambda * road_proximity[row, col]
                grid[row, col] += w * kernel * road_factor

    total = grid.sum()
    if total > 0:
        grid /= total

    fade = edge_fade_cells if edge_fade_cells is not None else settings.kde_edge_fade_cells
    return _apply_edge_fade(grid, fade)


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
    """Return top weighted particles as [lat, lon, weight] for WS payload."""
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
