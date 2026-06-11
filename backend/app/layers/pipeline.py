from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from app.core.config import settings
from app.layers.base import LayerContext, PredictState
from app.layers.registry import LAYER_REGISTRY, get_layer_weights, is_layer_enabled
from app.layers.roads import roads_layer
from app.layers.weather import weather_layer
from app.models.layers import LayerFlags
from app.services.particle_types import (
    EnvForcing,
    Particles,
    apply_edge_fade,
    momentum_scales,
)

if TYPE_CHECKING:
    from app.geospatial.grid import ProbabilityGrid
    from app.services.env_ingestion import TerrainContext


def run_predict_pipeline(
    particles: Particles,
    env: EnvForcing,
    dt: float,
    terrain: Optional["TerrainContext"],
    grid: Optional["ProbabilityGrid"],
    terrain_grid: Optional["ProbabilityGrid"],
    layers: LayerFlags,
) -> Particles:
    rng = np.random.default_rng()
    sigma_v = settings.sigma_v
    sigma_x = settings.sigma_x
    weights = get_layer_weights(layers)

    for layer in LAYER_REGISTRY:
        w = weights[layer.config.id]
        if w > 0:
            sigma_v, sigma_x = layer.adjust_sigmas(sigma_v, sigma_x, w)

    alpha_dt, sigma_v_eff, pos_noise_std, dt_eff = momentum_scales(dt, sigma_v, sigma_x)

    ctx = LayerContext(
        terrain=terrain,
        grid=grid,
        terrain_grid=terrain_grid,
        env=env,
        dt=dt,
        dt_eff=dt_eff,
        pos_noise_std=pos_noise_std,
        rng=rng,
    )

    wind_n, wind_e = weather_layer.wind_forcing(ctx, weights["weather"])

    n = particles.count
    v_n = alpha_dt * particles.v_n + (1.0 - alpha_dt) * wind_n + sigma_v_eff * rng.standard_normal(n)
    v_e = alpha_dt * particles.v_e + (1.0 - alpha_dt) * wind_e + sigma_v_eff * rng.standard_normal(n)

    state = PredictState(
        particles=particles,
        v_n=v_n,
        v_e=v_e,
        de=np.zeros(n),
        dn=np.zeros(n),
        eastings=particles.eastings.copy(),
        northings=particles.northings.copy(),
        sigma_v=sigma_v,
        sigma_x=sigma_x,
    )

    for layer in LAYER_REGISTRY:
        w = weights[layer.config.id]
        if w > 0:
            state = layer.apply_velocity(state, ctx, w)

    state.de = state.v_e * dt_eff + pos_noise_std * rng.standard_normal(n)
    state.dn = state.v_n * dt_eff + pos_noise_std * rng.standard_normal(n)

    for layer in LAYER_REGISTRY:
        w = weights[layer.config.id]
        if w > 0:
            state = layer.apply_displacement(state, ctx, w)

    state.eastings = particles.eastings + state.de
    state.northings = particles.northings + state.dn

    for layer in LAYER_REGISTRY:
        w = weights[layer.config.id]
        if w > 0:
            state = layer.apply_post_step(state, ctx, w)

    return Particles(state.eastings, state.northings, state.v_n, state.v_e, particles.weights.copy())


def run_kde_pipeline(
    particles: Particles,
    grid_rows: int,
    grid_cols: int,
    resolution_m: float,
    origin_e: float,
    origin_n: float,
    road_proximity: Optional[np.ndarray],
    layers: LayerFlags,
    edge_fade_cells: Optional[int] = None,
) -> np.ndarray:
    half_rows = (grid_rows * resolution_m) / 2.0
    half_cols = (grid_cols * resolution_m) / 2.0
    weights = get_layer_weights(layers)

    ctx = LayerContext(road_proximity=road_proximity)

    grid = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    h = resolution_m * settings.kde_bandwidth_factor
    h_sq = h * h
    radius_cells = max(4, int(np.ceil(3.0 * h / resolution_m)))

    for i in range(particles.count):
        col_f = (particles.eastings[i] - (origin_e - half_cols)) / resolution_m
        row_f = ((origin_n + half_rows) - particles.northings[i]) / resolution_m
        w = particles.weights[i]
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
                if is_layer_enabled(layers, "roads") and road_proximity is not None:
                    road_factor = roads_layer.kde_road_factor(row, col, ctx, weights["roads"])
                grid[row, col] += w * kernel * road_factor

    total = grid.sum()
    if total > 0:
        grid /= total

    fade = edge_fade_cells if edge_fade_cells is not None else settings.kde_edge_fade_cells
    if fade > 0:
        grid = apply_edge_fade(grid, fade)
    return grid
