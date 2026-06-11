from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.config import settings
from app.geospatial.grid import particle_cell_indices
from app.layers.base import LayerConfig, LayerContext, PredictState


@dataclass
class RoadsLayer:
    config: LayerConfig = field(
        default_factory=lambda: LayerConfig(id="roads", default_enabled=False, default_weight=1.0)
    )

    def adjust_sigmas(self, sigma_v: float, sigma_x: float, weight: float) -> tuple[float, float]:
        return sigma_v, sigma_x

    def apply_velocity(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        if weight <= 0 or ctx.terrain is None:
            return state
        sample_grid = ctx.terrain_grid or ctx.grid
        if sample_grid is None:
            return state
        rows, cols = particle_cell_indices(sample_grid, state.particles.eastings, state.particles.northings)
        prox = ctx.terrain.road_proximity[rows, cols]
        snap_threshold = np.exp(-settings.road_snap_radius_m / settings.road_proximity_decay_m)
        near_road = prox >= snap_threshold
        if not np.any(near_road):
            return state
        te = ctx.terrain.road_tangent_e[rows, cols]
        tn = ctx.terrain.road_tangent_n[rows, cols]
        strength = settings.road_snap_strength * prox * weight
        v_dot_t = state.v_e * te + state.v_n * tn
        state.v_e = np.where(near_road, state.v_e + strength * (v_dot_t * te - state.v_e), state.v_e)
        state.v_n = np.where(near_road, state.v_n + strength * (v_dot_t * tn - state.v_n), state.v_n)
        return state

    def apply_displacement(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        if weight <= 0 or ctx.terrain is None:
            return state
        sample_grid = ctx.terrain_grid or ctx.grid
        if sample_grid is None:
            return state
        rows, cols = particle_cell_indices(sample_grid, state.particles.eastings, state.particles.northings)
        prox = ctx.terrain.road_proximity[rows, cols]
        snap_threshold = np.exp(-settings.road_snap_radius_m / settings.road_proximity_decay_m)
        near_road = prox >= snap_threshold
        if not np.any(near_road):
            return state
        te = ctx.terrain.road_tangent_e[rows, cols]
        tn = ctx.terrain.road_tangent_n[rows, cols]
        pull = settings.road_displacement_pull * prox * ctx.pos_noise_std * weight
        state.de = np.where(near_road, state.de + pull * te, state.de)
        state.dn = np.where(near_road, state.dn + pull * tn, state.dn)
        return state

    def apply_post_step(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def kde_road_factor(self, row: int, col: int, ctx: LayerContext, weight: float) -> float:
        if weight <= 0 or ctx.road_proximity is None:
            return 1.0
        return 1.0 + settings.road_kde_bonus * weight * ctx.road_proximity[row, col]


roads_layer = RoadsLayer()
