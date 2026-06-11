from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.geospatial.grid import grid_extent_m, particle_cell_indices
from app.layers.base import LayerConfig, LayerContext, PredictState


def _apply_land_mask(particles, grid, terrain, eastings, northings, v_e, v_n):
    rows, cols = particle_cell_indices(grid, eastings, northings)
    on_water = ~terrain.is_land[rows, cols]
    if not np.any(on_water):
        return eastings, northings, v_e, v_n

    land_rows, land_cols = np.where(terrain.is_land)
    if land_rows.size == 0:
        return eastings, northings, v_e, v_n

    res = grid.metadata.resolution_m
    west, _east, _south, north = grid_extent_m(grid.rows, res)
    origin_e = grid.crs.origin_e
    origin_n = grid.crs.origin_n

    for idx in np.where(on_water)[0]:
        r, c = rows[idx], cols[idx]
        dists = (land_rows - r) ** 2 + (land_cols - c) ** 2
        nearest = int(np.argmin(dists))
        lr, lc = int(land_rows[nearest]), int(land_cols[nearest])
        target_e = origin_e - west + (lc + 0.5) * res
        target_n = origin_n + north - (lr + 0.5) * res
        eastings[idx] = 0.85 * eastings[idx] + 0.15 * target_e
        northings[idx] = 0.85 * northings[idx] + 0.15 * target_n
        v_e[idx] *= 0.2
        v_n[idx] *= 0.2

    return eastings, northings, v_e, v_n


@dataclass
class TopographyLayer:
    """
    Tobler + Dijkstra reachability is applied at KDE time in mission_store.
    This layer only blocks water cells during particle propagation.
    """

    config: LayerConfig = field(
        default_factory=lambda: LayerConfig(id="topography", default_enabled=True, default_weight=0.65)
    )

    def adjust_sigmas(self, sigma_v: float, sigma_x: float, weight: float) -> tuple[float, float]:
        return sigma_v, sigma_x

    def apply_velocity(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def apply_displacement(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def apply_post_step(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        if weight <= 0 or ctx.terrain is None:
            return state
        sample_grid = ctx.terrain_grid or ctx.grid
        if sample_grid is None:
            return state
        state.eastings, state.northings, state.v_e, state.v_n = _apply_land_mask(
            state.particles,
            sample_grid,
            ctx.terrain,
            state.eastings,
            state.northings,
            state.v_e,
            state.v_n,
        )
        return state

    def kde_road_factor(self, row: int, col: int, ctx: LayerContext, weight: float) -> float:
        return 1.0


topography_layer = TopographyLayer()
