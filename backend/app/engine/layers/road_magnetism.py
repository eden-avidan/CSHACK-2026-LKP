from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.cost_surface_diffusion import (
    build_terrain_cost_map,
    cost_surface_diffusion,
    diffusion_steps_for_tick,
    l2_neighbor_distance,
    transition_weight_l2_cost,
)
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class RoadMagnetismLayer(BaseProbabilityLayer):
    """
    Continuous cost-surface diffusion along trails.

    Blends L2 (Euclidean) neighbor distance with terrain friction so probability
    forms road "fingers" with soft ambient bleed into forest — not hard walls.
    """

    layer_id = "roads"
    default_enabled = False
    default_weight = 0.68

    def apply_field(
        self,
        ctx: TransitionContext,
        weight: float,
    ) -> np.ndarray:
        if weight <= 0:
            return ctx.probabilities.astype(np.float64, copy=True)

        p_in = ctx.probabilities.astype(np.float64, copy=True)
        fields = ctx.node_fields
        matrix = ctx.matrix
        is_road = fields.is_land & fields.is_road.astype(bool)
        lkp_r, lkp_c = matrix.lkp_row, matrix.lkp_col
        anchor = float(p_in[lkp_r, lkp_c])

        terrain_cost = build_terrain_cost_map(
            is_road,
            fields.slope,
            fields.is_land,
        )
        steps = diffusion_steps_for_tick(
            ctx.dt_sec, weight, tick_count=ctx.tick_count
        )
        diffused = cost_surface_diffusion(
            p_in,
            terrain_cost,
            is_road,
            steps=steps,
        )

        prox = fields.road_proximity.astype(np.float64, copy=False)
        boost = 1.0 + weight * settings.road_kde_bonus * prox
        target = diffused * boost

        # Keep the LKP cell anchored on the pin so the first frame stays centered.
        target[lkp_r, lkp_c] = max(
            float(target[lkp_r, lkp_c]),
            anchor * float(boost[lkp_r, lkp_c]),
        )

        return (1.0 - weight) * p_in + weight * target

    def transition_weights(
        self,
        ctx: TransitionContext,
        row: int,
        col: int,
        weight: float,
    ) -> np.ndarray:
        """Per-neighbor outflow weights (legacy / test hook)."""
        if weight <= 0:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        fields = ctx.node_fields
        is_road = fields.is_land & fields.is_road.astype(bool)
        terrain_cost = build_terrain_cost_map(
            is_road,
            fields.slope,
            fields.is_land,
        )
        road_a = bool(is_road[row, col])
        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr, dc) == (0, 0):
                continue
            nr, nc = row + dr, col + dc
            road_b = bool(is_road[nr, nc])
            l2 = l2_neighbor_distance(dr, dc)
            cost_b = np.array([[terrain_cost[nr, nc]]], dtype=np.float64)
            road_a_arr = np.array([[road_a]], dtype=bool)
            road_b_arr = np.array([[road_b]], dtype=bool)
            adjustments[i] = weight * float(
                transition_weight_l2_cost(l2, cost_b, road_a_arr, road_b_arr)[0, 0]
            )

        return adjustments
