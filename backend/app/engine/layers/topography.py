from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, SELF_INDEX, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class TopographyLayer(BaseProbabilityLayer):
    """
    Blocks probability flow into water cells and biases transitions toward
    cells with higher Tobler/Dijkstra reachability from the LKP.
    """

    layer_id = "topography"
    default_enabled = True
    default_weight = 0.65

    def transition_weights(
        self,
        ctx: TransitionContext,
        row: int,
        col: int,
        weight: float,
    ) -> np.ndarray:
        if weight <= 0:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        fields = ctx.node_fields
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i]:
                continue
            nr, nc = row + dr, col + dc
            if not fields.is_land[nr, nc]:
                adjustments[i] -= 10.0 * weight
                continue
            reach = fields.reachability[nr, nc]
            adjustments[i] += weight * reach * settings.terrain_beta

        # Penalize uphill transitions (row decreases = north = uphill in aspect_n)
        slope = fields.slope[row, col]
        if slope > 1e-6:
            for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
                if not valid[i] or (dr == 0 and dc == 0):
                    continue
                nr, nc = row + dr, col + dc
                uphill = fields.elevation[nr, nc] - fields.elevation[row, col]
                if uphill > 0:
                    grade = uphill / ctx.resolution_m
                    adjustments[i] -= weight * settings.uphill_factor * grade

        return adjustments
