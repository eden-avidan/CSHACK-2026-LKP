from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.transition_context import TransitionContext


class TopographyLayer(BaseProbabilityLayer):
    """
    Projects probability along the Tobler/Dijkstra reach field from the LKP.

    Starting from an impulse at the pin, this spreads mass to land cells
    proportional to ``reachability_score`` (1 at LKP, 0 beyond the horizon).
    """

    layer_id = "topography"
    default_enabled = True
    default_weight = 1.0

    def apply_field(
        self,
        ctx: TransitionContext,
        weight: float,
    ) -> np.ndarray:
        if weight <= 0:
            return ctx.probabilities.astype(np.float64, copy=True)

        fields = ctx.node_fields
        matrix = ctx.matrix
        p_in = ctx.probabilities.astype(np.float64, copy=True)

        score = fields.reachability_score.astype(np.float64, copy=True)
        score[~fields.is_land] = 0.0

        slope_deg = np.degrees(fields.slope)
        steep = slope_deg >= settings.topo_steep_threshold_deg
        score[steep] *= settings.topo_steep_weight

        anchor = float(p_in[matrix.lkp_row, matrix.lkp_col])
        target = anchor * score
        target[~fields.is_land] = 0.0

        out = (1.0 - weight) * p_in + weight * target
        out[~fields.is_land] = 0.0
        return out
