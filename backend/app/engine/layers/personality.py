from __future__ import annotations

import numpy as np

from app.engine.layers.base import BaseProbabilityLayer
from app.engine.personality_heuristic import apply_mobility_scale, mobility_multiplier
from app.engine.transition_context import TransitionContext


class PersonalityLayer(BaseProbabilityLayer):
    """
    Adjusts spread using subject age, fitness, and injury via a mobility heuristic.
    """

    layer_id = "personality"
    default_enabled = False
    default_weight = 1.0

    def apply_field(
        self,
        ctx: TransitionContext,
        weight: float,
    ) -> np.ndarray:
        if weight <= 0 or ctx.personality is None:
            return ctx.probabilities.astype(np.float64, copy=True)

        p_in = ctx.probabilities.astype(np.float64, copy=True)
        mobility = mobility_multiplier(ctx.personality)
        matrix = ctx.matrix
        target = apply_mobility_scale(
            p_in,
            lkp_row=matrix.lkp_row,
            lkp_col=matrix.lkp_col,
            mobility=mobility,
        )
        return (1.0 - weight) * p_in + weight * target
