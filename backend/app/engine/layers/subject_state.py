from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, SELF_INDEX, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class SubjectStateLayer(BaseProbabilityLayer):
    """
    Models reduced mobility (injured subject): probability spreads more slowly
    by retaining mass at the current cell and suppressing neighbor outflow.
    """

    layer_id = "subject_injured"
    default_enabled = False
    default_weight = 1.0

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
        retain = weight * (1.0 - settings.injured_velocity_factor)
        adjustments[SELF_INDEX] += retain

        valid = valid_neighbor_mask(ctx.size, row, col)
        for i in range(NEIGHBOR_COUNT):
            if i == SELF_INDEX or not valid[i]:
                continue
            adjustments[i] -= retain / max(1, valid.sum() - 1)

        return adjustments
