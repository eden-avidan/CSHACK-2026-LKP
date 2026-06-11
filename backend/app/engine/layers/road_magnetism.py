from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.transition_context import TransitionContext


class RoadMagnetismLayer(BaseProbabilityLayer):
    """
    Boosts cell values near OSM roads using the per-cell road proximity field.
    """

    layer_id = "roads"
    default_enabled = False
    default_weight = 1.0

    def apply_field(
        self,
        ctx: TransitionContext,
        weight: float,
    ) -> np.ndarray:
        if weight <= 0:
            return ctx.probabilities.astype(np.float64, copy=True)

        p_in = ctx.probabilities.astype(np.float64, copy=True)
        prox = ctx.node_fields.road_proximity.astype(np.float64, copy=False)
        boost = 1.0 + weight * settings.road_kde_bonus * prox
        return p_in * boost
