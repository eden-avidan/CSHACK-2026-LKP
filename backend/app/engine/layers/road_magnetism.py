from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, SELF_INDEX, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class RoadMagnetismLayer(BaseProbabilityLayer):
    """
    Pulls probability mass along road tangents — probability "bleeds" preferentially
    in the direction of nearby trails and highways.
    """

    layer_id = "roads"
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

        fields = ctx.node_fields
        prox = float(fields.road_proximity[row, col])
        snap_threshold = np.exp(-settings.road_snap_radius_m / settings.road_proximity_decay_m)
        if prox < snap_threshold:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        te = float(fields.road_tangent_e[row, col])
        tn = float(fields.road_tangent_n[row, col])
        if abs(te) < 1e-9 and abs(tn) < 1e-9:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)
        strength = settings.road_snap_strength * prox * weight

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr == 0 and dc == 0):
                continue
            # Grid: row↓ = south, col→ = east
            neighbor_n = -float(dr)
            neighbor_e = float(dc)
            alignment = tn * neighbor_n + te * neighbor_e
            if alignment > 0:
                adjustments[i] += strength * alignment

        return adjustments
