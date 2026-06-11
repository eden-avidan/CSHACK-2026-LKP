from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class WeatherLayer(BaseProbabilityLayer):
    """
    Shifts probability downwind — mass bleeds preferentially in the wind direction.
    """

    layer_id = "weather"
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

        wind_e = float(ctx.env.u_w)
        wind_n = float(ctx.env.v_w)
        wind_speed = np.hypot(wind_e, wind_n)
        if wind_speed < 1e-6:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        dt_scale = np.sqrt(max(ctx.dt_sec, 1.0) / settings.momentum_reference_dt_sec)
        strength = weight * dt_scale * min(wind_speed / 5.0, 1.5)

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr == 0 and dc == 0):
                continue
            neighbor_n = -float(dr)
            neighbor_e = float(dc)
            alignment = (wind_e * neighbor_e + wind_n * neighbor_n) / wind_speed
            if alignment > 0:
                adjustments[i] += strength * alignment

        return adjustments
