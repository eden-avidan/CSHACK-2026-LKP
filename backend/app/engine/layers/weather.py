from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.flow_advection import advect_flow_cone
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class WeatherLayer(BaseProbabilityLayer):
    """
    Shifts probability downwind using per-cell mock wind and cone advection.
    """

    layer_id = "weather"
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
        fields = ctx.node_fields
        on_land = fields.is_land.astype(bool, copy=False)
        if not np.any(on_land):
            return p_in

        steps = max(1, settings.weather_advection_steps)
        advected = advect_flow_cone(
            p_in,
            fields.wind_u,
            fields.wind_v,
            on_land,
            dt_sec=ctx.dt_sec,
            strength=weight,
            resolution_m=ctx.resolution_m,
            steps=steps,
        )
        return (1.0 - weight) * p_in + weight * advected

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
        if not fields.is_land[row, col]:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        wind_e = float(fields.wind_u[row, col])
        wind_n = float(fields.wind_v[row, col])
        wind_speed = np.hypot(wind_e, wind_n)
        if wind_speed < 1e-6:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        dt_scale = np.sqrt(max(ctx.dt_sec, 1.0) / settings.momentum_reference_dt_sec)
        strength = weight * dt_scale * min(wind_speed / 5.0, 1.5)
        cone_cos = np.cos(np.radians(settings.flow_cone_half_angle_deg))

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr == 0 and dc == 0):
                continue
            neighbor_n = -float(dr)
            neighbor_e = float(dc)
            norm = np.hypot(neighbor_n, neighbor_e)
            alignment = (wind_e * neighbor_e + wind_n * neighbor_n) / (wind_speed * norm)
            if alignment >= cone_cos:
                cone_t = (alignment - cone_cos) / max(1.0 - cone_cos, 1e-12)
                adjustments[i] += strength * cone_t ** settings.flow_cone_sharpness

        return adjustments
