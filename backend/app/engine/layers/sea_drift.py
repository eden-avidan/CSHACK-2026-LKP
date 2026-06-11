from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, valid_neighbor_mask
from app.engine.transition_context import TransitionContext
from app.engine.water_advection import advect_on_water


class SeaDriftLayer(BaseProbabilityLayer):
    """
    Water drift using a live Open-Meteo marine current (cached at mission create).

    Probability on water cells is advected along the fetched [u_east, v_north]
    vector; land cells are unchanged here (zeroed by the engine land mask).
    """

    layer_id = "sea_drift"
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
        is_water = ~fields.is_land.astype(bool, copy=False)

        if not np.any(is_water):
            return p_in

        steps = max(1, settings.marine_drift_steps)
        advected = advect_on_water(
            p_in,
            fields.current_u,
            fields.current_v,
            is_water,
            dt_sec=ctx.dt_sec,
            strength=weight,
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
        """Per-neighbor drift bias using cached current_u/v (legacy hook)."""
        if weight <= 0:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        fields = ctx.node_fields
        if fields.is_land[row, col]:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        drift_e = float(fields.current_u[row, col])
        drift_n = float(fields.current_v[row, col])
        drift_speed = np.hypot(drift_e, drift_n)
        if drift_speed < 1e-6:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        drift_cells = drift_speed * max(ctx.dt_sec, 1.0) / ctx.resolution_m
        strength = (
            weight
            * settings.sea_drift_strength
            * settings.marine_drift_advection_strength
            * min(drift_cells, 1.5)
        )
        if strength <= 0:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr == 0 and dc == 0):
                continue
            neighbor_n = -float(dr)
            neighbor_e = float(dc)
            norm = np.hypot(neighbor_n, neighbor_e)
            if norm < 1e-9:
                continue
            alignment = (drift_e * neighbor_e + drift_n * neighbor_n) / (drift_speed * norm)
            if alignment > 0:
                adjustments[i] += strength * alignment

        return adjustments
