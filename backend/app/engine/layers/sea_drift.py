from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.layers.base import BaseProbabilityLayer
from app.engine.neighbors import NEIGHBOR_COUNT, NEIGHBOR_OFFSETS, valid_neighbor_mask
from app.engine.transition_context import TransitionContext


class SeaDriftLayer(BaseProbabilityLayer):
    """
    Models a person/object adrift at sea under a constant drift velocity
    (surface current + leeway). Probability mass is advected in the configured
    drift direction each tick; the engine's baseline diffusion supplies the
    uncertainty growth around that drift.

    Enabled automatically when the LKP falls on water (see MissionStore.create).
    Unlike the weather layer, the drift here is a fixed configured vector, not
    the per-tick `env` wind, so the cloud translates steadily over time.
    """

    layer_id = "sea_drift"
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

        # Compass heading -> east/north unit components (0=N, 90=E, clockwise).
        heading = np.radians(settings.sea_drift_heading_deg)
        drift_e = np.sin(heading)
        drift_n = np.cos(heading)

        # Drift distance this tick, in cells; more drift -> stronger bias.
        drift_cells = (
            settings.sea_drift_speed_mps * max(ctx.dt_sec, 1.0) / ctx.resolution_m
        )
        strength = weight * settings.sea_drift_strength * min(drift_cells, 1.5)
        if strength <= 0:
            return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)

        adjustments = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        valid = valid_neighbor_mask(ctx.size, row, col)

        for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
            if not valid[i] or (dr == 0 and dc == 0):
                continue
            # Grid: row down = south, col right = east.
            neighbor_n = -float(dr)
            neighbor_e = float(dc)
            norm = np.hypot(neighbor_n, neighbor_e)
            if norm < 1e-9:
                continue
            alignment = (drift_e * neighbor_e + drift_n * neighbor_n) / norm
            if alignment > 0:
                adjustments[i] += strength * alignment

        return adjustments
