from __future__ import annotations

import numpy as np

from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.registry import get_active_layers
from app.engine.transition_context import TransitionContext
from app.models.layers import LayerFlags
from app.models.personality import PersonalityProfile
from app.services.particle_types import EnvForcing


class GridEngine:
    """
    Interactive heatmap engine: applies each active layer's ``apply_field``
    in registry order. No automatic normalization — cell values are raw.
    """

    def apply_layers(
        self,
        matrix: GridMatrix,
        layers: LayerFlags,
        *,
        dt_sec: float = 0.0,
        tick_count: int = 0,
        env: EnvForcing | None = None,
        personality: PersonalityProfile | None = None,
    ) -> np.ndarray:
        ctx = TransitionContext(
            matrix=matrix,
            node_fields=matrix.node_fields,
            probabilities=matrix.probabilities,
            dt_sec=dt_sec,
            tick_count=tick_count,
            env=env or EnvForcing(),
            size=matrix.size,
            resolution_m=matrix.resolution_m,
            sea_mode=layers.sea_drift,
            personality=personality,
        )
        active = get_active_layers(layers)
        probs = matrix.probabilities.astype(np.float64, copy=True)

        for layer, layer_weight in active:
            ctx = TransitionContext(
                matrix=ctx.matrix,
                node_fields=ctx.node_fields,
                probabilities=probs,
                dt_sec=ctx.dt_sec,
                tick_count=ctx.tick_count,
                env=ctx.env,
                size=ctx.size,
                resolution_m=ctx.resolution_m,
                sea_mode=ctx.sea_mode,
                personality=ctx.personality,
            )
            probs = layer.apply_field(ctx, layer_weight)

        self._apply_land_mask(probs, matrix.node_fields, sea_mode=layers.sea_drift)
        return probs

    def tick(
        self,
        matrix: GridMatrix,
        layers: LayerFlags,
        dt_sec: float,
        tick_count: int,
        env: EnvForcing | None = None,
        personality: PersonalityProfile | None = None,
    ) -> np.ndarray:
        return self.apply_layers(
            matrix,
            layers,
            dt_sec=dt_sec,
            tick_count=tick_count,
            env=env,
            personality=personality,
        )

    @staticmethod
    def _apply_land_mask(
        probs: np.ndarray, fields: NodeFields, sea_mode: bool = False
    ) -> None:
        if sea_mode:
            land = fields.is_land
            if np.any(land):
                probs[land] = 0.0
            return
        water = ~fields.is_land
        if np.any(water):
            probs[water] = 0.0
