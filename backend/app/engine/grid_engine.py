from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.grid_matrix import GridMatrix, NodeFields
from app.engine.layers.registry import get_active_layers
from app.engine.neighbors import (
    NEIGHBOR_COUNT,
    NEIGHBOR_OFFSETS,
    SELF_INDEX,
    normalize_weights,
    valid_neighbor_mask,
)
from app.engine.transition_context import TransitionContext
from app.models.layers import LayerFlags
from app.services.particle_types import EnvForcing


class GridEngine:
    """
    Core heatmap engine: iterates the A×A matrix each tick, aggregates
    transition weights from all active layers, and produces a normalized
    probability field (sum = 1.0).
    """

    def __init__(self, base_outflow: float | None = None) -> None:
        self.base_outflow = base_outflow or settings.grid_base_outflow

    def tick(
        self,
        matrix: GridMatrix,
        layers: LayerFlags,
        dt_sec: float,
        tick_count: int,
        env: EnvForcing | None = None,
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
        )
        active = get_active_layers(layers)
        new_probs = np.zeros_like(matrix.probabilities)
        size = matrix.size
        source = matrix.probabilities

        for row in range(size):
            for col in range(size):
                mass = source[row, col]
                if mass <= 0:
                    continue
                outflow = self._aggregate_transitions(ctx, row, col, active)
                for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        new_probs[nr, nc] += mass * outflow[i]

        self._apply_land_mask(new_probs, matrix.node_fields)
        return self._normalize(new_probs)

    def _aggregate_transitions(
        self,
        ctx: TransitionContext,
        row: int,
        col: int,
        active: list,
    ) -> np.ndarray:
        valid = valid_neighbor_mask(ctx.size, row, col)
        baseline = np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
        neighbor_slots = max(1, int(valid.sum()) - (1 if valid[SELF_INDEX] else 0))
        leak = self.base_outflow / neighbor_slots

        for i in range(NEIGHBOR_COUNT):
            if not valid[i]:
                continue
            if i == SELF_INDEX:
                baseline[i] = 1.0 - self.base_outflow
            else:
                baseline[i] = leak

        combined = baseline.copy()
        for layer, weight in active:
            combined += layer.transition_weights(ctx, row, col, weight)

        combined = np.where(valid, combined, 0.0)
        combined = np.clip(combined, 0.0, None)
        return normalize_weights(combined)

    @staticmethod
    def _apply_land_mask(probs: np.ndarray, fields: NodeFields) -> None:
        water = ~fields.is_land
        if np.any(water):
            probs[water] = 0.0

    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        total = float(probs.sum())
        if total <= 0:
            return probs
        return probs / total
