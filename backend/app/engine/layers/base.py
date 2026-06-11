from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.engine.transition_context import TransitionContext


class BaseProbabilityLayer(ABC):
    """
    Plugin interface for grid-matrix probability layers.

    Interactive pipeline: each layer implements ``apply_field`` to transform
    the full probability matrix using per-cell NodeFields. Values are not
    required to sum to 1.
    """

    @property
    @abstractmethod
    def layer_id(self) -> str:
        """Stable id matching LayerFlags field (e.g. 'roads', 'topography')."""

    @property
    @abstractmethod
    def default_enabled(self) -> bool: ...

    @property
    def default_weight(self) -> float:
        return 1.0

    def apply_field(
        self,
        ctx: TransitionContext,
        weight: float,
    ) -> np.ndarray:
        """Transform the full matrix. Default: pass-through (no change)."""
        return ctx.probabilities.astype(np.float64, copy=True)

    def transition_weights(
        self,
        ctx: TransitionContext,
        row: int,
        col: int,
        weight: float,
    ) -> np.ndarray:
        """
        Legacy neighbor outflow hook (deprecated — use apply_field).

        Kept for tests and migration reference; not called by GridEngine.
        """
        from app.engine.neighbors import NEIGHBOR_COUNT

        return np.zeros(NEIGHBOR_COUNT, dtype=np.float64)
