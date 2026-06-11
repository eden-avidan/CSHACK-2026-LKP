from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.engine.transition_context import TransitionContext


class BaseProbabilityLayer(ABC):
    """
    Plugin interface for grid-matrix probability transitions.

    Each layer defines ONLY the next-step transition function: given a source
    cell and its neighborhood, return relative outflow weights to adjacent
    nodes (including self). The GridEngine aggregates all active layers.
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

    @abstractmethod
    def transition_weights(
        self,
        ctx: TransitionContext,
        row: int,
        col: int,
        weight: float,
    ) -> np.ndarray:
        """
        Return a length-9 vector aligned with NEIGHBOR_OFFSETS (index 4 = self).

        Values are *additive adjustments* to the baseline isotropic transition.
        Negative values suppress flow; positive values attract mass.
        """
