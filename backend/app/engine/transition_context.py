from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.engine.grid_matrix import GridMatrix, NodeFields
from app.services.particle_types import EnvForcing


@dataclass
class TransitionContext:
    """Read-only snapshot passed to every layer during a tick."""

    matrix: GridMatrix
    node_fields: NodeFields
    probabilities: np.ndarray
    dt_sec: float
    tick_count: int
    env: EnvForcing
    size: int
    resolution_m: float
    sea_mode: bool = False

    def neighbor_indices(self, row: int, col: int, dr: int, dc: int) -> tuple[int, int] | None:
        nr, nc = row + dr, col + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            return nr, nc
        return None
