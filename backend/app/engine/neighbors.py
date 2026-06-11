from __future__ import annotations

import numpy as np

# 8-connected neighborhood + self (index 4).
# Grid rows increase south; cols increase east.
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SELF_INDEX = 4
NEIGHBOR_COUNT = len(NEIGHBOR_OFFSETS)


def valid_neighbor_mask(size: int, row: int, col: int) -> np.ndarray:
    """Boolean mask aligned with NEIGHBOR_OFFSETS."""
    mask = np.zeros(NEIGHBOR_COUNT, dtype=bool)
    for i, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
        nr, nc = row + dr, col + dc
        mask[i] = 0 <= nr < size and 0 <= nc < size
    return mask


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        out = np.zeros_like(weights)
        out[SELF_INDEX] = 1.0
        return out
    return weights / total
