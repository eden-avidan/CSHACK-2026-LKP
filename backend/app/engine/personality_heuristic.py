from __future__ import annotations

import numpy as np

from app.models.personality import PersonalityProfile


def mobility_multiplier(profile: PersonalityProfile) -> float:
    """
    Combined mobility scalar from age, fitness, and injury.

    - Age: negatively correlated (older → smaller M).
    - Fitness 1–5: multiplier > 1 when high (1 → 0.85, 5 → 1.25).
    - Injured: multiplier < 1 (0.45) when True.
    """
    age_factor = max(0.35, min(1.20, 1.20 - (profile.age - 10) / 100.0))
    fitness_factor = 0.85 + 0.10 * (profile.fitness - 1)
    injured_factor = 0.45 if profile.injured else 1.0
    return age_factor * fitness_factor * injured_factor


def apply_mobility_scale(
    probabilities: np.ndarray,
    *,
    lkp_row: int,
    lkp_col: int,
    mobility: float,
) -> np.ndarray:
    """
    Scale probability mass by distance from LKP.

    ``scale = mobility ** (1 + dist_norm)`` with scale=1 at the LKP cell.
    M > 1 expands the fringe; M < 1 contracts it.
    """
    size = probabilities.shape[0]
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    dist = np.hypot(rows - lkp_row, cols - lkp_col)
    dist_norm = np.clip(dist / max(size * 0.5, 1.0), 0.0, 1.0)
    scale = np.power(mobility, 1.0 + dist_norm)
    scale[lkp_row, lkp_col] = 1.0
    return probabilities * scale
