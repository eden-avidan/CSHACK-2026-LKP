"""Discrete advection of probability mass along a 2D current vector."""

from __future__ import annotations

import numpy as np

from app.engine.flow_advection import advect_flow_cone


def current_advect_step(
    probabilities: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    is_water: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
    resolution_m: float,
) -> np.ndarray:
    """One cone-advection step on water cells (legacy entry point)."""
    return advect_flow_cone(
        probabilities,
        current_u,
        current_v,
        is_water.astype(bool, copy=False),
        dt_sec=dt_sec,
        strength=strength,
        resolution_m=resolution_m,
        steps=1,
    )


def advect_on_water(
    probabilities: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    is_water: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
    resolution_m: float,
    steps: int = 1,
) -> np.ndarray:
    if steps <= 0:
        return probabilities.astype(np.float64, copy=True)
    return advect_flow_cone(
        probabilities,
        current_u,
        current_v,
        is_water.astype(bool, copy=False),
        dt_sec=dt_sec,
        strength=strength,
        resolution_m=resolution_m,
        steps=steps,
    )
