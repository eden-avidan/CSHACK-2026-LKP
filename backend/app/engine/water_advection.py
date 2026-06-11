"""Discrete advection of probability mass along a 2D current vector."""

from __future__ import annotations

import numpy as np

from app.core.config import settings

_EIGHT: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _push_to_neighbor(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    size = arr.shape[0]
    out = np.zeros_like(arr)
    dst_r = slice(max(0, dr), size + min(0, dr))
    src_r = slice(max(0, -dr), size + min(0, -dr))
    dst_c = slice(max(0, dc), size + min(0, dc))
    src_c = slice(max(0, -dc), size + min(0, -dc))
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


def current_advect_step(
    probabilities: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    is_water: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
) -> np.ndarray:
    """
    Skew outflow on water cells toward the local [u_east, v_north] vector.

    Land cells are unchanged. Neighbors aligned with the current receive
    higher transition weight.
    """
    p = probabilities.astype(np.float64, copy=False)
    water = is_water.astype(bool, copy=False)
    u = current_u.astype(np.float64, copy=False)
    v = current_v.astype(np.float64, copy=False)

    speed = np.hypot(u, v)
    active = water & (speed > 1e-9)
    if not np.any(active):
        return p.astype(np.float64, copy=True)

    dt_scale = np.sqrt(max(dt_sec, 1.0) / max(settings.momentum_reference_dt_sec, 1.0))
    bias_strength = strength * dt_scale * settings.marine_drift_advection_strength

    self_w = settings.marine_drift_self_weight
    weight_sum = np.full(p.shape, self_w, dtype=np.float64)
    neighbor_weights: list[tuple[int, int, np.ndarray]] = []

    for dr, dc in _EIGHT:
        de = float(dc)
        dn = -float(dr)
        norm = np.hypot(de, dn)
        alignment = (u * de + v * dn) / np.maximum(speed * norm, 1e-12)
        alignment = np.maximum(alignment, 0.0)
        w = np.where(active, 1.0 + bias_strength * alignment, 0.0)
        weight_sum += w
        neighbor_weights.append((dr, dc, w))

    weight_sum = np.maximum(weight_sum, 1e-12)
    emit = active.astype(np.float64)

    p_new = p.copy()
    p_new += np.where(active, p * (self_w / weight_sum - 1.0), 0.0)
    for dr, dc, w in neighbor_weights:
        flow = p * emit * (w / weight_sum)
        p_new += _push_to_neighbor(flow, dr, dc)

    total_in = p.sum()
    total_out = p_new.sum()
    if total_out > 1e-15 and total_in > 0:
        p_new *= total_in / total_out

    return p_new


def advect_on_water(
    probabilities: np.ndarray,
    current_u: np.ndarray,
    current_v: np.ndarray,
    is_water: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
    steps: int = 1,
) -> np.ndarray:
    if steps <= 0:
        return probabilities.astype(np.float64, copy=True)
    p = probabilities.astype(np.float64, copy=True)
    for _ in range(steps):
        p = current_advect_step(p, current_u, current_v, is_water, dt_sec=dt_sec, strength=strength)
    return p
