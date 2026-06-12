"""Directional cone advection — mass moves down-flow, not isotropically."""

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


def flow_cone_advect_step(
    probabilities: np.ndarray,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    active: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
    resolution_m: float,
) -> np.ndarray:
    """
    Advect probability along a local flow vector using a downstream cone.

    Mass leaves the source cell only toward neighbors inside the cone aligned
    with [u_east, v_north]; lateral bleed outside the cone is suppressed.
    """
    p = probabilities.astype(np.float64, copy=False)
    mask = active.astype(bool, copy=False)
    u = flow_u.astype(np.float64, copy=False)
    v = flow_v.astype(np.float64, copy=False)

    speed = np.hypot(u, v)
    moving = mask & (speed > 1e-9)
    if not np.any(moving):
        return p.astype(np.float64, copy=True)

    dt_scale = np.sqrt(max(dt_sec, 1.0) / max(settings.momentum_reference_dt_sec, 1.0))
    bias = strength * dt_scale * settings.flow_cone_strength
    cone_cos = np.cos(np.radians(settings.flow_cone_half_angle_deg))

    cells_per_sec = speed / max(resolution_m, 1e-6)
    emit_frac = np.clip(bias * cells_per_sec * dt_sec, 0.0, settings.flow_cone_max_emit_frac)
    emit = np.where(moving, emit_frac, 0.0)

    p_new = p * (1.0 - emit)
    outflow = p * emit

    weight_sum = np.zeros(p.shape, dtype=np.float64)
    neighbor_weights: list[tuple[int, int, np.ndarray]] = []

    for dr, dc in _EIGHT:
        de = float(dc)
        dn = -float(dr)
        norm = np.hypot(de, dn)
        alignment = (u * de + v * dn) / np.maximum(speed * norm, 1e-12)
        in_cone = alignment >= cone_cos
        cone_t = np.where(
            in_cone & moving,
            (alignment - cone_cos) / max(1.0 - cone_cos, 1e-12),
            0.0,
        )
        w = np.power(cone_t, settings.flow_cone_sharpness)
        weight_sum += w
        neighbor_weights.append((dr, dc, w))

    weight_sum = np.maximum(weight_sum, 1e-12)
    for dr, dc, w in neighbor_weights:
        flow = outflow * (w / weight_sum)
        p_new += _push_to_neighbor(flow, dr, dc)

    total_in = p.sum()
    total_out = p_new.sum()
    if total_out > 1e-15 and total_in > 0:
        p_new *= total_in / total_out

    return p_new


def advect_flow_cone(
    probabilities: np.ndarray,
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    active: np.ndarray,
    *,
    dt_sec: float,
    strength: float,
    resolution_m: float,
    steps: int = 1,
) -> np.ndarray:
    if steps <= 0:
        return probabilities.astype(np.float64, copy=True)
    p = probabilities.astype(np.float64, copy=True)
    for _ in range(steps):
        p = flow_cone_advect_step(
            p,
            flow_u,
            flow_v,
            active,
            dt_sec=dt_sec,
            strength=strength,
            resolution_m=resolution_m,
        )
    return p
