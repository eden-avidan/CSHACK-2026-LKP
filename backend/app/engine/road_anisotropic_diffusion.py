"""Cost-weighted anisotropic diffusion along the road network."""

from __future__ import annotations

import numpy as np

from app.core.config import settings

# 8-connected neighbors (no self).
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


def pairwise_transition_weight(road_a: bool, road_b: bool) -> float:
    """Transition weight from node A to neighbor B based on road class."""
    if road_a and road_b:
        return settings.road_to_road_transition
    if road_a and not road_b:
        return settings.road_to_offroad_transition
    if not road_a and road_b:
        return settings.offroad_to_road_transition
    return settings.offroad_to_offroad_transition


def _neighbor_field(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """At (r,c), value of ``arr`` at neighbor (r+dr, c+dc); zero if out of bounds."""
    size = arr.shape[0]
    out = np.zeros_like(arr)
    src_r = slice(max(0, dr), size + min(0, dr))
    dst_r = slice(max(0, -dr), size + min(0, -dr))
    src_c = slice(max(0, dc), size + min(0, dc))
    dst_c = slice(max(0, -dc), size + min(0, -dc))
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


def _push_to_neighbor(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Place ``arr[r,c]`` into output position (r+dr, c+dc)."""
    size = arr.shape[0]
    out = np.zeros_like(arr)
    dst_r = slice(max(0, dr), size + min(0, dr))
    src_r = slice(max(0, -dr), size + min(0, -dr))
    dst_c = slice(max(0, dc), size + min(0, dc))
    src_c = slice(max(0, -dc), size + min(0, -dc))
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


def _vectorized_pairwise_weight(
    road_a: np.ndarray, road_b: np.ndarray
) -> np.ndarray:
    w = np.full(road_a.shape, settings.offroad_to_offroad_transition, dtype=np.float64)
    w[road_a & road_b] = settings.road_to_road_transition
    w[road_a & ~road_b] = settings.road_to_offroad_transition
    w[~road_a & road_b] = settings.offroad_to_road_transition
    return w


def anisotropic_diffuse_step(probabilities: np.ndarray, is_road: np.ndarray) -> np.ndarray:
    """
    One cost-weighted diffusion step.

    Outflow from each cell is split across neighbors by pairwise road transition
    weights; a small self-retention term keeps mass locally stable.
    """
    p = probabilities.astype(np.float64, copy=False)
    road = is_road.astype(bool, copy=False)
    self_w = settings.road_diffusion_self_weight
    weight_sum = np.full(p.shape, self_w, dtype=np.float64)

    neighbor_weights: list[tuple[int, int, np.ndarray]] = []
    for dr, dc in _EIGHT:
        n_road = _neighbor_field(road, dr, dc)
        w = _vectorized_pairwise_weight(road, n_road)
        weight_sum += w
        neighbor_weights.append((dr, dc, w))

    weight_sum = np.maximum(weight_sum, 1e-12)
    p_new = p * (self_w / weight_sum)
    for dr, dc, w in neighbor_weights:
        flow = p * (w / weight_sum)
        p_new += _push_to_neighbor(flow, dr, dc)
    return p_new


def anisotropic_road_diffusion(
    probabilities: np.ndarray,
    is_road: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Run multiple anisotropic diffusion steps so mass winds along trails."""
    if steps <= 0:
        return probabilities.astype(np.float64, copy=True)
    p = probabilities.astype(np.float64, copy=True)
    road = is_road.astype(bool, copy=False)
    for _ in range(steps):
        p = anisotropic_diffuse_step(p, road)
    return p


def diffusion_steps_for_tick(dt_sec: float, layer_weight: float) -> int:
    """Scale diffusion iterations with simulated tick length and layer weight."""
    ref = max(settings.momentum_reference_dt_sec, 1.0)
    scaled = settings.road_diffusion_steps * layer_weight * max(dt_sec, 1.0) / ref
    return max(1, min(int(round(scaled)), settings.road_diffusion_steps_max))
