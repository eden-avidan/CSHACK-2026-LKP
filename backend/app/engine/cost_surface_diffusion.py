"""Continuous cost-surface diffusion: L2 distance × terrain friction."""

from __future__ import annotations

import numpy as np

from app.core.config import settings

# 8-connected neighbors (no self). Row 0 = north; col 0 = west.
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


def l2_neighbor_distance(dr: int, dc: int) -> float:
    """Euclidean grid distance: cardinal = 1.0, diagonal = √2."""
    return float(np.hypot(dr, dc))


def build_terrain_cost_map(
    is_road: np.ndarray,
    slope: np.ndarray,
    is_land: np.ndarray,
) -> np.ndarray:
    """
    Per-cell traversal cost (higher = slower movement).

    Roads are baseline; off-road is slower; steep slopes and water add friction.
    """
    road = is_road.astype(bool, copy=False)
    cost = np.full(road.shape, settings.cost_offroad, dtype=np.float64)
    cost[road] = settings.cost_road

    slope_deg = np.degrees(slope.astype(np.float64, copy=False))
    steep = slope_deg >= settings.topo_steep_threshold_deg
    cost[steep] = np.maximum(cost[steep], settings.cost_steep_slope)

    water = ~is_land.astype(bool, copy=False)
    cost[water] = np.maximum(cost[water], settings.cost_water)
    return np.maximum(cost, settings.cost_floor)


def transition_weight_l2_cost(
    l2_distance: float,
    terrain_cost_b: np.ndarray,
    road_a: np.ndarray,
    road_b: np.ndarray,
) -> np.ndarray:
    """
    Combined L2 + topology weight for A → neighbor B.

    ``weight = scale / (L2_distance × terrain_cost[B])`` with soft trail magnetism
    when an off-road cell transitions onto a road neighbor.
    """
    cost_b = np.maximum(terrain_cost_b, settings.cost_floor)
    w = settings.transition_weight_scale / (l2_distance * cost_b)

    trail_pull = (~road_a.astype(bool)) & road_b.astype(bool)
    w[trail_pull] *= 1.0 + settings.trail_magnetism_bonus
    return w


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


def cost_surface_diffuse_step(
    probabilities: np.ndarray,
    terrain_cost: np.ndarray,
    is_road: np.ndarray,
) -> np.ndarray:
    """
    One smooth diffusion step over the cost surface.

    Outflow from each cell is split across neighbors proportional to
    ``1 / (L2 × cost[neighbor])``, normalized so mass is conserved.
    """
    p = probabilities.astype(np.float64, copy=False)
    road = is_road.astype(bool, copy=False)
    cost = terrain_cost.astype(np.float64, copy=False)
    self_w = settings.diffusion_self_weight
    weight_sum = np.full(p.shape, self_w, dtype=np.float64)

    neighbor_weights: list[tuple[int, int, np.ndarray]] = []
    for dr, dc in _EIGHT:
        l2 = l2_neighbor_distance(dr, dc)
        cost_b = _neighbor_field(cost, dr, dc)
        n_road = _neighbor_field(road, dr, dc)
        w = transition_weight_l2_cost(l2, cost_b, road, n_road)
        weight_sum += w
        neighbor_weights.append((dr, dc, w))

    weight_sum = np.maximum(weight_sum, 1e-12)
    p_new = p * (self_w / weight_sum)
    for dr, dc, w in neighbor_weights:
        flow = p * (w / weight_sum)
        p_new += _push_to_neighbor(flow, dr, dc)
    return p_new


def cost_surface_diffusion(
    probabilities: np.ndarray,
    terrain_cost: np.ndarray,
    is_road: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Run multiple cost-surface diffusion steps (road fingers + soft forest bleed)."""
    if steps <= 0:
        return probabilities.astype(np.float64, copy=True)
    p = probabilities.astype(np.float64, copy=True)
    cost = terrain_cost.astype(np.float64, copy=False)
    road = is_road.astype(bool, copy=False)
    for _ in range(steps):
        p = cost_surface_diffuse_step(p, cost, road)
    return p


def diffusion_steps_for_tick(dt_sec: float, layer_weight: float) -> int:
    """Scale diffusion iterations with simulated tick length and layer weight."""
    ref = max(settings.momentum_reference_dt_sec, 1.0)
    scaled = settings.diffusion_steps * layer_weight * max(dt_sec, 1.0) / ref
    return max(1, min(int(round(scaled)), settings.diffusion_steps_max))
