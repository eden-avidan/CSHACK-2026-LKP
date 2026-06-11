"""Backward-compatible re-exports; prefer ``cost_surface_diffusion``."""

from __future__ import annotations

from app.engine.cost_surface_diffusion import (
    build_terrain_cost_map,
    cost_surface_diffuse_step,
    cost_surface_diffusion,
    diffusion_steps_for_tick,
    l2_neighbor_distance,
    transition_weight_l2_cost,
)

# Legacy names used by older tests / docs.
anisotropic_diffuse_step = cost_surface_diffuse_step
def anisotropic_road_diffusion(
    probabilities: np.ndarray,
    is_road: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Legacy wrapper: flat terrain, land everywhere."""
    land = np.ones(is_road.shape, dtype=bool)
    slope = np.zeros(is_road.shape, dtype=np.float64)
    cost = build_terrain_cost_map(is_road, slope, land)
    return cost_surface_diffusion(probabilities, cost, is_road, steps=steps)


def pairwise_transition_weight(road_a: bool, road_b: bool) -> float:
    """Approximate relative conductance for legacy transition_weight hooks."""
    from app.core.config import settings

    if road_a and road_b:
        return settings.transition_weight_scale / settings.cost_road
    if road_a and not road_b:
        return settings.transition_weight_scale / settings.cost_offroad
    if not road_a and road_b:
        base = settings.transition_weight_scale / settings.cost_road
        return base * (1.0 + settings.trail_magnetism_bonus)
    return settings.transition_weight_scale / settings.cost_offroad
