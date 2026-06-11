from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import settings
from app.layers.base import LayerConfig, LayerContext, PredictState


@dataclass
class PersonalityParticleLayer:
    """Legacy particle pipeline hook when personality layer is enabled."""

    config: LayerConfig = field(
        default_factory=lambda: LayerConfig(id="personality", default_enabled=False, default_weight=1.0)
    )

    def adjust_sigmas(self, sigma_v: float, sigma_x: float, weight: float) -> tuple[float, float]:
        if weight <= 0:
            return sigma_v, sigma_x
        factor = 1.0 - weight * 0.5
        return sigma_v * factor, sigma_x * factor

    def apply_velocity(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        if weight <= 0:
            return state
        factor = 1.0 - weight * (1.0 - settings.injured_velocity_factor)
        state.v_n *= factor
        state.v_e *= factor
        return state

    def apply_displacement(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def apply_post_step(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def kde_road_factor(self, row: int, col: int, ctx: LayerContext, weight: float) -> float:
        return 1.0


personality_layer = PersonalityParticleLayer()
