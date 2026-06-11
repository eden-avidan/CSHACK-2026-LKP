from __future__ import annotations

from dataclasses import dataclass

from app.layers.base import LayerConfig, LayerContext, PredictState


@dataclass
class WeatherLayer:
    config: LayerConfig = LayerConfig(id="weather", default_enabled=False, default_weight=1.0)

    def adjust_sigmas(self, sigma_v: float, sigma_x: float, weight: float) -> tuple[float, float]:
        return sigma_v, sigma_x

    def apply_velocity(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def apply_displacement(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def apply_post_step(self, state: PredictState, ctx: LayerContext, weight: float) -> PredictState:
        return state

    def kde_road_factor(self, row: int, col: int, ctx: LayerContext, weight: float) -> float:
        return 1.0

    def wind_forcing(self, ctx: LayerContext, weight: float) -> tuple[float, float]:
        if weight <= 0 or ctx.env is None:
            return 0.0, 0.0
        return (ctx.env.u_w + ctx.env.u_c) * weight, (ctx.env.v_w + ctx.env.v_c) * weight


weather_layer = WeatherLayer()
