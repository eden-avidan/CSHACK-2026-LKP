from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.models.mission import LatLon


@dataclass
class LayerFlags:
    topography: bool = True
    roads: bool = False
    personality: bool = False
    weather: bool = False
    sea_drift: bool = False

    def apply_update(self, layers: dict[str, bool]) -> None:
        if "topography" in layers:
            self.topography = bool(layers["topography"])
        if "roads" in layers:
            self.roads = bool(layers["roads"])
        if "personality" in layers:
            self.personality = bool(layers["personality"])
        if "weather" in layers:
            self.weather = bool(layers["weather"])
        if "sea_drift" in layers:
            self.sea_drift = bool(layers["sea_drift"])
        from app.engine.layers.registry import ensure_min_one_layer

        ensure_min_one_layer(self)

    def as_dict(self) -> dict[str, bool]:
        return {
            "topography": self.topography,
            "roads": self.roads,
            "personality": self.personality,
            "weather": self.weather,
            "sea_drift": self.sea_drift,
        }

    def any_enabled(self) -> bool:
        return (
            self.topography
            or self.roads
            or self.personality
            or self.weather
            or self.sea_drift
        )


class UpdateLayersMessage(BaseModel):
    event: Literal["update_layers"]
    layers: dict[str, bool]


class EngineTickMessage(BaseModel):
    event: Literal["engine_tick"]
    tick_count: int = 0
    lkp_coords: LatLon
    mpp_coords: LatLon
    layers: dict[str, bool] = Field(default_factory=dict)
    particle_matrix: list[list[float]] = Field(
        default_factory=list,
        description="Downsampled [lat, lon, weight] tuples",
    )
