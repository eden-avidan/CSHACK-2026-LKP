from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.models.mission import LatLon


@dataclass
class LayerFlags:
    topography: bool = True
    roads: bool = False
    subject_injured: bool = False
    weather: bool = False

    def apply_update(self, layers: dict[str, bool]) -> None:
        if "topography" in layers:
            self.topography = bool(layers["topography"])
        if "roads" in layers:
            self.roads = bool(layers["roads"])
        if "subject_injured" in layers:
            self.subject_injured = bool(layers["subject_injured"])
        if "weather" in layers:
            self.weather = bool(layers["weather"])

    def as_dict(self) -> dict[str, bool]:
        return {
            "topography": self.topography,
            "roads": self.roads,
            "subject_injured": self.subject_injured,
            "weather": self.weather,
        }


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
