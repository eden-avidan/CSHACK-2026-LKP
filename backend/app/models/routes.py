from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class GeoJsonLineString(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[list[float]]


class DroneRouteResponse(BaseModel):
    mission_id: UUID
    route: GeoJsonLineString
    expected_coverage: float
    length_m: float
    route_points: int
