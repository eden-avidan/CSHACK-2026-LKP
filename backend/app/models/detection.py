from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.mission import LatLon


class DetectionEventMessage(BaseModel):
    type: Literal["detection_event"] = "detection_event"
    mission_id: UUID
    asset_id: str = "drone"
    timestamp: datetime
    person_found: Literal[True] = True
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_percent: float = Field(ge=0.0, le=100.0)
    frame: int | None = None
    bbox: list[float] | None = None
    position: LatLon | None = None


class DroneTrackItem(BaseModel):
    """One drone sortie's revealed path + current position."""

    asset_id: str
    # True for the sortie that locates the person; drives path coloring.
    found: bool = False
    # True while this sortie is mid-flight (vs. launched-and-landed).
    active: bool = False
    position: LatLon | None = None
    # [[lon, lat], ...] — GeoJSON order, ready to drop into a LineString.
    path: list[list[float]] = Field(default_factory=list)


class DroneTrackMessage(BaseModel):
    """Live drone position + the path(s) flown so far this mission.

    ``position``/``path`` mirror the currently active drone (back-compat); the
    ``drones`` list carries every sortie revealed so far so the client can draw
    several drones flying one after another.
    """

    type: Literal["drone_track"] = "drone_track"
    mission_id: UUID
    asset_id: str = "drone"
    timestamp: datetime
    position: LatLon | None = None
    # [[lon, lat], ...] — GeoJSON order, ready to drop into a LineString.
    path: list[list[float]] = Field(default_factory=list)
    drones: list[DroneTrackItem] = Field(default_factory=list)
