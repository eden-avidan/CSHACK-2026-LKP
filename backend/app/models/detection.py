from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
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
    frame: Optional[int] = None
    bbox: Optional[list[float]] = None
    position: Optional[LatLon] = None


class DroneTrackItem(BaseModel):
    """One drone sortie's revealed path + current position."""

    asset_id: str
    found: bool = False
    active: bool = False
    position: Optional[LatLon] = None
    path: list[list[float]] = Field(default_factory=list)


class DroneTrackMessage(BaseModel):
    """Live drone position + the path(s) flown so far this mission."""

    type: Literal["drone_track"] = "drone_track"
    mission_id: UUID
    asset_id: str = "drone"
    timestamp: datetime
    position: Optional[LatLon] = None
    path: list[list[float]] = Field(default_factory=list)
    drones: list[DroneTrackItem] = Field(default_factory=list)
