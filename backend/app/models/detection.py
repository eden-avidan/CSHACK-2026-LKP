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
