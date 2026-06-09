from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MissionStatus(str, Enum):
    SEARCHING = "searching"
    TARGET_LOCATED = "target_located"


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class CreateMissionRequest(BaseModel):
    lkp: LatLon
    timestamp: Optional[datetime] = None
    sigma_0_m: Optional[float] = Field(default=None, gt=0)
    step_sec: float = Field(default=60.0, ge=0.1, le=3600, description="Simulated seconds advanced per tick")
    update_interval_sec: float = Field(
        default=60.0, ge=0.1, le=3600, description="Wall-clock seconds between heatmap updates"
    )
    layers: Optional[dict[str, bool]] = None


class UpdatePaceRequest(BaseModel):
    step_sec: Optional[float] = Field(default=None, ge=0.1, le=3600)
    update_interval_sec: Optional[float] = Field(default=None, ge=0.1, le=3600)


class CreateMissionResponse(BaseModel):
    mission_id: UUID
    status: MissionStatus


class MissionResponse(BaseModel):
    mission_id: UUID
    status: MissionStatus
    lkp: LatLon
    created_at: datetime
    tick_count: int
    step_sec: float
    update_interval_sec: float
    simulation_running: bool = True
