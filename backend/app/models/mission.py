from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.personality import PersonalityProfile


class MissionStatus(str, Enum):
    SEARCHING = "searching"
    TARGET_LOCATED = "target_located"


class MissionMode(str, Enum):
    LIVE = "live"
    OFFLINE = "offline"


class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


BASE_STEP_SEC = 10.0


def live_update_interval_sec() -> float:
    """Wall-clock seconds between live heatmap pushes (from settings.filter_hz)."""
    from app.core.config import settings

    hz = max(float(settings.filter_hz), 0.01)
    return 1.0 / hz


# Deprecated alias — prefer live_update_interval_sec() so .env can tune filter_hz.
LIVE_UPDATE_INTERVAL_SEC = 1.0


class CreateMissionRequest(BaseModel):
    lkp: LatLon
    mode: MissionMode = MissionMode.LIVE
    lkp_timestamp: Optional[datetime] = None
    timestamp: Optional[datetime] = None  # deprecated alias for lkp_timestamp
    simulation_start_timestamp: Optional[datetime] = Field(
        default=None,
        description="Offline: simulated 'now' when batch ends and live playback begins",
    )
    sigma_0_m: Optional[float] = Field(default=None, gt=0)
    pace: float = Field(default=1.0, ge=0.1, le=120.0, description="Simulation speed multiplier")
    step_sec: Optional[float] = Field(default=None, ge=0.1, le=3600, deprecated=True)
    update_interval_sec: Optional[float] = Field(
        default=None, ge=0.1, le=3600, deprecated=True
    )
    layers: Optional[dict[str, bool]] = None
    personality: Optional[PersonalityProfile] = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "CreateMissionRequest":
        if self.lkp_timestamp is None and self.timestamp is not None:
            self.lkp_timestamp = self.timestamp
        if self.mode == MissionMode.OFFLINE and self.lkp_timestamp is None:
            raise ValueError("lkp_timestamp is required for offline mode")
        if self.mode == MissionMode.OFFLINE:
            if self.simulation_start_timestamp is None:
                self.simulation_start_timestamp = datetime.now(timezone.utc)
            lkp_ts = self.lkp_timestamp
            sim_start = self.simulation_start_timestamp
            if lkp_ts.tzinfo is None:
                lkp_ts = lkp_ts.replace(tzinfo=timezone.utc)
            if sim_start.tzinfo is None:
                sim_start = sim_start.replace(tzinfo=timezone.utc)
            if sim_start < lkp_ts:
                raise ValueError("simulation_start_timestamp must be on or after lkp_timestamp")
        layers = self.layers or {}
        if layers.get("personality") and self.personality is None:
            self.personality = PersonalityProfile()
        return self


class UpdatePaceRequest(BaseModel):
    pace: Optional[float] = Field(default=None, ge=0.1, le=120.0)
    step_sec: Optional[float] = Field(default=None, ge=0.1, le=3600, deprecated=True)
    update_interval_sec: Optional[float] = Field(default=None, ge=0.1, le=3600, deprecated=True)


class CreateMissionResponse(BaseModel):
    mission_id: UUID
    status: MissionStatus


class MissionResponse(BaseModel):
    mission_id: UUID
    status: MissionStatus
    mode: MissionMode
    lkp: LatLon
    lkp_timestamp: Optional[datetime] = None
    simulation_start_timestamp: Optional[datetime] = None
    created_at: datetime
    tick_count: int
    pace: float
    step_sec: float
    update_interval_sec: float
    simulation_running: bool = True
