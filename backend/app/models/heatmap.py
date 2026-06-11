from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.mission import LatLon


class GridBounds(BaseModel):
    north: float
    south: float
    east: float
    west: float


class GridCorners(BaseModel):
    """Actual WGS84 corners of the UTM-aligned grid (NOT axis-aligned).

    Use these (not `bounds`) when you need accurate per-cell lat/lon math on
    the frontend — `bounds` is the axis-aligned bbox of these four points
    and can drift O(100 m) from true cell positions near UTM zone edges.
    """

    nw: LatLon
    ne: LatLon
    se: LatLon
    sw: LatLon


class GridMetadata(BaseModel):
    origin: LatLon
    resolution_m: float
    rows: int
    cols: int
    crs_epsg: int
    bounds: GridBounds
    corners: GridCorners


class HeatmapResponse(BaseModel):
    mission_id: UUID
    metadata: GridMetadata
    probabilities: list[float]


class HeatmapFullMessage(BaseModel):
    type: str = "heatmap_full"
    mission_id: UUID
    timestamp: datetime
    metadata: GridMetadata
    probabilities: list[float]


class HeatmapCellDelta(BaseModel):
    row: int
    col: int
    probability: float


class HeatmapDeltaMessage(BaseModel):
    type: str = "heatmap_delta"
    mission_id: UUID
    timestamp: datetime
    cells: list[HeatmapCellDelta]


class NegativeSearchRequest(BaseModel):
    mission_id: UUID
    polygon: dict
    pod: float = Field(default=0.85, ge=0.01, le=0.99)


class NegativeSearchResponse(BaseModel):
    mission_id: UUID
    cells_updated: int
    cells: list[HeatmapCellDelta]
