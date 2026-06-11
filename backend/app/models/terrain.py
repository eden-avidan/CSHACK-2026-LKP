"""Request/response models for the terrain inspection endpoint."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.heatmap import GridMetadata
from app.models.mission import LatLon


class TerrainInspectRequest(BaseModel):
    lkp: LatLon
    grid_size: Optional[int] = Field(default=None, ge=8, le=512)
    resolution_m: Optional[float] = Field(default=None, gt=0)


class TerrainFieldMeta(BaseModel):
    id: str
    label: str
    kind: str  # "scalar" | "mask"
    unit: Optional[str] = None
    description: Optional[str] = None


class TerrainInspectResponse(BaseModel):
    metadata: GridMetadata
    rows: int
    cols: int
    fields: Dict[str, List[float]]
    field_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    available: List[TerrainFieldMeta]
