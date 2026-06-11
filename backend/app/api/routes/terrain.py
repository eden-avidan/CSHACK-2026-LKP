"""Terrain inspection — project the raw layer inputs (roads, elevation, etc.) for a point.

Debug/visualization aid: returns the exact per-cell fields the Grid Matrix engine
receives, so layer authors can validate data before designing transition functions.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter

from app.core.config import settings
from app.geospatial.grid import create_empty_grid
from app.models.terrain import (
    TerrainFieldMeta,
    TerrainInspectRequest,
    TerrainInspectResponse,
)
from app.services.env_ingestion import build_terrain_context
from app.services.topo_reachability import compute_reachability, lkp_to_grid_cell

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terrain", tags=["terrain"])

_AVAILABLE_FIELDS = [
    TerrainFieldMeta(
        id="road_proximity",
        label="Road proximity",
        kind="scalar",
        description="0..1 closeness to nearest OSM road (decay by distance).",
    ),
    TerrainFieldMeta(
        id="is_road",
        label="Road cells (mask)",
        kind="mask",
        description="Cells within snap radius of a road.",
    ),
    TerrainFieldMeta(
        id="reachability",
        label="Reachability (Tobler/Dijkstra)",
        kind="scalar",
        description="Travel-time-based prior from the LKP over 6 h horizon.",
    ),
    TerrainFieldMeta(
        id="elevation",
        label="Elevation",
        kind="scalar",
        unit="m",
        description="SRTM elevation sampled from OpenTopoData / Open-Meteo.",
    ),
    TerrainFieldMeta(
        id="slope",
        label="Slope",
        kind="scalar",
        unit="\u00b0",
        description="Terrain steepness in degrees.",
    ),
    TerrainFieldMeta(
        id="is_land",
        label="Land / Water (mask)",
        kind="mask",
        description="Cells above the land elevation threshold.",
    ),
]


@router.post("/inspect", response_model=TerrainInspectResponse)
async def inspect_terrain(body: TerrainInspectRequest) -> TerrainInspectResponse:
    # Same geographic coverage as the mission grid, but finer cells by default.
    full_extent_m = settings.grid_size * settings.grid_resolution_m
    if body.resolution_m is not None:
        resolution = body.resolution_m
        size = body.grid_size or int(round(full_extent_m / resolution))
    elif body.grid_size is not None:
        size = body.grid_size
        resolution = full_extent_m / size
    else:
        resolution = settings.terrain_inspect_resolution_m
        size = int(round(full_extent_m / resolution))
    size = max(8, min(512, size))
    resolution = full_extent_m / size

    grid = create_empty_grid(body.lkp, resolution, size)
    terrain = await build_terrain_context(grid)

    start_row, start_col = lkp_to_grid_cell(grid, grid.crs.origin_e, grid.crs.origin_n)
    try:
        reachability = compute_reachability(
            grid, terrain.elevation, start_row, start_col, max_hours=6.0
        )
    except Exception as exc:  # reachability is best-effort for visualization
        logger.warning("Reachability inspect failed: %s", exc)
        reachability = np.zeros((size, size), dtype=np.float64)

    snap_threshold = np.exp(
        -settings.road_snap_radius_m / settings.road_proximity_decay_m
    )
    is_road = (terrain.road_proximity >= snap_threshold).astype(np.float64)

    fields_np = {
        "road_proximity": terrain.road_proximity,
        "is_road": is_road,
        "reachability": reachability,
        "elevation": terrain.elevation,
        "slope": np.degrees(terrain.slope),
        "is_land": terrain.is_land.astype(np.float64),
    }
    fields = {
        key: np.asarray(value, dtype=float).flatten(order="C").tolist()
        for key, value in fields_np.items()
    }
    field_stats: dict[str, dict[str, float]] = {}
    for key, value in fields_np.items():
        arr = np.asarray(value, dtype=np.float64)
        field_stats[key] = {
            "min": float(np.nanmin(arr)) if arr.size else 0.0,
            "max": float(np.nanmax(arr)) if arr.size else 0.0,
            "nonzero_frac": float(np.count_nonzero(arr > 1e-9)) / float(arr.size or 1),
        }
    warnings: list[str] = []
    if field_stats["road_proximity"]["max"] <= 1e-9:
        warnings.append(
            "No roads were fetched from OSM providers for this area/time (road_proximity is all zeros)."
        )
    if field_stats["elevation"]["max"] <= 1e-9 and field_stats["slope"]["max"] <= 1e-9:
        warnings.append(
            "Elevation provider returned no usable terrain; elevation/slope/reachability are flat fallback fields. Road fields are still valid."
        )
    if field_stats["is_land"]["nonzero_frac"] <= 1e-9:
        warnings.append("Land mask is empty (all water) at this resolution.")

    return TerrainInspectResponse(
        metadata=grid.metadata,
        rows=size,
        cols=size,
        fields=fields,
        field_stats=field_stats,
        warnings=warnings,
        available=_AVAILABLE_FIELDS,
    )
