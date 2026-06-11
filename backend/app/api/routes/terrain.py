"""Terrain inspection — project the raw layer inputs (roads, elevation, etc.) for a point.

Debug/visualization aid: returns the exact per-cell fields the Grid Matrix engine
receives, so layer authors can validate data before designing transition functions.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter

from app.core.config import settings
from app.engine.node_builder import build_node_fields
from app.geospatial.grid import create_empty_grid
from app.models.terrain import TerrainInspectRequest, TerrainInspectResponse
from app.services.env_ingestion import build_terrain_context
from app.services.terrain_serialize import build_inspect_response
from app.services.topo_reachability import compute_reachability, lkp_to_grid_cell

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terrain", tags=["terrain"])


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

    terrain.reachability = reachability
    node_fields = build_node_fields(terrain, size, weather_enabled=False)
    return build_inspect_response(grid, node_fields)
