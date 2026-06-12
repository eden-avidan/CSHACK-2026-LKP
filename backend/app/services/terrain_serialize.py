"""Serialize GridMatrix NodeFields into TerrainInspectResponse shape."""

from __future__ import annotations

import numpy as np

from app.engine.grid_matrix import NodeFields
from app.geospatial.grid import ProbabilityGrid
from app.models.terrain import MarineCurrentInfo, TerrainFieldMeta, TerrainInspectResponse
from app.services.marine_current import MarineCurrent


AVAILABLE_FIELDS = [
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
        label="Reachability",
        kind="scalar",
        description="Tobler/Dijkstra walking reach from LKP: 1 at pin, 0 beyond time horizon, smooth falloff.",
    ),
    TerrainFieldMeta(
        id="elevation",
        label="Elevation",
        kind="scalar",
        unit="m",
        description="SRTM elevation sampled at mission create.",
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
    TerrainFieldMeta(
        id="wind_vectors",
        label="Wind (vectors)",
        kind="vector",
        unit="m/s",
        description="Arrow overlay: per-cell mock wind across the grid (W at bottom-left twisting to N at top-right).",
    ),
    TerrainFieldMeta(
        id="current_vectors",
        label="Current (vectors)",
        kind="vector",
        unit="m/s",
        description="Arrow overlay: flow direction and speed on water cells (Open-Meteo at LKP).",
    ),
]


def _marine_info(marine: MarineCurrent | None) -> MarineCurrentInfo | None:
    if marine is None:
        return None
    return MarineCurrentInfo(
        u_east_mps=marine.u_east_mps,
        v_north_mps=marine.v_north_mps,
        speed_mps=marine.speed_mps,
        direction_deg=marine.direction_deg,
        source=marine.source,
    )


def build_inspect_response(
    grid: ProbabilityGrid,
    node_fields: NodeFields,
    *,
    warnings: list[str] | None = None,
    marine_current: MarineCurrent | None = None,
) -> TerrainInspectResponse:
    """Build inspect payload from the exact NodeFields the grid engine uses."""
    size = grid.rows
    current_speed = np.hypot(node_fields.current_u, node_fields.current_v)
    current_heading = np.degrees(
        np.arctan2(node_fields.current_u, node_fields.current_v)
    ) % 360.0
    fields_np = {
        "road_proximity": node_fields.road_proximity,
        "is_road": node_fields.is_road.astype(np.float64),
        "reachability": node_fields.reachability_score,
        "elevation": node_fields.elevation,
        "slope": np.degrees(node_fields.slope),
        "is_land": node_fields.is_land.astype(np.float64),
        "wind_u": node_fields.wind_u,
        "wind_v": node_fields.wind_v,
        "current_u": node_fields.current_u,
        "current_v": node_fields.current_v,
        "current_speed": current_speed,
        "current_heading": current_heading,
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

    out_warnings = list(warnings or [])
    if field_stats["road_proximity"]["max"] <= 1e-9:
        out_warnings.append(
            "No roads were fetched from OSM providers for this area/time (road_proximity is all zeros)."
        )
    if field_stats["elevation"]["max"] <= 1e-9 and field_stats["slope"]["max"] <= 1e-9:
        out_warnings.append(
            "Elevation provider returned no usable terrain; elevation/slope/reachability may be flat fallback. Road fields are still valid."
        )
    if field_stats["is_land"]["nonzero_frac"] <= 1e-9:
        out_warnings.append("Land mask is empty (all water) at this resolution.")
    if field_stats["reachability"]["max"] <= 1e-9:
        out_warnings.append(
            "Reachability is all zero (topography layer off, sea mode, or flat elevation fallback)."
        )

    return TerrainInspectResponse(
        metadata=grid.metadata,
        rows=size,
        cols=size,
        fields=fields,
        field_stats=field_stats,
        warnings=out_warnings,
        available=AVAILABLE_FIELDS,
        marine_current=_marine_info(marine_current),
    )
