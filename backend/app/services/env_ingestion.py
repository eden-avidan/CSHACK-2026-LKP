"""Fetch elevation and OSM road data for terrain-aware particle filter."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx
import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


@dataclass
class TerrainContext:
    slope: np.ndarray  # radians
    aspect_n: np.ndarray  # downhill north component (unit)
    aspect_e: np.ndarray  # downhill east component (unit)
    road_proximity: np.ndarray  # 0..1
    is_land: np.ndarray  # bool
    road_tangent_e: np.ndarray  # unit east component of nearest road
    road_tangent_n: np.ndarray  # unit north component of nearest road


def _flat_terrain(rows: int, cols: int) -> TerrainContext:
    return TerrainContext(
        slope=np.zeros((rows, cols), dtype=np.float64),
        aspect_n=np.zeros((rows, cols), dtype=np.float64),
        aspect_e=np.zeros((rows, cols), dtype=np.float64),
        road_proximity=np.zeros((rows, cols), dtype=np.float64),
        is_land=np.ones((rows, cols), dtype=bool),
        road_tangent_e=np.zeros((rows, cols), dtype=np.float64),
        road_tangent_n=np.zeros((rows, cols), dtype=np.float64),
    )


async def fetch_elevations(grid: ProbabilityGrid) -> np.ndarray | None:
    rows, cols = grid.rows, grid.cols

    step = 4
    sample_indices: list[tuple[int, int]] = []
    sample_locs: list[dict[str, float]] = []
    for row in range(0, rows, step):
        for col in range(0, cols, step):
            lat, lon = cell_centroid_latlon(grid, row, col)
            sample_indices.append((row, col))
            sample_locs.append({"latitude": lat, "longitude": lon})

    try:
        async with httpx.AsyncClient(timeout=settings.env_fetch_timeout_sec) as client:
            resp = await client.post(OPEN_ELEVATION_URL, json={"locations": sample_locs})
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if len(results) != len(sample_locs):
                return None
            sparse = np.zeros((rows, cols), dtype=np.float64)
            for (row, col), item in zip(sample_indices, results):
                sparse[row, col] = float(item["elevation"])
            elev = np.repeat(np.repeat(sparse, step, axis=0), step, axis=1)
            return elev[:rows, :cols]
    except Exception as exc:
        logger.warning("Elevation fetch failed: %s", exc)
        return None


async def fetch_osm_roads(grid: ProbabilityGrid) -> list[list[tuple[float, float]]]:
    b = grid.metadata.bounds
    query = f"""
    [out:json][timeout:25];
    way["highway"~"^(primary|secondary|tertiary|residential|path|footway|track)$"]
        ({b.south},{b.west},{b.north},{b.east});
    out geom;
    """
    try:
        async with httpx.AsyncClient(timeout=settings.env_fetch_timeout_sec) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            data = resp.json()
            segments: list[list[tuple[float, float]]] = []
            for el in data.get("elements", []):
                if el.get("type") != "way":
                    continue
                geom = el.get("geometry", [])
                if len(geom) < 2:
                    continue
                pts = [(p["lon"], p["lat"]) for p in geom]
                segments.append(pts)
            return segments
    except Exception as exc:
        logger.warning("Overpass road fetch failed: %s", exc)
        return []


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _point_segment_dist_m(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float]:
    """Return distance (m) and segment unit tangent (east, north) in degrees space."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return _haversine_m(px, py, ax, ay), 0.0, 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_lon = ax + t * dx
    proj_lat = ay + t * dy
    dist = _haversine_m(px, py, proj_lon, proj_lat)
    seg_len = math.sqrt(dx * dx + dy * dy) + 1e-12
    return dist, dx / seg_len, dy / seg_len


def rasterize_road_fields(
    grid: ProbabilityGrid, segments: list[list[tuple[float, float]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = grid.rows, grid.cols
    proximity = np.zeros((rows, cols), dtype=np.float64)
    tangent_e = np.zeros((rows, cols), dtype=np.float64)
    tangent_n = np.zeros((rows, cols), dtype=np.float64)
    if not segments:
        return proximity, tangent_e, tangent_n

    crs = grid.crs
    for row in range(rows):
        for col in range(cols):
            lat, lon = cell_centroid_latlon(grid, row, col)
            min_dist = float("inf")
            best_te, best_tn = 0.0, 0.0
            for seg in segments:
                for i in range(len(seg) - 1):
                    lon1, lat1 = seg[i]
                    lon2, lat2 = seg[i + 1]
                    d, te_deg, tn_deg = _point_segment_dist_m(lon, lat, lon1, lat1, lon2, lat2)
                    if d < min_dist:
                        min_dist = d
                        # Convert lon/lat tangent to approximate UTM east/north
                        e1, n1 = crs.to_utm(lon1, lat1)
                        e2, n2 = crs.to_utm(lon2, lat2)
                        de, dn = e2 - e1, n2 - n1
                        mag = math.sqrt(de * de + dn * dn) + 1e-8
                        best_te, best_tn = de / mag, dn / mag
            if min_dist < float("inf"):
                proximity[row, col] = math.exp(-min_dist / settings.road_proximity_decay_m)
                tangent_e[row, col] = best_te
                tangent_n[row, col] = best_tn
    return proximity, tangent_e, tangent_n


def build_slope_aspect(elevation: np.ndarray, resolution_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(elevation, resolution_m, resolution_m)
    slope = np.arctan(np.sqrt(gx * gx + gy * gy))
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    aspect_e = -gx / mag
    aspect_n = -gy / mag
    return slope, aspect_n, aspect_e


async def build_terrain_context(grid: ProbabilityGrid) -> TerrainContext:
    rows, cols = grid.rows, grid.cols
    elevation = await fetch_elevations(grid)
    if elevation is None:
        logger.warning("Using flat terrain fallback")
        return _flat_terrain(rows, cols)

    is_land = elevation > settings.land_elevation_threshold_m
    slope, aspect_n, aspect_e = build_slope_aspect(elevation, grid.metadata.resolution_m)
    segments = await fetch_osm_roads(grid)
    road_proximity, road_tangent_e, road_tangent_n = rasterize_road_fields(grid, segments)

    return TerrainContext(
        slope=slope,
        aspect_n=aspect_n,
        aspect_e=aspect_e,
        road_proximity=road_proximity,
        is_land=is_land,
        road_tangent_e=road_tangent_e,
        road_tangent_n=road_tangent_n,
    )
