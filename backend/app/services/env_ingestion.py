"""Fetch elevation and OSM road data for terrain-aware particle filter."""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import httpx
import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon

logger = logging.getLogger(__name__)

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_TOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_BATCH_SIZE = 100
ELEVATION_TARGET_SAMPLES = 256
OPEN_TOPODATA_MIN_INTERVAL_SEC = 1.1


@dataclass
class TerrainContext:
    elevation: np.ndarray
    slope: np.ndarray  # radians
    aspect_n: np.ndarray  # downhill north component (unit)
    aspect_e: np.ndarray  # downhill east component (unit)
    road_proximity: np.ndarray  # 0..1
    is_land: np.ndarray  # bool
    road_tangent_e: np.ndarray  # unit east component of nearest road
    road_tangent_n: np.ndarray  # unit north component of nearest road
    reachability: np.ndarray | None = None  # Tobler/Dijkstra prior on terrain grid


def _flat_terrain(rows: int, cols: int) -> TerrainContext:
    return TerrainContext(
        elevation=np.zeros((rows, cols), dtype=np.float64),
        slope=np.zeros((rows, cols), dtype=np.float64),
        aspect_n=np.zeros((rows, cols), dtype=np.float64),
        aspect_e=np.zeros((rows, cols), dtype=np.float64),
        road_proximity=np.zeros((rows, cols), dtype=np.float64),
        is_land=np.ones((rows, cols), dtype=bool),
        road_tangent_e=np.zeros((rows, cols), dtype=np.float64),
        road_tangent_n=np.zeros((rows, cols), dtype=np.float64),
        reachability=None,
    )


async def _fetch_elevation_opentopodata(
    client: httpx.AsyncClient, lats: list[float], lons: list[float]
) -> list[float] | None:
    elevations: list[float] = []
    for batch_idx, start in enumerate(range(0, len(lats), ELEVATION_BATCH_SIZE)):
        if batch_idx > 0:
            await asyncio.sleep(OPEN_TOPODATA_MIN_INTERVAL_SEC)
        batch_lats = lats[start : start + ELEVATION_BATCH_SIZE]
        batch_lons = lons[start : start + ELEVATION_BATCH_SIZE]
        locs = "|".join(f"{lat},{lon}" for lat, lon in zip(batch_lats, batch_lons))
        resp = await client.get(OPEN_TOPODATA_URL, params={"locations": locs})
        if resp.status_code == 429:
            await asyncio.sleep(2.0)
            resp = await client.get(OPEN_TOPODATA_URL, params={"locations": locs})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if len(results) != len(batch_lats):
            return None
        for item in results:
            elev = item.get("elevation")
            if elev is None:
                return None
            elevations.append(float(elev))
    return elevations


async def _fetch_elevation_open_meteo(
    client: httpx.AsyncClient, lats: list[float], lons: list[float]
) -> list[float] | None:
    elevations: list[float] = []
    batch_size = 50
    for start in range(0, len(lats), batch_size):
        batch_lats = lats[start : start + batch_size]
        batch_lons = lons[start : start + batch_size]
        resp = await client.get(
            OPEN_METEO_ELEVATION_URL,
            params={
                "latitude": ",".join(str(v) for v in batch_lats),
                "longitude": ",".join(str(v) for v in batch_lons),
            },
        )
        resp.raise_for_status()
        batch = resp.json().get("elevation", [])
        if len(batch) != len(batch_lats):
            return None
        elevations.extend(float(v) for v in batch)
    return elevations


async def _fetch_elevation_open_elevation(
    client: httpx.AsyncClient, lats: list[float], lons: list[float]
) -> list[float] | None:
    elevations: list[float] = []
    for start in range(0, len(lats), ELEVATION_BATCH_SIZE):
        batch_lats = lats[start : start + ELEVATION_BATCH_SIZE]
        batch_lons = lons[start : start + ELEVATION_BATCH_SIZE]
        locs = [{"latitude": lat, "longitude": lon} for lat, lon in zip(batch_lats, batch_lons)]
        resp = await client.post(OPEN_ELEVATION_URL, json={"locations": locs})
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if len(results) != len(batch_lats):
            return None
        elevations.extend(float(item["elevation"]) for item in results)
    return elevations


def _upsample_bilinear(coarse: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Expand a coarse sample grid to full resolution with bilinear interpolation."""
    n_r, n_c = coarse.shape
    if n_r == rows and n_c == cols:
        return coarse.astype(np.float64, copy=True)
    row_f = np.linspace(0.0, n_r - 1, rows)
    col_f = np.linspace(0.0, n_c - 1, cols)
    r0 = np.floor(row_f).astype(np.int64)
    c0 = np.floor(col_f).astype(np.int64)
    r1 = np.minimum(r0 + 1, n_r - 1)
    c1 = np.minimum(c0 + 1, n_c - 1)
    dr = (row_f - r0)[:, None]
    dc = (col_f - c0)[None, :]
    v00 = coarse[r0[:, None], c0[None, :]]
    v01 = coarse[r0[:, None], c1[None, :]]
    v10 = coarse[r1[:, None], c0[None, :]]
    v11 = coarse[r1[:, None], c1[None, :]]
    return (
        v00 * (1.0 - dr) * (1.0 - dc)
        + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc)
        + v11 * dr * dc
    )


def _smooth_elevation(elevation: np.ndarray, passes: int = 1) -> np.ndarray:
    """Light 3×3 box smooth to soften block boundaries before slope."""
    if passes <= 0:
        return elevation
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64)
    kernel /= kernel.sum()
    out = elevation.astype(np.float64, copy=True)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        smoothed = np.zeros_like(out)
        for i in range(3):
            for j in range(3):
                smoothed += kernel[i, j] * padded[i : i + out.shape[0], j : j + out.shape[1]]
        out = smoothed
    return out


async def fetch_elevations(grid: ProbabilityGrid) -> np.ndarray | None:
    rows, cols = grid.rows, grid.cols

    # Adaptive stride keeps API load bounded on high-res inspect grids.
    step = max(1, int(np.ceil(np.sqrt(rows * cols / ELEVATION_TARGET_SAMPLES))))
    sample_indices: list[tuple[int, int]] = []
    lats: list[float] = []
    lons: list[float] = []
    for row in range(0, rows, step):
        for col in range(0, cols, step):
            lat, lon = cell_centroid_latlon(grid, row, col)
            sample_indices.append((row, col))
            lats.append(lat)
            lons.append(lon)

    timeout = max(settings.env_fetch_timeout_sec, 20.0)
    providers = (
        ("Open-Meteo", _fetch_elevation_open_meteo),
        ("OpenTopoData", _fetch_elevation_opentopodata),
        ("Open-Elevation", _fetch_elevation_open_elevation),
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": "RescuEdge/0.1 terrain-ingest"},
    ) as client:
        for name, fetch_fn in providers:
            try:
                elevations = await fetch_fn(client, lats, lons)
                if not elevations:
                    logger.warning("%s returned no elevation data", name)
                    continue
                n_sr = len(range(0, rows, step))
                n_sc = len(range(0, cols, step))
                coarse = np.zeros((n_sr, n_sc), dtype=np.float64)
                for idx, (row, col) in enumerate(sample_indices):
                    coarse[row // step, col // step] = elevations[idx]
                filled = _upsample_bilinear(coarse, rows, cols)
                filled = _smooth_elevation(filled, passes=1)
                logger.info(
                    "Fetched %d elevation samples via %s (step=%d, coarse=%dx%d)",
                    len(elevations),
                    name,
                    step,
                    n_sr,
                    n_sc,
                )
                return filled
            except Exception as exc:
                logger.warning("Elevation fetch failed (%s): %s", name, exc)
                continue
    return None


async def fetch_osm_roads(grid: ProbabilityGrid) -> list[list[tuple[float, float]]]:
    b = grid.metadata.bounds
    query = f"""
    [out:json][timeout:25];
    way["highway"~"^(primary|secondary|tertiary|residential|path|footway|track)$"]
        ({b.south},{b.west},{b.north},{b.east});
    out geom;
    """
    source_pref = settings.roads_data_source.strip().lower()
    try_overpass = source_pref in {"auto", "overpass"}
    try_osm_map = source_pref in {"auto", "osm_map", "overpass"}

    async with httpx.AsyncClient(
        timeout=max(settings.env_fetch_timeout_sec, 15.0),
        headers={
            "User-Agent": "RescuEdge/0.1 terrain-ingest",
            "Accept": "application/json",
        },
    ) as client:
        if try_overpass:
            for url in OVERPASS_URLS:
                try:
                    resp = await client.post(url, data={"data": query})
                    if resp.status_code == 406:
                        # Some mirrors reject form POST but accept query-param GET.
                        resp = await client.get(url, params={"data": query})
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
                    if segments:
                        logger.info("Fetched %d road segments via %s", len(segments), url)
                    else:
                        logger.warning("Overpass returned 0 road segments via %s", url)
                    return segments
                except Exception as exc:
                    snippet = ""
                    try:
                        snippet = resp.text[:180]  # type: ignore[name-defined]
                    except Exception:
                        pass
                    logger.warning("Overpass road fetch failed (%s): %s %s", url, exc, snippet)
                    continue
        else:
            logger.info("Skipping Overpass by ROADS_DATA_SOURCE=%s", source_pref)
    # Fallback: direct OSM map API (XML).
    # For dense urban bboxes OSM may return 400 "too many nodes", so we split recursively.
    if not try_osm_map:
        logger.warning("Skipping OSM map fallback by ROADS_DATA_SOURCE=%s", source_pref)
        return []
    try:
        bbox = (b.west, b.south, b.east, b.north)
        async with httpx.AsyncClient(
            timeout=max(settings.env_fetch_timeout_sec, 15.0),
            headers={"User-Agent": "RescuEdge/0.1 terrain-ingest"},
        ) as fallback_client:
            segments = await _fetch_osm_map_segments_split(fallback_client, bbox)
        if segments:
            logger.info("Fetched %d road segments via OSM map API", len(segments))
            return segments
        logger.warning("OSM map API returned 0 road segments")
    except Exception as exc:
        logger.warning("OSM map API fallback failed: %s", exc)
    return []


def _parse_osm_xml_segments(xml_text: str) -> list[list[tuple[float, float]]]:
    root = ET.fromstring(xml_text)
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        node_id = node.attrib.get("id")
        lat = node.attrib.get("lat")
        lon = node.attrib.get("lon")
        if node_id and lat and lon:
            nodes[node_id] = (float(lon), float(lat))

    allowed = {
        "primary",
        "secondary",
        "tertiary",
        "residential",
        "path",
        "footway",
        "track",
    }
    segments: list[list[tuple[float, float]]] = []
    for way in root.findall("way"):
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in way.findall("tag")}
        highway = tags.get("highway")
        if highway not in allowed:
            continue
        pts: list[tuple[float, float]] = []
        for nd in way.findall("nd"):
            ref = nd.attrib.get("ref")
            if not ref:
                continue
            p = nodes.get(ref)
            if p is not None:
                pts.append(p)
        if len(pts) >= 2:
            segments.append(pts)
    return segments


def _split_bbox_4(
    bbox: tuple[float, float, float, float]
) -> Iterable[tuple[float, float, float, float]]:
    west, south, east, north = bbox
    mid_lon = (west + east) * 0.5
    mid_lat = (south + north) * 0.5
    return (
        (west, south, mid_lon, mid_lat),
        (mid_lon, south, east, mid_lat),
        (west, mid_lat, mid_lon, north),
        (mid_lon, mid_lat, east, north),
    )


async def _fetch_osm_map_segments_split(
    client: httpx.AsyncClient,
    bbox: tuple[float, float, float, float],
    depth: int = 0,
    max_depth: int = 3,
) -> list[list[tuple[float, float]]]:
    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    resp = await client.get(OSM_MAP_URL, params={"bbox": bbox_str})
    if resp.status_code == 200:
        return _parse_osm_xml_segments(resp.text)

    body = resp.text.lower()
    too_many_nodes = resp.status_code == 400 and "too many nodes" in body
    if too_many_nodes and depth < max_depth:
        logger.warning("OSM bbox too dense; splitting depth=%d bbox=%s", depth, bbox_str)
        out: list[list[tuple[float, float]]] = []
        for child in _split_bbox_4(bbox):
            out.extend(await _fetch_osm_map_segments_split(client, child, depth + 1, max_depth))
        return out

    resp.raise_for_status()
    return []


_EDT_INF = 1e20


def _dt1d(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Felzenszwalb 1D squared distance transform with nearest-source index."""
    n = f.shape[0]
    d = np.empty(n, dtype=np.float64)
    arg = np.empty(n, dtype=np.int64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -_EDT_INF
    z[1] = _EDT_INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _EDT_INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
        arg[q] = v[k]
    return d, arg


def _edt_with_index(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact squared Euclidean distance (in cells) + nearest source linear index."""
    rows, cols = mask.shape
    f = np.where(mask, 0.0, _EDT_INF)

    d_col = np.empty((rows, cols), dtype=np.float64)
    src_row = np.empty((rows, cols), dtype=np.int64)
    for c in range(cols):
        d_col[:, c], src_row[:, c] = _dt1d(f[:, c])

    dist2 = np.empty((rows, cols), dtype=np.float64)
    src_col = np.empty((rows, cols), dtype=np.int64)
    for r in range(rows):
        dist2[r, :], src_col[r, :] = _dt1d(d_col[r, :])

    nearest_row = src_row[np.arange(rows)[:, None], src_col]
    nearest_lin = nearest_row * cols + src_col
    return dist2, nearest_lin


def rasterize_road_fields(
    grid: ProbabilityGrid, segments: list[list[tuple[float, float]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized: rasterize road polylines to the grid, then distance-transform.

    Old impl was O(rows*cols*segment_points) pure Python and froze the server for
    thousands of segments. This samples points along each edge, snaps them to cells,
    and uses an exact Euclidean distance transform (O(rows*cols)) for proximity, with
    nearest-road tangent propagated via the distance-transform's source index.
    """
    rows, cols = grid.rows, grid.cols
    proximity = np.zeros((rows, cols), dtype=np.float64)
    tangent_e = np.zeros((rows, cols), dtype=np.float64)
    tangent_n = np.zeros((rows, cols), dtype=np.float64)
    if not segments:
        return proximity, tangent_e, tangent_n

    res = grid.metadata.resolution_m
    half = (rows * res) / 2.0
    min_e = grid.crs.origin_e - half
    max_n = grid.crs.origin_n + half

    es_list: list[np.ndarray] = []
    ns_list: list[np.ndarray] = []
    te_list: list[np.ndarray] = []
    tn_list: list[np.ndarray] = []
    for seg in segments:
        if len(seg) < 2:
            continue
        lons = np.fromiter((p[0] for p in seg), dtype=np.float64, count=len(seg))
        lats = np.fromiter((p[1] for p in seg), dtype=np.float64, count=len(seg))
        e, n = grid.crs.to_utm_array(lons, lats)
        de = np.diff(e)
        dn = np.diff(n)
        seg_len = np.hypot(de, dn)
        for i in range(de.shape[0]):
            length = seg_len[i]
            if length <= 0.0:
                continue
            steps = max(1, int(length / (res * 0.5)))
            t = np.linspace(0.0, 1.0, steps + 1)
            es_list.append(e[i] + t * de[i])
            ns_list.append(n[i] + t * dn[i])
            te_list.append(np.full(t.shape, de[i] / length, dtype=np.float64))
            tn_list.append(np.full(t.shape, dn[i] / length, dtype=np.float64))

    if not es_list:
        return proximity, tangent_e, tangent_n

    se = np.concatenate(es_list)
    sn = np.concatenate(ns_list)
    ste = np.concatenate(te_list)
    stn = np.concatenate(tn_list)

    col_idx = np.floor((se - min_e) / res).astype(np.int64)
    row_idx = np.floor((max_n - sn) / res).astype(np.int64)
    valid = (row_idx >= 0) & (row_idx < rows) & (col_idx >= 0) & (col_idx < cols)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    ste, stn = ste[valid], stn[valid]
    if row_idx.size == 0:
        return proximity, tangent_e, tangent_n

    lin = row_idx * cols + col_idx
    size = rows * cols
    te_sum = np.zeros(size, dtype=np.float64)
    tn_sum = np.zeros(size, dtype=np.float64)
    cnt = np.zeros(size, dtype=np.float64)
    np.add.at(te_sum, lin, ste)
    np.add.at(tn_sum, lin, stn)
    np.add.at(cnt, lin, 1.0)
    nz = cnt > 0
    te_cell = np.zeros(size, dtype=np.float64)
    tn_cell = np.zeros(size, dtype=np.float64)
    te_cell[nz] = te_sum[nz] / cnt[nz]
    tn_cell[nz] = tn_sum[nz] / cnt[nz]

    mask = np.zeros(size, dtype=bool)
    mask[lin] = True
    dist2_cells, nearest_lin = _edt_with_index(mask.reshape(rows, cols))
    dist_m = np.sqrt(dist2_cells) * res
    proximity = np.exp(-dist_m / settings.road_proximity_decay_m)
    tangent_e = te_cell[nearest_lin].reshape(rows, cols)
    tangent_n = tn_cell[nearest_lin].reshape(rows, cols)
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
    # Fetch roads and elevation concurrently so total latency is ~max(roads, elev)
    # instead of their sum (elevation often times out on restricted networks).
    segments_task = asyncio.create_task(fetch_osm_roads(grid))
    elevation_task = asyncio.create_task(fetch_elevations(grid))

    segments = await segments_task
    # Rasterization is CPU-bound; run off the event loop so /health and the
    # mission WebSocket stay responsive even for thousands of road segments.
    road_proximity, road_tangent_e, road_tangent_n = await asyncio.to_thread(
        rasterize_road_fields, grid, segments
    )

    elevation = await elevation_task
    elevation_missing = elevation is None or not np.isfinite(elevation).any()
    elevation_is_flat_zero = elevation is not None and np.nanmax(elevation) <= 1e-9
    if elevation_missing or elevation_is_flat_zero:
        logger.warning("Using flat terrain fallback (roads still applied)")
        terrain = _flat_terrain(rows, cols)
        terrain.road_proximity = road_proximity
        terrain.road_tangent_e = road_tangent_e
        terrain.road_tangent_n = road_tangent_n
        return terrain

    is_land = elevation > settings.land_elevation_threshold_m
    slope, aspect_n, aspect_e = build_slope_aspect(elevation, grid.metadata.resolution_m)

    return TerrainContext(
        elevation=elevation,
        slope=slope,
        aspect_n=aspect_n,
        aspect_e=aspect_e,
        road_proximity=road_proximity,
        is_land=is_land,
        road_tangent_e=road_tangent_e,
        road_tangent_n=road_tangent_n,
        reachability=None,
    )
