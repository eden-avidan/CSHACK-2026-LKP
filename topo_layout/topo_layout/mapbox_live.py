from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _StubResponse:
        def __init__(self, content: Any = None, media_type: str | None = None) -> None:
            self.content = content
            self.media_type = media_type

    class HTMLResponse(_StubResponse):
        pass

    class JSONResponse(_StubResponse):
        pass

    class Response(_StubResponse):
        pass

    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default: Any, **_: Any) -> Any:
        return default

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def get(self, *args: Any, **kwargs: Any):
            def decorator(func):
                return func

            return decorator

        def post(self, *args: Any, **kwargs: Any):
            def decorator(func):
                return func

            return decorator

from .mobility import (
    TerrainInfluenceConfig,
    compute_heatmap,
    render_heatmap_image,
)
from .preprocessing import DemRaster, GeoReference


HTML_PATH = Path(__file__).with_name("web").joinpath("index.html")
TERRAIN_TILESET_ID = "mapbox.terrain-rgb"


class MapboxHeatmapRequest(BaseModel):
    west: float = Field(..., description="Bounding box west longitude")
    south: float = Field(..., description="Bounding box south latitude")
    east: float = Field(..., description="Bounding box east longitude")
    north: float = Field(..., description="Bounding box north latitude")
    last_known_lon: float = Field(..., description="Last known longitude")
    last_known_lat: float = Field(..., description="Last known latitude")
    last_known_timestamp: str | None = Field(
        None,
        description="Last known timestamp in ISO 8601 format. When provided, elapsed time is computed on the backend.",
    )
    max_hours: float | None = Field(
        None,
        gt=0,
        description="Time horizon in hours. Used as a fallback when no timestamp is provided.",
    )
    map_zoom: float = Field(12.0, description="Current frontend map zoom")
    grid_width: int = Field(256, ge=64, le=768)
    grid_height: int = Field(256, ge=64, le=768)
    probability_method: str = Field("linear", pattern="^(linear|exponential)$")
    steep_weight: float = 0.7
    cliff_like_weight: float = 0.2
    valley_weight: float = 1.15
    ridge_weight: float = 0.9
    terrain_steep_threshold_deg: float = 30.0
    terrain_cliff_threshold_deg: float = 45.0
    terrain_neighborhood_size: int = 3
    terrain_ridge_threshold_m: float = 5.0
    terrain_valley_threshold_m: float = 5.0
    use_terrain_influence: bool = True


@dataclass(slots=True)
class _TileImage:
    rgb: np.ndarray
    width: int
    height: int


class _TileCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int], _TileImage] = {}

    def get(self, zoom: int, x: int, y: int, access_token: str) -> _TileImage:
        key = (zoom, x, y)
        if key not in self._cache:
            self._cache[key] = _fetch_terrain_tile(zoom, x, y, access_token)
        return self._cache[key]


class _HeatmapStore:
    def __init__(self) -> None:
        self._images: dict[str, bytes] = {}
        self._created_at: dict[str, float] = {}

    def put(self, image_bytes: bytes) -> str:
        heatmap_id = uuid4().hex
        self._images[heatmap_id] = image_bytes
        self._created_at[heatmap_id] = time.time()
        self._prune(max_age_seconds=3600)
        return heatmap_id

    def get(self, heatmap_id: str) -> bytes | None:
        return self._images.get(heatmap_id)

    def _prune(self, *, max_age_seconds: int) -> None:
        now = time.time()
        expired = [
            heatmap_id
            for heatmap_id, created_at in self._created_at.items()
            if now - created_at > max_age_seconds
        ]
        for heatmap_id in expired:
            self._created_at.pop(heatmap_id, None)
            self._images.pop(heatmap_id, None)


app = FastAPI(title="topo-layout Mapbox Integration")
_tile_cache = _TileCache()
_heatmap_store = _HeatmapStore()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/config")
def config() -> JSONResponse:
    public_token = os.getenv("MAPBOX_PUBLIC_TOKEN") or os.getenv("MAPBOX_ACCESS_TOKEN")
    if not public_token:
        raise HTTPException(
            status_code=500,
            detail="MAPBOX_ACCESS_TOKEN or MAPBOX_PUBLIC_TOKEN must be set",
        )
    return JSONResponse(
        {
            "mapboxPublicToken": public_token,
            "defaultCenter": {"lon": 34.99, "lat": 32.80},
            "defaultZoom": 12,
        }
    )


@app.post("/api/heatmap/mapbox")
def create_heatmap(request: MapboxHeatmapRequest) -> JSONResponse:
    access_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    if not access_token:
        raise HTTPException(status_code=500, detail="MAPBOX_ACCESS_TOKEN must be set on the backend")

    _validate_request_bounds(request)
    max_hours = _resolve_max_hours(request)
    terrain_zoom = _choose_terrain_zoom(request.map_zoom)
    dem = _build_dem_from_mapbox_terrain(request, terrain_zoom, access_token)
    terrain_influence = TerrainInfluenceConfig(
        steep_weight=request.steep_weight,
        cliff_like_weight=request.cliff_like_weight,
        valley_weight=request.valley_weight,
        ridge_weight=request.ridge_weight,
    )
    start_x, start_y = _lonlat_to_local_xy(
        lon=request.last_known_lon,
        lat=request.last_known_lat,
        west=request.west,
        south=request.south,
        east=request.east,
        north=request.north,
    )
    heatmap = compute_heatmap(
        dem,
        start_x=start_x,
        start_y=start_y,
        max_hours=max_hours,
        probability_method=request.probability_method,
        use_terrain_influence=request.use_terrain_influence,
        terrain_influence=terrain_influence,
        terrain_steep_threshold_deg=request.terrain_steep_threshold_deg,
        terrain_cliff_threshold_deg=request.terrain_cliff_threshold_deg,
        terrain_neighborhood_size=request.terrain_neighborhood_size,
        terrain_ridge_threshold_m=request.terrain_ridge_threshold_m,
        terrain_valley_threshold_m=request.terrain_valley_threshold_m,
    )
    image = render_heatmap_image(heatmap, layer="probability_color")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    heatmap_id = _heatmap_store.put(image_buffer.getvalue())
    reachable_cells = int(np.count_nonzero(heatmap.probability > 0))

    return JSONResponse(
        {
            "heatmapImageUrl": f"/api/heatmaps/{heatmap_id}.png",
            "imageCoordinates": [
                [request.west, request.north],
                [request.east, request.north],
                [request.east, request.south],
                [request.west, request.south],
            ],
            "stats": {
                "terrainZoom": terrain_zoom,
                "gridWidth": request.grid_width,
                "gridHeight": request.grid_height,
                "reachableCells": reachable_cells,
                "elapsedHours": max_hours,
                "probabilityMax": float(np.max(heatmap.probability)),
            },
        }
    )


@app.get("/api/heatmaps/{heatmap_id}.png")
def get_heatmap_image(heatmap_id: str) -> Response:
    image_bytes = _heatmap_store.get(heatmap_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Heatmap image not found")
    return Response(content=image_bytes, media_type="image/png")


def _validate_request_bounds(request: MapboxHeatmapRequest) -> None:
    if request.west >= request.east:
        raise HTTPException(status_code=400, detail="west must be smaller than east")
    if request.south >= request.north:
        raise HTTPException(status_code=400, detail="south must be smaller than north")
    if not (request.west <= request.last_known_lon <= request.east):
        raise HTTPException(status_code=400, detail="last known longitude must be inside the bbox")
    if not (request.south <= request.last_known_lat <= request.north):
        raise HTTPException(status_code=400, detail="last known latitude must be inside the bbox")
    if request.last_known_timestamp is None and request.max_hours is None:
        raise HTTPException(
            status_code=400,
            detail="Either last_known_timestamp or max_hours must be provided",
        )


def _resolve_max_hours(request: MapboxHeatmapRequest) -> float:
    if request.last_known_timestamp:
        return _elapsed_hours_since(request.last_known_timestamp)
    assert request.max_hours is not None
    return float(request.max_hours)


def _elapsed_hours_since(timestamp_text: str) -> float:
    parsed = _parse_iso_timestamp(timestamp_text)
    now = datetime.now(timezone.utc)
    elapsed_seconds = (now - parsed).total_seconds()
    if elapsed_seconds < 0:
        raise HTTPException(status_code=400, detail="last known timestamp cannot be in the future")
    return max(elapsed_seconds / 3600.0, 1e-6)


def _parse_iso_timestamp(timestamp_text: str) -> datetime:
    normalized = timestamp_text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="last known timestamp must be a valid ISO 8601 datetime",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _choose_terrain_zoom(map_zoom: float) -> int:
    return max(8, min(15, int(round(map_zoom))))


def _build_dem_from_mapbox_terrain(
    request: MapboxHeatmapRequest,
    terrain_zoom: int,
    access_token: str,
) -> DemRaster:
    width_m, height_m = _lonlat_extent_to_local_meters(
        west=request.west,
        south=request.south,
        east=request.east,
        north=request.north,
    )
    rows = np.linspace(request.north, request.south, request.grid_height, endpoint=False)
    cols = np.linspace(request.west, request.east, request.grid_width, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(cols, rows)
    lat_grid += ((request.north - request.south) / request.grid_height) / 2.0
    lon_grid += ((request.east - request.west) / request.grid_width) / 2.0

    x_float, y_float = _lonlat_to_tile_xy_arrays(lon_grid, lat_grid, terrain_zoom)
    tile_x = np.floor(x_float).astype(np.int32, copy=False)
    tile_y = np.floor(y_float).astype(np.int32, copy=False)
    pixel_x = np.clip(
        ((x_float - tile_x) * 256.0).astype(np.int32, copy=False),
        0,
        255,
    )
    pixel_y = np.clip(
        ((y_float - tile_y) * 256.0).astype(np.int32, copy=False),
        0,
        255,
    )

    elevation = np.zeros((request.grid_height, request.grid_width), dtype=np.float32)
    tile_pairs = np.stack((tile_x.reshape(-1), tile_y.reshape(-1)), axis=1)
    unique_tile_pairs = np.unique(tile_pairs, axis=0)

    for current_tile_x, current_tile_y in unique_tile_pairs:
        tile = _tile_cache.get(terrain_zoom, int(current_tile_x), int(current_tile_y), access_token)
        mask = (tile_x == current_tile_x) & (tile_y == current_tile_y)
        selected_pixel_x = pixel_x[mask]
        selected_pixel_y = pixel_y[mask]
        rgb = tile.rgb[selected_pixel_y, selected_pixel_x, :3].astype(np.int32, copy=False)
        elevation[mask] = -10000.0 + (
            (
                rgb[:, 0] * 256 * 256
                + rgb[:, 1] * 256
                + rgb[:, 2]
            ) * 0.1
        )

    return DemRaster(
        elevation=elevation,
        georef=GeoReference(
            crs="LOCAL_METERS_FROM_MAPBOX_BBOX",
            min_x=0.0,
            min_y=0.0,
            max_x=width_m,
            max_y=height_m,
        ),
        elevation_min_m=float(np.min(elevation)),
        elevation_max_m=float(np.max(elevation)),
        source_image=f"{TERRAIN_TILESET_ID}@z{terrain_zoom}",
        source_dem_path=None,
    )


def _sample_terrain_elevation(
    *,
    lon: float,
    lat: float,
    zoom: int,
    access_token: str,
) -> float:
    x_float, y_float = _lonlat_to_tile_xy(lon, lat, zoom)
    tile_x = int(math.floor(x_float))
    tile_y = int(math.floor(y_float))
    tile = _tile_cache.get(zoom, tile_x, tile_y, access_token)
    pixel_x = min(tile.width - 1, max(0, int((x_float - tile_x) * tile.width)))
    pixel_y = min(tile.height - 1, max(0, int((y_float - tile_y) * tile.height)))
    red, green, blue = tile.rgb[pixel_y, pixel_x, :3]
    return float(-10000.0 + ((int(red) * 256 * 256 + int(green) * 256 + int(blue)) * 0.1))


def _fetch_terrain_tile(zoom: int, x: int, y: int, access_token: str) -> _TileImage:
    url = (
        f"https://api.mapbox.com/v4/{TERRAIN_TILESET_ID}/{zoom}/{x}/{y}.pngraw"
        f"?access_token={access_token}"
    )
    try:
        with urlopen(url, timeout=20) as response:
            content = response.read()
    except HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Mapbox tile request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Mapbox tile request failed: {exc.reason}") from exc

    image = Image.open(io.BytesIO(content)).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    return _TileImage(rgb=rgb, width=image.width, height=image.height)


def _lonlat_to_tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _lonlat_to_tile_xy_arrays(
    lon: np.ndarray,
    lat: np.ndarray,
    zoom: int,
) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.radians(lat)
    n = float(2**zoom)
    x = ((lon + 180.0) / 360.0) * n
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n
    return x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)


def _lonlat_extent_to_local_meters(
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float]:
    center_lat_rad = math.radians((south + north) / 2.0)
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(center_lat_rad)
    width_m = (east - west) * meters_per_degree_lon
    height_m = (north - south) * meters_per_degree_lat
    return width_m, height_m


def _lonlat_to_local_xy(
    *,
    lon: float,
    lat: float,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float]:
    width_m, height_m = _lonlat_extent_to_local_meters(
        west=west,
        south=south,
        east=east,
        north=north,
    )
    x = ((lon - west) / (east - west)) * width_m
    y = ((lat - south) / (north - south)) * height_m
    return x, y


def build_app() -> FastAPI:
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("fastapi is required to build the live Mapbox app")
    return app
