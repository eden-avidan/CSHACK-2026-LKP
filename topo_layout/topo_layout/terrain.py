from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from .preprocessing import DemRaster, GeoReference, compute_slope


TerrainLayer = Literal["bitmask", "steep", "cliff_like", "valley", "ridge", "tpi"]

TERRAIN_BIT_STEEP = 1
TERRAIN_BIT_CLIFF_LIKE = 2
TERRAIN_BIT_VALLEY = 4
TERRAIN_BIT_RIDGE = 8


@dataclass(slots=True)
class TerrainRaster:
    bitmask: np.ndarray
    steep_mask: np.ndarray
    cliff_like_mask: np.ndarray
    valley_mask: np.ndarray
    ridge_mask: np.ndarray
    topographic_position_index: np.ndarray
    georef: GeoReference
    source_dem: str
    steep_threshold_deg: float
    cliff_threshold_deg: float
    neighborhood_size: int
    ridge_threshold_m: float
    valley_threshold_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.bitmask.shape

    def metadata(self) -> dict:
        height, width = self.shape
        return {
            "source_dem": self.source_dem,
            "crs": self.georef.crs,
            "bounds": {
                "min_x": self.georef.min_x,
                "min_y": self.georef.min_y,
                "max_x": self.georef.max_x,
                "max_y": self.georef.max_y,
            },
            "raster": {
                "width": width,
                "height": height,
                "pixel_size_x": self.georef.pixel_size_x(width),
                "pixel_size_y": self.georef.pixel_size_y(height),
            },
            "thresholds": {
                "steep_degrees": self.steep_threshold_deg,
                "cliff_like_degrees": self.cliff_threshold_deg,
                "ridge_tpi_meters": self.ridge_threshold_m,
                "valley_tpi_meters": self.valley_threshold_m,
                "neighborhood_size": self.neighborhood_size,
            },
            "layers": {
                "bitmask": {
                    "description": "Combined terrain flags stored as a bitmask.",
                    "dtype": str(self.bitmask.dtype),
                    "bits": {
                        str(TERRAIN_BIT_STEEP): "steep",
                        str(TERRAIN_BIT_CLIFF_LIKE): "cliff_like",
                        str(TERRAIN_BIT_VALLEY): "valley",
                        str(TERRAIN_BIT_RIDGE): "ridge",
                    },
                },
                "steep": {
                    "description": "Cells whose slope is above the steep threshold.",
                    "dtype": str(self.steep_mask.dtype),
                },
                "cliff_like": {
                    "description": "Cells whose slope is above the cliff-like threshold.",
                    "dtype": str(self.cliff_like_mask.dtype),
                },
                "valley": {
                    "description": "Cells lower than their local neighborhood by at least the valley threshold.",
                    "dtype": str(self.valley_mask.dtype),
                },
                "ridge": {
                    "description": "Cells higher than their local neighborhood by at least the ridge threshold.",
                    "dtype": str(self.ridge_mask.dtype),
                },
                "tpi": {
                    "description": "Topographic Position Index in meters: cell elevation minus local neighbor mean.",
                    "dtype": str(self.topographic_position_index.dtype),
                },
            },
        }


def classify_terrain(
    dem: DemRaster,
    *,
    steep_threshold_deg: float = 30.0,
    cliff_threshold_deg: float = 45.0,
    neighborhood_size: int = 3,
    ridge_threshold_m: float = 5.0,
    valley_threshold_m: float = 5.0,
) -> TerrainRaster:
    """
    Classify terrain features that can be derived directly from a DEM.

    - steep / cliff-like come from slope thresholds
    - ridge / valley come from local topographic position
    """
    if steep_threshold_deg < 0 or cliff_threshold_deg < 0:
        raise ValueError("Slope thresholds must be non-negative")
    if cliff_threshold_deg < steep_threshold_deg:
        raise ValueError("cliff_threshold_deg must be greater than or equal to steep_threshold_deg")
    if neighborhood_size < 3 or neighborhood_size % 2 == 0:
        raise ValueError("neighborhood_size must be an odd integer greater than or equal to 3")
    if ridge_threshold_m < 0 or valley_threshold_m < 0:
        raise ValueError("TPI thresholds must be non-negative")

    slope = compute_slope(dem)
    steep_mask = slope.slope_degrees >= steep_threshold_deg
    cliff_like_mask = slope.slope_degrees >= cliff_threshold_deg
    tpi = _compute_topographic_position_index(
        dem.elevation.astype(np.float32, copy=False),
        neighborhood_size=neighborhood_size,
    )
    ridge_mask = tpi >= ridge_threshold_m
    valley_mask = tpi <= -valley_threshold_m

    bitmask = np.zeros(dem.shape, dtype=np.uint8)
    bitmask[steep_mask] |= TERRAIN_BIT_STEEP
    bitmask[cliff_like_mask] |= TERRAIN_BIT_CLIFF_LIKE
    bitmask[valley_mask] |= TERRAIN_BIT_VALLEY
    bitmask[ridge_mask] |= TERRAIN_BIT_RIDGE

    return TerrainRaster(
        bitmask=bitmask,
        steep_mask=steep_mask.astype(np.uint8, copy=False),
        cliff_like_mask=cliff_like_mask.astype(np.uint8, copy=False),
        valley_mask=valley_mask.astype(np.uint8, copy=False),
        ridge_mask=ridge_mask.astype(np.uint8, copy=False),
        topographic_position_index=tpi.astype(np.float32, copy=False),
        georef=dem.georef,
        source_dem=dem.source_dem_path or dem.source_image,
        steep_threshold_deg=steep_threshold_deg,
        cliff_threshold_deg=cliff_threshold_deg,
        neighborhood_size=neighborhood_size,
        ridge_threshold_m=ridge_threshold_m,
        valley_threshold_m=valley_threshold_m,
    )


def save_terrain_tiff(
    terrain: TerrainRaster,
    output_path: str | Path,
    *,
    layer: TerrainLayer = "bitmask",
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if layer == "bitmask":
        array = terrain.bitmask.astype(np.uint8, copy=False)
        image = Image.fromarray(array, mode="L")
    elif layer == "steep":
        array = terrain.steep_mask.astype(np.uint8, copy=False) * 255
        image = Image.fromarray(array, mode="L")
    elif layer == "cliff_like":
        array = terrain.cliff_like_mask.astype(np.uint8, copy=False) * 255
        image = Image.fromarray(array, mode="L")
    elif layer == "valley":
        array = terrain.valley_mask.astype(np.uint8, copy=False) * 255
        image = Image.fromarray(array, mode="L")
    elif layer == "ridge":
        array = terrain.ridge_mask.astype(np.uint8, copy=False) * 255
        image = Image.fromarray(array, mode="L")
    elif layer == "tpi":
        array = terrain.topographic_position_index.astype(np.float32, copy=False)
        image = Image.fromarray(array, mode="F")
    else:
        raise ValueError(f"Unsupported terrain layer: {layer}")

    image.save(output_path, format="TIFF")

    metadata = terrain.metadata()
    metadata["saved_layer"] = layer
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, sidecar_path


def _compute_topographic_position_index(
    elevation: np.ndarray,
    *,
    neighborhood_size: int,
) -> np.ndarray:
    pad = neighborhood_size // 2
    padded = np.pad(elevation, pad_width=pad, mode="edge")
    windows = sliding_window_view(padded, (neighborhood_size, neighborhood_size))
    neighborhood_sum = windows.sum(axis=(-1, -2), dtype=np.float32) - elevation
    neighbor_count = (neighborhood_size * neighborhood_size) - 1
    neighbor_mean = neighborhood_sum / neighbor_count
    return elevation - neighbor_mean
