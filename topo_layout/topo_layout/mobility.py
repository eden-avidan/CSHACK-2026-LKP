from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .preprocessing import DemRaster, GeoReference, convert_image_to_dem
from .terrain import TerrainRaster, classify_terrain


ProbabilityMethod = Literal["linear", "exponential"]
HeatmapLayer = Literal["probability", "travel_time_hours", "probability_color", "travel_time_color"]


@dataclass(slots=True)
class TerrainInfluenceConfig:
    steep_weight: float = 0.7
    cliff_like_weight: float = 0.2
    valley_weight: float = 1.15
    ridge_weight: float = 0.9

    def metadata(self) -> dict:
        return {
            "steep_weight": self.steep_weight,
            "cliff_like_weight": self.cliff_like_weight,
            "valley_weight": self.valley_weight,
            "ridge_weight": self.ridge_weight,
        }


@dataclass(slots=True)
class HeatmapRaster:
    probability: np.ndarray
    travel_time_hours: np.ndarray
    georef: GeoReference
    source_dem: str
    start_row: int
    start_col: int
    start_x: float
    start_y: float
    max_hours: float
    probability_method: ProbabilityMethod
    terrain_influence_enabled: bool
    terrain_influence_config: TerrainInfluenceConfig | None = None
    terrain_thresholds: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.probability.shape

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
            "start_point": {
                "row": self.start_row,
                "col": self.start_col,
                "x": self.start_x,
                "y": self.start_y,
            },
            "time_horizon_hours": self.max_hours,
            "probability_method": self.probability_method,
            "terrain_influence": {
                "enabled": self.terrain_influence_enabled,
                "weights": (
                    None if self.terrain_influence_config is None else self.terrain_influence_config.metadata()
                ),
                "classification_thresholds": self.terrain_thresholds,
            },
            "layers": {
                "probability": {
                    "description": (
                        "Normalized probability-like heatmap weights derived from travel time "
                        "and optional terrain-class weighting."
                    ),
                    "dtype": str(self.probability.dtype),
                },
                "travel_time_hours": {
                    "description": "Least travel time from the last known point to each cell, in hours.",
                    "dtype": str(self.travel_time_hours.dtype),
                },
            },
        }


def tobler_hiking_speed_kmh(signed_slope_grade: np.ndarray | float) -> np.ndarray | float:
    """
    Tobler's hiking function.

    signed_slope_grade is rise/run in the direction of travel.
    """
    return 6.0 * np.exp(-3.5 * np.abs(np.asarray(signed_slope_grade) + 0.05))


def coordinate_to_row_col(
    georef: GeoReference,
    shape: tuple[int, int],
    *,
    x: float,
    y: float,
) -> tuple[int, int]:
    height, width = shape
    pixel_size_x = georef.pixel_size_x(width)
    pixel_size_y = georef.pixel_size_y(height)

    if not (georef.min_x <= x <= georef.max_x and georef.min_y <= y <= georef.max_y):
        raise ValueError("The last known point is outside the DEM bounds")

    col = min(width - 1, max(0, int((x - georef.min_x) / pixel_size_x)))
    row = min(height - 1, max(0, int((georef.max_y - y) / pixel_size_y)))
    return row, col


def row_col_to_coordinate(
    georef: GeoReference,
    shape: tuple[int, int],
    *,
    row: int,
    col: int,
) -> tuple[float, float]:
    height, width = shape
    pixel_size_x = georef.pixel_size_x(width)
    pixel_size_y = georef.pixel_size_y(height)
    x = georef.min_x + (col + 0.5) * pixel_size_x
    y = georef.max_y - (row + 0.5) * pixel_size_y
    return x, y


def compute_heatmap(
    dem: DemRaster,
    *,
    start_x: float,
    start_y: float,
    max_hours: float,
    probability_method: ProbabilityMethod = "linear",
    terrain: TerrainRaster | None = None,
    use_terrain_influence: bool = True,
    terrain_influence: TerrainInfluenceConfig | None = None,
    terrain_steep_threshold_deg: float = 30.0,
    terrain_cliff_threshold_deg: float = 45.0,
    terrain_neighborhood_size: int = 3,
    terrain_ridge_threshold_m: float = 5.0,
    terrain_valley_threshold_m: float = 5.0,
) -> HeatmapRaster:
    if max_hours <= 0:
        raise ValueError("max_hours must be greater than 0")

    terrain_influence = terrain_influence or TerrainInfluenceConfig()
    start_row, start_col = coordinate_to_row_col(
        dem.georef,
        dem.shape,
        x=start_x,
        y=start_y,
    )
    travel_time_hours = _least_travel_time_hours(
        dem,
        start_row=start_row,
        start_col=start_col,
        max_hours=max_hours,
    )
    terrain_thresholds = None
    terrain_weight = None
    if use_terrain_influence:
        if terrain is None:
            terrain = classify_terrain(
                dem,
                steep_threshold_deg=terrain_steep_threshold_deg,
                cliff_threshold_deg=terrain_cliff_threshold_deg,
                neighborhood_size=terrain_neighborhood_size,
                ridge_threshold_m=terrain_ridge_threshold_m,
                valley_threshold_m=terrain_valley_threshold_m,
            )
        terrain_weight = _terrain_probability_weight(terrain, terrain_influence)
        terrain_thresholds = {
            "steep_degrees": terrain.steep_threshold_deg,
            "cliff_like_degrees": terrain.cliff_threshold_deg,
            "neighborhood_size": terrain.neighborhood_size,
            "ridge_tpi_meters": terrain.ridge_threshold_m,
            "valley_tpi_meters": terrain.valley_threshold_m,
        }

    probability = _travel_time_to_probability(
        travel_time_hours,
        max_hours=max_hours,
        method=probability_method,
        terrain_weight=terrain_weight,
    )

    return HeatmapRaster(
        probability=probability,
        travel_time_hours=travel_time_hours,
        georef=dem.georef,
        source_dem=dem.source_dem_path or dem.source_image,
        start_row=start_row,
        start_col=start_col,
        start_x=start_x,
        start_y=start_y,
        max_hours=max_hours,
        probability_method=probability_method,
        terrain_influence_enabled=use_terrain_influence,
        terrain_influence_config=terrain_influence if use_terrain_influence else None,
        terrain_thresholds=terrain_thresholds,
    )


def compute_heatmap_from_heightmap(
    image_path: str | Path,
    georef: GeoReference,
    *,
    start_x: float,
    start_y: float,
    max_hours: float,
    elevation_min_m: float,
    elevation_max_m: float,
    probability_method: ProbabilityMethod = "linear",
    terrain: TerrainRaster | None = None,
    use_terrain_influence: bool = True,
    terrain_influence: TerrainInfluenceConfig | None = None,
    terrain_steep_threshold_deg: float = 30.0,
    terrain_cliff_threshold_deg: float = 45.0,
    terrain_neighborhood_size: int = 3,
    terrain_ridge_threshold_m: float = 5.0,
    terrain_valley_threshold_m: float = 5.0,
) -> tuple[DemRaster, HeatmapRaster]:
    dem = convert_image_to_dem(
        image_path=image_path,
        georef=georef,
        mode="heightmap",
        elevation_min_m=elevation_min_m,
        elevation_max_m=elevation_max_m,
    )
    heatmap = compute_heatmap(
        dem,
        start_x=start_x,
        start_y=start_y,
        max_hours=max_hours,
        probability_method=probability_method,
        terrain=terrain,
        use_terrain_influence=use_terrain_influence,
        terrain_influence=terrain_influence,
        terrain_steep_threshold_deg=terrain_steep_threshold_deg,
        terrain_cliff_threshold_deg=terrain_cliff_threshold_deg,
        terrain_neighborhood_size=terrain_neighborhood_size,
        terrain_ridge_threshold_m=terrain_ridge_threshold_m,
        terrain_valley_threshold_m=terrain_valley_threshold_m,
    )
    return dem, heatmap


def save_heatmap_tiff(
    heatmap: HeatmapRaster,
    output_path: str | Path,
    *,
    layer: HeatmapLayer = "probability",
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_heatmap_image(heatmap, layer=layer)
    image.save(output_path)

    metadata = heatmap.metadata()
    metadata["saved_layer"] = layer
    if layer.endswith("_color"):
        metadata["render"] = {
            "palette": "cool-to-hot",
            "contrast_stretch": "2nd to 98th percentile of valid values",
        }
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, sidecar_path


def render_heatmap_image(
    heatmap: HeatmapRaster,
    *,
    layer: HeatmapLayer = "probability_color",
) -> Image.Image:
    if layer == "probability":
        array = heatmap.probability.astype(np.float32, copy=False)
        return Image.fromarray(array, mode="F")
    if layer == "travel_time_hours":
        array = heatmap.travel_time_hours.astype(np.float32, copy=True)
        array[~np.isfinite(array)] = np.nan
        return Image.fromarray(array, mode="F")
    if layer == "probability_color":
        return Image.fromarray(_colorize_heat_array(heatmap.probability), mode="RGB")
    if layer == "travel_time_color":
        return Image.fromarray(_colorize_heat_array(heatmap.travel_time_hours, finite_only=True), mode="RGB")
    raise ValueError(f"Unsupported heatmap layer: {layer}")


def _least_travel_time_hours(
    dem: DemRaster,
    *,
    start_row: int,
    start_col: int,
    max_hours: float,
) -> np.ndarray:
    height, width = dem.shape
    pixel_size_x = dem.georef.pixel_size_x(width)
    pixel_size_y = dem.georef.pixel_size_y(height)
    elevation = dem.elevation.astype(np.float32, copy=False)

    travel_time = np.full((height, width), np.inf, dtype=np.float32)
    travel_time[start_row, start_col] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, start_row, start_col)]

    neighbor_offsets = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    while heap:
        current_time, row, col = heapq.heappop(heap)
        if current_time > float(travel_time[row, col]):
            continue
        if current_time > max_hours:
            continue

        for delta_row, delta_col in neighbor_offsets:
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue

            horizontal_distance_m = float(
                np.hypot(delta_col * pixel_size_x, delta_row * pixel_size_y)
            )
            elevation_change_m = float(elevation[next_row, next_col] - elevation[row, col])
            signed_slope_grade = elevation_change_m / horizontal_distance_m
            speed_kmh = float(tobler_hiking_speed_kmh(signed_slope_grade))
            speed_kmh = max(speed_kmh, 1e-6)
            step_time_hours = (horizontal_distance_m / 1000.0) / speed_kmh
            next_time = current_time + step_time_hours

            if next_time < float(travel_time[next_row, next_col]) and next_time <= max_hours:
                travel_time[next_row, next_col] = next_time
                heapq.heappush(heap, (next_time, next_row, next_col))

    return travel_time


def _travel_time_to_probability(
    travel_time_hours: np.ndarray,
    *,
    max_hours: float,
    method: ProbabilityMethod,
    terrain_weight: np.ndarray | None = None,
) -> np.ndarray:
    reachable = np.isfinite(travel_time_hours) & (travel_time_hours <= max_hours)
    probability = np.zeros_like(travel_time_hours, dtype=np.float32)

    if method == "linear":
        probability[reachable] = (
            (max_hours - travel_time_hours[reachable]) / max_hours
        ).astype(np.float32, copy=False)
    elif method == "exponential":
        decay_hours = max_hours / 3.0
        probability[reachable] = np.exp(-travel_time_hours[reachable] / decay_hours).astype(
            np.float32,
            copy=False,
        )
    else:
        raise ValueError(f"Unsupported probability method: {method}")

    if terrain_weight is not None:
        probability[reachable] *= terrain_weight[reachable]

    total = float(np.sum(probability))
    if total > 0:
        probability /= total
    return probability


def _terrain_probability_weight(
    terrain: TerrainRaster,
    config: TerrainInfluenceConfig,
) -> np.ndarray:
    if min(
        config.steep_weight,
        config.cliff_like_weight,
        config.valley_weight,
        config.ridge_weight,
    ) < 0:
        raise ValueError("Terrain influence weights must be non-negative")

    weight = np.ones(terrain.shape, dtype=np.float32)
    weight *= np.where(terrain.steep_mask > 0, config.steep_weight, 1.0)
    weight *= np.where(terrain.cliff_like_mask > 0, config.cliff_like_weight, 1.0)
    weight *= np.where(terrain.valley_mask > 0, config.valley_weight, 1.0)
    weight *= np.where(terrain.ridge_mask > 0, config.ridge_weight, 1.0)
    return weight


def _colorize_heat_array(
    array: np.ndarray,
    *,
    positive_only: bool = False,
    finite_only: bool = False,
) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(data)
    if positive_only:
        valid &= data > 0
    if finite_only:
        valid &= np.isfinite(data)

    rgb = np.zeros(data.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    values = data[valid]
    lo = float(np.quantile(values, 0.02))
    hi = float(np.quantile(values, 0.98))
    if hi <= lo:
        hi = float(np.max(values))
        lo = float(np.min(values))
    scale = max(hi - lo, 1e-12)
    normalized = np.clip((data - lo) / scale, 0.0, 1.0)
    palette_points = np.array(
        [
            [0.0, 0, 0, 255],
            [0.2, 0, 180, 255],
            [0.4, 0, 180, 255],
            [0.6, 255, 255, 0],
            [0.8, 255, 120, 0],
            [1.0, 255, 0, 0],
        ],
        dtype=np.float32,
    )

    flat = normalized.reshape(-1)
    channels = []
    for channel_index in range(1, 4):
        channels.append(
            np.interp(
                flat,
                palette_points[:, 0],
                palette_points[:, channel_index],
            )
        )

    colored = np.stack(channels, axis=1).reshape(data.shape + (3,))
    rgb[valid] = colored[valid].astype(np.uint8)
    return rgb
