from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


ImageMode = Literal["heightmap", "scanned_topo_map"]


class UnsupportedTopographicImageError(ValueError):
    """Raised when the input image cannot be safely interpreted as elevation."""


@dataclass(slots=True)
class GeoReference:
    crs: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def span_x(self) -> float:
        return self.max_x - self.min_x

    @property
    def span_y(self) -> float:
        return self.max_y - self.min_y

    def pixel_size_x(self, width: int) -> float:
        if width <= 0:
            raise ValueError("width must be greater than 0")
        return self.span_x / width

    def pixel_size_y(self, height: int) -> float:
        if height <= 0:
            raise ValueError("height must be greater than 0")
        return self.span_y / height


@dataclass(slots=True)
class DemRaster:
    elevation: np.ndarray
    georef: GeoReference
    elevation_min_m: float
    elevation_max_m: float
    source_image: str
    source_dem_path: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def metadata(self) -> dict:
        height, width = self.shape
        return {
            "source_image": self.source_image,
            "source_dem_path": self.source_dem_path,
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
            "elevation_meters": {
                "min": self.elevation_min_m,
                "max": self.elevation_max_m,
            },
            "dtype": str(self.elevation.dtype),
        }


@dataclass(slots=True)
class SlopeRaster:
    slope_grade: np.ndarray
    slope_degrees: np.ndarray
    georef: GeoReference
    source_dem: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.slope_grade.shape

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
            "layers": {
                "slope_grade": {
                    "description": "Rise over run. Example: 0.5 means 50 percent grade.",
                    "dtype": str(self.slope_grade.dtype),
                },
                "slope_degrees": {
                    "description": "Slope angle in degrees.",
                    "dtype": str(self.slope_degrees.dtype),
                },
            },
        }


def convert_image_to_dem(
    image_path: str | Path,
    georef: GeoReference,
    *,
    mode: ImageMode = "heightmap",
    elevation_min_m: float = 0.0,
    elevation_max_m: float = 1000.0,
) -> DemRaster:
    """
    Convert a topo image into a DEM raster.

    Supported now:
    - heightmap images where brightness corresponds to elevation

    Explicitly unsupported:
    - scanned topographic maps with contour lines and labels
    """
    image_path = Path(image_path)
    if mode == "scanned_topo_map":
        raise UnsupportedTopographicImageError(
            "Scanned topographic maps cannot be converted directly into a DEM. "
            "They need a separate contour extraction and interpolation pipeline."
        )

    if elevation_max_m <= elevation_min_m:
        raise ValueError("elevation_max_m must be greater than elevation_min_m")

    image = Image.open(image_path)
    grayscale = _to_grayscale_array(image)
    elevation = _scale_grayscale_to_elevation(
        grayscale,
        elevation_min_m=elevation_min_m,
        elevation_max_m=elevation_max_m,
    )

    return DemRaster(
        elevation=elevation,
        georef=georef,
        elevation_min_m=elevation_min_m,
        elevation_max_m=elevation_max_m,
        source_image=str(image_path),
        source_dem_path=None,
    )


def save_dem_tiff(dem: DemRaster, output_path: str | Path) -> tuple[Path, Path]:
    """
    Save the DEM raster to a TIFF and write geospatial metadata as a JSON sidecar.

    This keeps the base project lightweight and avoids forcing a GIS stack early.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = dem.elevation.astype(np.float32, copy=False)
    image = Image.fromarray(array, mode="F")
    image.save(output_path, format="TIFF")

    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(dem.metadata(), indent=2), encoding="utf-8")
    return output_path, sidecar_path


def import_worldfile_dem(
    input_glob: str,
    *,
    crs: str,
    shift_to_origin: bool = False,
) -> DemRaster:
    """
    Import one or more TIFF tiles described by adjacent TFW world files.

    If shift_to_origin is True, the imported DEM bounds are shifted so the
    lower-left corner becomes (0, 0). This is convenient for local testing.
    """
    tif_paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not tif_paths:
        raise FileNotFoundError(f"No TIFF files matched input_glob: {input_glob}")

    tiles = [_load_worldfile_tile(path) for path in tif_paths]
    pixel_size_x = tiles[0]["pixel_size_x"]
    pixel_size_y = tiles[0]["pixel_size_y"]

    for tile in tiles[1:]:
        if abs(tile["pixel_size_x"] - pixel_size_x) > 1e-6:
            raise ValueError("All tiles must use the same pixel_size_x")
        if abs(tile["pixel_size_y"] - pixel_size_y) > 1e-6:
            raise ValueError("All tiles must use the same pixel_size_y")

    global_min_x = min(tile["min_x"] for tile in tiles)
    global_min_y = min(tile["min_y"] for tile in tiles)
    global_max_x = max(tile["max_x"] for tile in tiles)
    global_max_y = max(tile["max_y"] for tile in tiles)
    width = int(round((global_max_x - global_min_x) / pixel_size_x))
    height = int(round((global_max_y - global_min_y) / pixel_size_y))
    mosaic = np.zeros((height, width), dtype=np.float32)

    for tile in tiles:
        row_start = int(round((global_max_y - tile["max_y"]) / pixel_size_y))
        col_start = int(round((tile["min_x"] - global_min_x) / pixel_size_x))
        tile_array = tile["array"]
        tile_height, tile_width = tile_array.shape
        mosaic[row_start : row_start + tile_height, col_start : col_start + tile_width] = tile_array

    if shift_to_origin:
        georef = GeoReference(
            crs=crs,
            min_x=0.0,
            min_y=0.0,
            max_x=global_max_x - global_min_x,
            max_y=global_max_y - global_min_y,
        )
    else:
        georef = GeoReference(
            crs=crs,
            min_x=global_min_x,
            min_y=global_min_y,
            max_x=global_max_x,
            max_y=global_max_y,
        )

    return DemRaster(
        elevation=mosaic,
        georef=georef,
        elevation_min_m=float(np.min(mosaic)),
        elevation_max_m=float(np.max(mosaic)),
        source_image=input_glob,
        source_dem_path=None,
    )


def import_geotiff_dem(
    input_path: str | Path,
    *,
    shift_to_origin: bool = False,
    convert_geographic_to_local_meters: bool = True,
) -> DemRaster:
    """
    Import a GeoTIFF DEM using embedded georeferencing tags.

    If the GeoTIFF is in a geographic CRS like EPSG:4326 and
    convert_geographic_to_local_meters is True, the bounds are converted into
    a local meter-based frame using an equirectangular approximation around the
    raster center. This keeps the downstream mobility model meaningful.
    """
    input_path = Path(input_path)
    image = Image.open(input_path)
    array = np.asarray(image, dtype=np.float32)
    width, height = image.size
    georef_info = _read_geotiff_georef(image)
    min_x = georef_info["min_x"]
    min_y = georef_info["min_y"]
    max_x = georef_info["max_x"]
    max_y = georef_info["max_y"]
    crs = georef_info["crs"]

    if convert_geographic_to_local_meters and crs.startswith("EPSG:4326"):
        min_x, min_y, max_x, max_y = _convert_lonlat_bounds_to_local_meters(
            min_lon=min_x,
            min_lat=min_y,
            max_lon=max_x,
            max_lat=max_y,
            shift_to_origin=shift_to_origin,
        )
        crs = "LOCAL_METERS_FROM_EPSG4326"
        shift_to_origin = False

    if shift_to_origin:
        georef = GeoReference(
            crs=crs,
            min_x=0.0,
            min_y=0.0,
            max_x=max_x - min_x,
            max_y=max_y - min_y,
        )
    else:
        georef = GeoReference(
            crs=crs,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )

    return DemRaster(
        elevation=array,
        georef=georef,
        elevation_min_m=float(np.min(array)),
        elevation_max_m=float(np.max(array)),
        source_image=str(input_path),
        source_dem_path=None,
    )


def load_dem_tiff(
    dem_path: str | Path,
    metadata_path: str | Path | None = None,
) -> DemRaster:
    dem_path = Path(dem_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else dem_path.with_suffix(dem_path.suffix + ".json")

    elevation = np.asarray(Image.open(dem_path), dtype=np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bounds = metadata["bounds"]
    elevation_metadata = metadata["elevation_meters"]

    return DemRaster(
        elevation=elevation,
        georef=GeoReference(
            crs=metadata["crs"],
            min_x=float(bounds["min_x"]),
            min_y=float(bounds["min_y"]),
            max_x=float(bounds["max_x"]),
            max_y=float(bounds["max_y"]),
        ),
        elevation_min_m=float(elevation_metadata["min"]),
        elevation_max_m=float(elevation_metadata["max"]),
        source_image=metadata["source_image"],
        source_dem_path=str(dem_path),
    )


def compute_slope(dem: DemRaster) -> SlopeRaster:
    """
    Compute terrain slope from a DEM raster.

    The result is returned in two representations:
    - slope_grade: rise/run
    - slope_degrees: angle in degrees
    """
    height, width = dem.shape
    pixel_size_x = dem.georef.pixel_size_x(width)
    pixel_size_y = dem.georef.pixel_size_y(height)

    dz_dy, dz_dx = np.gradient(
        dem.elevation.astype(np.float32, copy=False),
        pixel_size_y,
        pixel_size_x,
    )
    slope_grade = np.hypot(dz_dx, dz_dy).astype(np.float32, copy=False)
    slope_degrees = np.degrees(np.arctan(slope_grade)).astype(np.float32, copy=False)

    return SlopeRaster(
        slope_grade=slope_grade,
        slope_degrees=slope_degrees,
        georef=dem.georef,
        source_dem=dem.source_dem_path or dem.source_image,
    )


def save_slope_tiff(
    slope: SlopeRaster,
    output_path: str | Path,
    *,
    layer: Literal["grade", "degrees"] = "degrees",
) -> tuple[Path, Path]:
    """
    Save the slope raster to TIFF and write metadata as a JSON sidecar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if layer == "grade":
        array = slope.slope_grade.astype(np.float32, copy=False)
    else:
        array = slope.slope_degrees.astype(np.float32, copy=False)

    image = Image.fromarray(array, mode="F")
    image.save(output_path, format="TIFF")

    metadata = slope.metadata()
    metadata["saved_layer"] = layer
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, sidecar_path


def _to_grayscale_array(image: Image.Image) -> np.ndarray:
    if image.mode not in {"L", "I;16", "I", "F", "RGB", "RGBA"}:
        raise UnsupportedTopographicImageError(
            f"Unsupported image mode '{image.mode}'. Use grayscale or RGB/RGBA imagery."
        )

    grayscale = image.convert("L")
    return np.asarray(grayscale, dtype=np.float32)


def _scale_grayscale_to_elevation(
    grayscale: np.ndarray,
    *,
    elevation_min_m: float,
    elevation_max_m: float,
) -> np.ndarray:
    normalized = grayscale / 255.0
    elevation_range = elevation_max_m - elevation_min_m
    return elevation_min_m + (normalized * elevation_range)


def _read_geotiff_georef(image: Image.Image) -> dict:
    tags = image.tag_v2
    scale = tags.get(33550)
    tiepoint = tags.get(33922)
    geokeys = tags.get(34735)
    if scale is None or tiepoint is None:
        raise ValueError("GeoTIFF is missing required georeferencing tags")

    if len(scale) < 2 or len(tiepoint) < 6:
        raise ValueError("GeoTIFF georeferencing tags are malformed")

    pixel_size_x = float(scale[0])
    pixel_size_y = float(scale[1])
    tie_raster_x, tie_raster_y, _, tie_model_x, tie_model_y, _ = tiepoint[:6]
    if abs(float(tie_raster_x)) > 1e-9 or abs(float(tie_raster_y)) > 1e-9:
        raise ValueError("Only GeoTIFFs with origin tiepoints at raster (0,0) are supported")

    width, height = image.size
    min_x = float(tie_model_x)
    max_y = float(tie_model_y)
    max_x = min_x + (width * pixel_size_x)
    min_y = max_y - (height * pixel_size_y)

    crs = "UNKNOWN"
    if geokeys is not None:
        geokey_values = list(geokeys)
        if len(geokey_values) >= 4:
            number_of_keys = int(geokey_values[3])
            for index in range(number_of_keys):
                base = 4 + (index * 4)
                if base + 3 >= len(geokey_values):
                    break
                key_id, tiff_tag_location, count, value_offset = geokey_values[base : base + 4]
                if key_id == 2048 and tiff_tag_location == 0:
                    crs = f"EPSG:{int(value_offset)}"
                    break

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "crs": crs,
    }


def _load_worldfile_tile(tif_path: Path) -> dict:
    tfw_path = tif_path.with_suffix(".tfw")
    if not tfw_path.exists():
        raise FileNotFoundError(f"Missing world file for tile: {tif_path}")

    lines = [float(line.strip()) for line in tfw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 6:
        raise ValueError(f"World file must contain 6 lines: {tfw_path}")

    pixel_size_x, rotation_x, rotation_y, pixel_size_y_signed, upper_left_center_x, upper_left_center_y = lines
    if abs(rotation_x) > 1e-9 or abs(rotation_y) > 1e-9:
        raise ValueError(f"Rotated world files are not supported: {tfw_path}")

    array = np.asarray(Image.open(tif_path), dtype=np.float32)
    height, width = array.shape
    pixel_size_y = abs(pixel_size_y_signed)
    min_x = upper_left_center_x - (pixel_size_x / 2.0)
    max_y = upper_left_center_y + (pixel_size_y / 2.0)
    max_x = min_x + (width * pixel_size_x)
    min_y = max_y - (height * pixel_size_y)

    return {
        "path": str(tif_path),
        "array": array,
        "pixel_size_x": pixel_size_x,
        "pixel_size_y": pixel_size_y,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
    }


def _convert_lonlat_bounds_to_local_meters(
    *,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    shift_to_origin: bool,
) -> tuple[float, float, float, float]:
    center_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(center_lat_rad)

    width_m = (max_lon - min_lon) * meters_per_degree_lon
    height_m = (max_lat - min_lat) * meters_per_degree_lat
    if shift_to_origin:
        return 0.0, 0.0, width_m, height_m

    min_x = min_lon * meters_per_degree_lon
    max_x = max_lon * meters_per_degree_lon
    min_y = min_lat * meters_per_degree_lat
    max_y = max_lat * meters_per_degree_lat
    return min_x, min_y, max_x, max_y
