from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, shape

from app.geospatial.crs import CRSHelper
from app.models.heatmap import GridBounds, GridMetadata
from app.models.mission import LatLon


@dataclass
class ProbabilityGrid:
    probabilities: np.ndarray  # shape (rows, cols)
    metadata: GridMetadata
    crs: CRSHelper

    @property
    def rows(self) -> int:
        return self.probabilities.shape[0]

    @property
    def cols(self) -> int:
        return self.probabilities.shape[1]


def build_grid_metadata(
    lkp: LatLon,
    resolution_m: float,
    grid_size: int,
) -> tuple[GridMetadata, CRSHelper]:
    crs = CRSHelper(lkp.lat, lkp.lon)
    half = (grid_size * resolution_m) / 2.0

    corners = [
        crs.offset_to_wgs84(-half, -half),
        crs.offset_to_wgs84(half, -half),
        crs.offset_to_wgs84(half, half),
        crs.offset_to_wgs84(-half, half),
    ]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]

    metadata = GridMetadata(
        origin=lkp,
        resolution_m=resolution_m,
        rows=grid_size,
        cols=grid_size,
        crs_epsg=crs.epsg,
        bounds=GridBounds(
            north=max(lats),
            south=min(lats),
            east=max(lons),
            west=min(lons),
        ),
    )
    return metadata, crs


def create_empty_grid(lkp: LatLon, resolution_m: float, grid_size: int) -> ProbabilityGrid:
    metadata, crs = build_grid_metadata(lkp, resolution_m, grid_size)
    probs = np.zeros((grid_size, grid_size), dtype=np.float64)
    return ProbabilityGrid(probabilities=probs, metadata=metadata, crs=crs)


def cell_centroid_utm(grid: ProbabilityGrid, row: int, col: int) -> tuple[float, float]:
    half = (grid.rows * grid.metadata.resolution_m) / 2.0
    res = grid.metadata.resolution_m
    easting = grid.crs.origin_e - half + (col + 0.5) * res
    northing = grid.crs.origin_n + half - (row + 0.5) * res
    return easting, northing


def cell_centroid_latlon(grid: ProbabilityGrid, row: int, col: int) -> tuple[float, float]:
    e, n = cell_centroid_utm(grid, row, col)
    return grid.crs.to_wgs84(e, n)


def cells_in_polygon(grid: ProbabilityGrid, polygon_geojson: dict) -> list[tuple[int, int]]:
    poly = shape(polygon_geojson)
    cells: list[tuple[int, int]] = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            lat, lon = cell_centroid_latlon(grid, row, col)
            if poly.contains(Point(lon, lat)):
                cells.append((row, col))
    return cells


def grid_utm_bounds(grid: ProbabilityGrid) -> tuple[float, float, float, float]:
    """Return (min_e, min_n, max_e, max_n) for the probability grid in UTM."""
    half = (grid.rows * grid.metadata.resolution_m) / 2.0
    min_e = grid.crs.origin_e - half
    max_e = grid.crs.origin_e + half
    min_n = grid.crs.origin_n - half
    max_n = grid.crs.origin_n + half
    return min_e, min_n, max_e, max_n


def particle_cell_indices(
    grid: ProbabilityGrid,
    eastings: np.ndarray,
    northings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map particle UTM positions to clamped grid cell (row, col) indices."""
    half = (grid.rows * grid.metadata.resolution_m) / 2.0
    res = grid.metadata.resolution_m
    cols = (eastings - (grid.crs.origin_e - half)) / res
    rows = ((grid.crs.origin_n + half) - northings) / res
    r = np.clip(np.floor(rows).astype(np.int32), 0, grid.rows - 1)
    c = np.clip(np.floor(cols).astype(np.int32), 0, grid.cols - 1)
    return r, c


def extract_field_for_grid(
    display: ProbabilityGrid,
    source: ProbabilityGrid,
    field: np.ndarray,
) -> np.ndarray:
    """Sample a source-grid field at each display-grid cell centroid."""
    out = np.zeros((display.rows, display.cols), dtype=np.float64)
    max_r, max_c = field.shape[0] - 1, field.shape[1] - 1
    for row in range(display.rows):
        for col in range(display.cols):
            lat, lon = cell_centroid_latlon(display, row, col)
            e, n = source.crs.to_utm(lon, lat)
            half = (source.rows * source.metadata.resolution_m) / 2.0
            res = source.metadata.resolution_m
            col_f = (e - (source.crs.origin_e - half)) / res
            row_f = ((source.crs.origin_n + half) - n) / res
            sr_i = int(np.clip(np.floor(row_f), 0, max_r))
            sc_i = int(np.clip(np.floor(col_f), 0, max_c))
            out[row, col] = field[sr_i, sc_i]
    return out


def utm_particle_positions(grid: ProbabilityGrid, eastings: np.ndarray, northings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute UTM coords to grid-local indices for KDE."""
    half = (grid.rows * grid.metadata.resolution_m) / 2.0
    res = grid.metadata.resolution_m
    cols = (eastings - (grid.crs.origin_e - half)) / res
    rows = ((grid.crs.origin_n + half) - northings) / res
    return rows, cols
