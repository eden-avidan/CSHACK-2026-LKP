from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, shape

from app.geospatial.crs import CRSHelper
from app.models.heatmap import GridBounds, GridCorners, GridMetadata
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

    sw_lat, sw_lon = crs.offset_to_wgs84(-half, -half)
    se_lat, se_lon = crs.offset_to_wgs84( half, -half)
    ne_lat, ne_lon = crs.offset_to_wgs84( half,  half)
    nw_lat, nw_lon = crs.offset_to_wgs84(-half,  half)
    lats = [sw_lat, se_lat, ne_lat, nw_lat]
    lons = [sw_lon, se_lon, ne_lon, nw_lon]

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
        corners=GridCorners(
            nw=LatLon(lat=nw_lat, lon=nw_lon),
            ne=LatLon(lat=ne_lat, lon=ne_lon),
            se=LatLon(lat=se_lat, lon=se_lon),
            sw=LatLon(lat=sw_lat, lon=sw_lon),
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
    """Sample a source-grid field at each display-grid cell centroid (bilinear)."""
    if (
        display.rows == source.rows
        and display.cols == source.cols
        and display.metadata.origin.lat == source.metadata.origin.lat
        and display.metadata.origin.lon == source.metadata.origin.lon
    ):
        return field.astype(np.float64, copy=False)

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
            r0 = int(np.floor(row_f))
            c0 = int(np.floor(col_f))
            if r0 < 0 or c0 < 0 or r0 > max_r or c0 > max_c:
                continue
            r1 = min(r0 + 1, max_r)
            c1 = min(c0 + 1, max_c)
            dr = row_f - r0
            dc = col_f - c0
            v00 = field[r0, c0]
            v01 = field[r0, c1]
            v10 = field[r1, c0]
            v11 = field[r1, c1]
            out[row, col] = (
                v00 * (1.0 - dr) * (1.0 - dc)
                + v01 * (1.0 - dr) * dc
                + v10 * dr * (1.0 - dc)
                + v11 * dr * dc
            )
    return out


def utm_particle_positions(grid: ProbabilityGrid, eastings: np.ndarray, northings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute UTM coords to grid-local indices for KDE."""
    half = (grid.rows * grid.metadata.resolution_m) / 2.0
    res = grid.metadata.resolution_m
    cols = (eastings - (grid.crs.origin_e - half)) / res
    rows = ((grid.crs.origin_n + half) - northings) / res
    return rows, cols
