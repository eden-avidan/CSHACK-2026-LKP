from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon, create_empty_grid
from app.models.mission import LatLon


@dataclass
class NodeFields:
    """Per-cell static layer inputs — populated once at mission create."""

    elevation: np.ndarray
    slope: np.ndarray
    is_land: np.ndarray
    is_road: np.ndarray
    road_proximity: np.ndarray
    road_tangent_e: np.ndarray
    road_tangent_n: np.ndarray
    wind_u: np.ndarray
    wind_v: np.ndarray
    reachability: np.ndarray
    latitude: np.ndarray   # cell-center latitude (degrees, WGS84)
    longitude: np.ndarray  # cell-center longitude (degrees, WGS84)
    altitude: np.ndarray   # cell-center altitude (meters above sea level);
                           # alias of `elevation`, populated from the same DEM

    @classmethod
    def zeros(cls, size: int) -> NodeFields:
        z = np.zeros((size, size), dtype=np.float64)
        land = np.ones((size, size), dtype=bool)
        reach = np.ones((size, size), dtype=np.float64)
        return cls(
            elevation=z.copy(),
            slope=z.copy(),
            is_land=land,
            is_road=z.copy(),
            road_proximity=z.copy(),
            road_tangent_e=z.copy(),
            road_tangent_n=z.copy(),
            wind_u=z.copy(),
            wind_v=z.copy(),
            reachability=reach,
            latitude=z.copy(),
            longitude=z.copy(),
            altitude=z.copy(),
        )


@dataclass
class GridMatrix:
    """
    Discrete A×A spatial matrix.

    Cell [lkp_row][lkp_col] is the Last Known Position (LKP) at t=0.
    Each cell covers resolution_m × resolution_m meters.
    Total simulated area: (A × Y) × (A × Y) meters.
    """

    size: int
    resolution_m: float
    lkp_row: int
    lkp_col: int
    probabilities: np.ndarray
    node_fields: NodeFields
    grid: ProbabilityGrid

    @classmethod
    def create(
        cls,
        lkp: LatLon,
        size: int,
        resolution_m: float,
        node_fields: NodeFields | None = None,
    ) -> GridMatrix:
        grid = create_empty_grid(lkp, resolution_m, size)
        lkp_row = size // 2
        lkp_col = size // 2
        fields = node_fields or NodeFields.zeros(size)
        _fill_cell_latlon(fields.latitude, fields.longitude, grid)
        matrix = cls(
            size=size,
            resolution_m=resolution_m,
            lkp_row=lkp_row,
            lkp_col=lkp_col,
            probabilities=np.zeros((size, size), dtype=np.float64),
            node_fields=fields,
            grid=grid,
        )
        matrix.initialize_t0()
        return matrix

    def initialize_t0(self) -> None:
        """t=0: all nodes 0.0 except LKP center = 1.0."""
        self.probabilities.fill(0.0)
        self.probabilities[self.lkp_row, self.lkp_col] = 1.0
        self.sync_to_grid()

    def sync_to_grid(self) -> None:
        self.grid.probabilities = self.probabilities.copy()

    def sync_from_grid(self) -> None:
        self.probabilities = self.grid.probabilities.copy()

    @property
    def total_area_m(self) -> float:
        span = self.size * self.resolution_m
        return span * span


def _fill_cell_latlon(
    latitude: np.ndarray, longitude: np.ndarray, grid: ProbabilityGrid
) -> None:
    """Write each cell's centroid (lat, lon) in WGS84 degrees into the arrays."""
    rows, cols = latitude.shape
    for row in range(rows):
        for col in range(cols):
            lat, lon = cell_centroid_latlon(grid, row, col)
            latitude[row, col] = lat
            longitude[row, col] = lon
