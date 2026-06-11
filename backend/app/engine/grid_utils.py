from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon
from app.models.mission import LatLon


def compute_mpp(grid: ProbabilityGrid, probabilities: np.ndarray) -> LatLon:
    """Most Probable Position: weighted centroid of top 5% probability cells."""
    margin = max(1, settings.kde_edge_fade_cells // 2)
    masked = probabilities.copy()
    if margin * 2 < masked.shape[0] and margin * 2 < masked.shape[1]:
        masked[:margin, :] = 0.0
        masked[-margin:, :] = 0.0
        masked[:, :margin] = 0.0
        masked[:, -margin:] = 0.0

    flat = masked.ravel()
    nonzero = flat[flat > 1e-12]
    if nonzero.size == 0:
        flat = probabilities.ravel()
        nonzero = flat[flat > 1e-12]
        if nonzero.size == 0:
            return LatLon(lat=grid.metadata.origin.lat, lon=grid.metadata.origin.lon)
        masked = probabilities

    threshold = float(np.percentile(nonzero, 95))
    rows, cols = np.where(masked >= threshold)
    if rows.size == 0:
        peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
        rows, cols = np.array([peak[0]]), np.array([peak[1]])

    weights = masked[rows, cols]
    w_sum = weights.sum()
    if w_sum <= 0:
        return LatLon(lat=grid.metadata.origin.lat, lon=grid.metadata.origin.lon)

    lat_acc = 0.0
    lon_acc = 0.0
    for row, col, w in zip(rows, cols, weights):
        lat, lon = cell_centroid_latlon(grid, int(row), int(col))
        lat_acc += lat * w
        lon_acc += lon * w

    return LatLon(lat=lat_acc / w_sum, lon=lon_acc / w_sum)


def downsample_grid_peaks(
    probabilities: np.ndarray,
    grid: ProbabilityGrid,
    limit: Optional[int] = None,
) -> list[list[float]]:
    """Top probability cells as [lat, lon, weight] for WS overlay."""
    cap = limit or settings.engine_tick_particle_limit
    flat = probabilities.ravel()
    if flat.size == 0:
        return []
    indices = np.argsort(flat)[::-1][:cap]
    rows, cols = probabilities.shape
    matrix: list[list[float]] = []
    for idx in indices:
        p = float(flat[idx])
        if p <= 1e-12:
            break
        row, col = divmod(int(idx), cols)
        lat, lon = cell_centroid_latlon(grid, row, col)
        matrix.append([lat, lon, p])
    return matrix
