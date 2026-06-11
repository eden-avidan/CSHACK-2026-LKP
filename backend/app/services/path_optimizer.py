from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from app.geospatial.grid import ProbabilityGrid, cell_centroid_latlon, cell_centroid_utm

# Hackathon baseline constraints. These keep route generation fast and produce
# a practical, readable route while still sampling several probability regions.
MAX_ROUTE_POINTS = 12
MAX_CANDIDATE_CELLS = 128
MAX_ROUTE_LENGTH_M = 6_000.0
COVERAGE_RADIUS_CELLS = 2
DISTANCE_PENALTY_M = 750.0
MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class DroneRoute:
    coordinates: list[list[float]]
    expected_coverage: float
    length_m: float
    cells: list[tuple[int, int]]


def _covered_cells(row: int, col: int, rows: int, cols: int) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    for r in range(max(0, row - COVERAGE_RADIUS_CELLS), min(rows, row + COVERAGE_RADIUS_CELLS + 1)):
        for c in range(max(0, col - COVERAGE_RADIUS_CELLS), min(cols, col + COVERAGE_RADIUS_CELLS + 1)):
            covered.add((r, c))
    return covered


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def optimize_drone_route(grid: ProbabilityGrid) -> DroneRoute:
    """Build a spatially coherent route that maximizes newly covered mass."""
    probabilities = grid.probabilities
    flat = probabilities.ravel()
    candidate_count = min(MAX_CANDIDATE_CELLS, flat.size)
    if candidate_count == 0:
        return DroneRoute([], 0.0, 0.0, [])

    indices = np.argpartition(flat, -candidate_count)[-candidate_count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    candidates = [
        divmod(int(index), grid.cols)
        for index in indices
        if float(flat[index]) > MIN_PROBABILITY
    ]

    start_utm = (grid.crs.origin_e, grid.crs.origin_n)
    start_lat = grid.metadata.origin.lat
    start_lon = grid.metadata.origin.lon
    coordinates: list[list[float]] = [[start_lon, start_lat]]
    selected: list[tuple[int, int]] = []
    covered: set[tuple[int, int]] = set()
    current_utm = start_utm
    length_m = 0.0

    while candidates and len(selected) < MAX_ROUTE_POINTS:
        best: tuple[float, float, tuple[int, int], tuple[float, float], set[tuple[int, int]]] | None = None
        for cell in candidates:
            row, col = cell
            cell_utm = cell_centroid_utm(grid, row, col)
            travel_m = _distance(current_utm, cell_utm)
            if length_m + travel_m > MAX_ROUTE_LENGTH_M:
                continue

            footprint = _covered_cells(row, col, grid.rows, grid.cols)
            newly_covered = footprint - covered
            added_mass = sum(float(probabilities[r, c]) for r, c in newly_covered)
            score = added_mass / (1.0 + travel_m / DISTANCE_PENALTY_M)
            choice = (score, added_mass, cell, cell_utm, footprint)
            if best is None or choice[:2] > best[:2]:
                best = choice

        if best is None or best[1] <= MIN_PROBABILITY:
            break

        _, _, cell, cell_utm, footprint = best
        row, col = cell
        length_m += _distance(current_utm, cell_utm)
        current_utm = cell_utm
        covered.update(footprint)
        selected.append(cell)
        candidates.remove(cell)
        lat, lon = cell_centroid_latlon(grid, row, col)
        coordinates.append([lon, lat])

    expected_coverage = sum(float(probabilities[r, c]) for r, c in covered)
    return DroneRoute(coordinates, expected_coverage, length_m, selected)
