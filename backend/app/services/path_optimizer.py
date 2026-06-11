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


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(_distance(points[k], points[k + 1]) for k in range(len(points) - 1))


def _two_opt(points: list[tuple[float, float]]) -> list[int]:
    """Shorten an open path with 2-opt edge swaps; index 0 stays pinned.

    Repeatedly reverses the path segment between two edges whenever doing so
    shortens the total length. In the Euclidean plane this is equivalent to
    uncrossing edges, so the loop runs until no two edges cross (no reversal
    yields a shorter path). Returns the optimized order as indices into
    ``points``; ``points[0]`` (the drone launch point) is never moved.
    """
    order = list(range(len(points)))
    if len(order) < 3:
        return order

    improved = True
    while improved:
        improved = False
        # i >= 1 keeps index 0 pinned as the route start.
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                a, b = order[i - 1], order[i]
                c = order[j]
                d = order[j + 1] if j + 1 < len(order) else None
                # Current edges (a,b)+(c,d) vs. proposed (a,c)+(b,d).
                before = _distance(points[a], points[b])
                after = _distance(points[a], points[c])
                if d is not None:
                    before += _distance(points[c], points[d])
                    after += _distance(points[b], points[d])
                if after + 1e-9 < before:
                    order[i : j + 1] = reversed(order[i : j + 1])
                    improved = True
    return order


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

    # 2-opt edge swaps: reorder the visited cells to shorten the total path,
    # looping until no segment reversal helps (i.e. no two edges cross). The
    # launch point and the visited cell set are preserved; only the visiting
    # order (and therefore length_m / coordinates) changes.
    if selected:
        points = [start_utm] + [cell_centroid_utm(grid, r, c) for r, c in selected]
        order = _two_opt(points)
        selected = [selected[k - 1] for k in order[1:]]
        length_m = _path_length([points[k] for k in order])
        coordinates = [[start_lon, start_lat]]
        for r, c in selected:
            lat, lon = cell_centroid_latlon(grid, r, c)
            coordinates.append([lon, lat])

    expected_coverage = sum(float(probabilities[r, c]) for r, c in covered)
    return DroneRoute(coordinates, expected_coverage, length_m, selected)
