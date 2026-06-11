import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid
from app.models.mission import LatLon
from app.services.path_optimizer import (
    MAX_ROUTE_LENGTH_M,
    MAX_ROUTE_POINTS,
    _path_length,
    _two_opt,
    optimize_drone_route,
)


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True if open segment p1-p2 properly intersects p3-p4."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _has_crossing(points: list[tuple[float, float]]) -> bool:
    n = len(points)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j == n - 2:
                # endpoints of an open path can never "cross", skip adjacency
                pass
            if _segments_cross(points[i], points[i + 1], points[j], points[j + 1]):
                return True
    return False


HAIFA = LatLon(lat=32.7940, lon=34.9896)


def _grid(size: int = 32):
    return create_empty_grid(HAIFA, resolution_m=50.0, grid_size=size)


def test_route_covers_multiple_high_probability_regions():
    grid = _grid()
    grid.probabilities[8, 8] = 0.35
    grid.probabilities[8, 9] = 0.15
    grid.probabilities[23, 23] = 0.30
    grid.probabilities[23, 22] = 0.20

    result = optimize_drone_route(grid)

    assert len(result.cells) >= 2
    assert any(row < 16 and col < 16 for row, col in result.cells)
    assert any(row > 16 and col > 16 for row, col in result.cells)
    assert result.expected_coverage == pytest.approx(1.0)


def test_route_avoids_redundant_adjacent_points():
    grid = _grid()
    grid.probabilities[10:13, 10:13] = 1.0 / 9.0

    result = optimize_drone_route(grid)

    assert len(result.cells) == 1
    assert result.expected_coverage == pytest.approx(1.0)


def test_route_respects_point_and_length_limits():
    grid = _grid(size=128)
    grid.probabilities.fill(1.0 / grid.probabilities.size)

    result = optimize_drone_route(grid)

    assert len(result.cells) <= MAX_ROUTE_POINTS
    assert result.length_m <= MAX_ROUTE_LENGTH_M
    assert len(result.coordinates) == len(result.cells) + 1


def test_two_opt_uncrosses_a_crossed_path():
    # Square corners visited in an order that crosses (bowtie): the diagonal
    # ordering 0->2->1->3 has crossing edges; 2-opt should restore the perimeter.
    square = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    assert _has_crossing(square)

    order = _two_opt(square)
    optimized = [square[k] for k in order]

    assert order[0] == 0  # launch point stays pinned
    assert not _has_crossing(optimized)
    assert _path_length(optimized) < _path_length(square)


def test_two_opt_keeps_a_clean_path_unchanged():
    straight = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert _two_opt(straight) == [0, 1, 2, 3]


def test_route_has_no_crossing_edges():
    grid = _grid()
    grid.probabilities[5, 5] = 0.25
    grid.probabilities[5, 25] = 0.25
    grid.probabilities[25, 5] = 0.25
    grid.probabilities[25, 25] = 0.25
    grid.probabilities[15, 15] = 0.20

    result = optimize_drone_route(grid)

    pts = [(lon, lat) for lon, lat in result.coordinates]
    assert not _has_crossing(pts)
