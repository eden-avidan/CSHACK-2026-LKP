import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid
from app.models.mission import LatLon
from app.services.path_optimizer import (
    MAX_ROUTE_LENGTH_M,
    MAX_ROUTE_POINTS,
    optimize_drone_route,
)


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
