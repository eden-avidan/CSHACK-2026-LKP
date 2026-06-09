import numpy as np
import pytest

from app.geospatial.grid import create_empty_grid, cells_in_polygon
from app.models.mission import LatLon
from app.services.negative_search import apply_negative_search
from app.services.particle_filter import (
    effective_sample_size,
    initialize_particles,
    rasterize_kde,
    systematic_resample,
)


LKP = LatLon(lat=37.7749, lon=-122.4194)


def test_initialize_weights_sum_to_one():
    grid = create_empty_grid(LKP, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 1000, 200.0)
    assert np.isclose(particles.weights.sum(), 1.0)


def test_kde_normalization():
    grid = create_empty_grid(LKP, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 500, 200.0)
    probs = rasterize_kde(
        particles, grid.rows, grid.cols, grid.metadata.resolution_m,
        grid.crs.origin_e, grid.crs.origin_n,
    )
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)


def test_negative_search_monotonicity():
    grid = create_empty_grid(LKP, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 500, 200.0)
    grid.probabilities = rasterize_kde(
        particles, grid.rows, grid.cols, grid.metadata.resolution_m,
        grid.crs.origin_e, grid.crs.origin_n,
    )
    before = grid.probabilities.copy()
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [-122.4194, 37.7749],
            [-122.4180, 37.7749],
            [-122.4180, 37.7760],
            [-122.4194, 37.7760],
            [-122.4194, 37.7749],
        ]],
    }
    apply_negative_search(grid, polygon, pod=0.85)
    assert np.isclose(grid.probabilities.sum(), 1.0, atol=1e-6)
    polygon_cells = cells_in_polygon(grid, polygon)
    assert len(polygon_cells) > 0
    decreased = 0
    for r, c in polygon_cells:
        if before[r, c] > 1e-6 and grid.probabilities[r, c] < before[r, c]:
            decreased += 1
    assert decreased > 0


def test_resample_invariant():
    grid = create_empty_grid(LKP, 50.0, 32)
    particles = initialize_particles(grid.crs.origin_e, grid.crs.origin_n, 100, 200.0)
    particles.weights = np.random.default_rng(0).random(100)
    particles.weights /= particles.weights.sum()
    resampled = systematic_resample(particles)
    assert resampled.count == particles.count
    assert np.isclose(resampled.weights.sum(), 1.0)


def test_effective_sample_size_uniform():
    w = np.full(100, 1 / 100)
    assert effective_sample_size(w) == pytest.approx(100.0, rel=0.01)
