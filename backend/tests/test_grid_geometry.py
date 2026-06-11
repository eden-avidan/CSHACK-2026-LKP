"""Tests for per-cell lat/lon/altitude on NodeFields and the WGS84 corners
shipped in GridMetadata.

These cover the surface area the frontend relies on:
  * NodeFields.latitude / .longitude match the backend's authoritative
    cell_centroid_latlon (bit-exact, since both call the same helper).
  * NodeFields.altitude mirrors NodeFields.elevation when terrain is present
    (and stays at zero when it isn't).
  * GridMetadata.corners stores the four actual UTM->WGS84 corners
    (NOT the axis-aligned bbox), and is large enough vs. `bounds` to
    matter for frontend cell math.
  * Bilinear interpolation across the four corners reproduces
    cell_centroid_latlon to sub-meter accuracy across the entire grid
    (this is the contract the frontend's cellLatLon helper depends on).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.engine.grid_matrix import GridMatrix, NodeFields
from app.geospatial.grid import build_grid_metadata, cell_centroid_latlon
from app.models.mission import LatLon
from app.services.env_ingestion import TerrainContext


def _import_build_node_fields():
    """Lazy import: app.engine.node_builder pulls in the legacy app.layers
    chain which has a pre-existing Python 3.11+ dataclass issue unrelated
    to these tests. We surface a clear skip when that's the failure."""
    try:
        from app.engine.node_builder import build_node_fields
    except Exception as exc:  # noqa: BLE001 - intentionally broad
        pytest.skip(f"build_node_fields unavailable (legacy import error): {exc}")
    return build_node_fields

# A realistic SAR-area LKP near the eastern edge of UTM zone 36N — this is
# the *worst* case for the bbox-vs-corners discrepancy and a good stress test.
JERUSALEM = LatLon(lat=31.7170, lon=35.9993)
HAIFA = LatLon(lat=32.7940, lon=34.9896)


# --- NodeFields shape / defaults --------------------------------------------


def test_node_fields_zeros_includes_new_fields():
    fields = NodeFields.zeros(8)
    assert fields.latitude.shape == (8, 8)
    assert fields.longitude.shape == (8, 8)
    assert fields.altitude.shape == (8, 8)
    assert np.all(fields.latitude == 0.0)
    assert np.all(fields.longitude == 0.0)
    assert np.all(fields.altitude == 0.0)


# --- latitude / longitude population ----------------------------------------


def test_grid_matrix_create_populates_latlon_for_every_cell():
    size = 16
    matrix = GridMatrix.create(JERUSALEM, size=size, resolution_m=50.0)

    for row in range(size):
        for col in range(size):
            expected_lat, expected_lon = cell_centroid_latlon(matrix.grid, row, col)
            assert matrix.node_fields.latitude[row, col] == pytest.approx(expected_lat)
            assert matrix.node_fields.longitude[row, col] == pytest.approx(expected_lon)


def test_lkp_cell_latlon_is_close_to_lkp():
    size = 32
    matrix = GridMatrix.create(JERUSALEM, size=size, resolution_m=50.0)
    lat = matrix.node_fields.latitude[matrix.lkp_row, matrix.lkp_col]
    lon = matrix.node_fields.longitude[matrix.lkp_row, matrix.lkp_col]
    # LKP cell is one of the four cells around the LKP point; centroid is
    # within half a cell (~35 m diagonal) of the click in any direction.
    mlat = 111_132.0
    mlon = 111_320.0 * math.cos(math.radians(JERUSALEM.lat))
    err = math.hypot((lat - JERUSALEM.lat) * mlat, (lon - JERUSALEM.lon) * mlon)
    assert err < 50.0, f"LKP cell centroid is {err:.1f} m from the LKP click"


def test_longitude_varies_across_cols_constant_across_rows():
    size = 8
    matrix = GridMatrix.create(JERUSALEM, size=size, resolution_m=100.0)
    lons = matrix.node_fields.longitude

    for row in range(size):
        assert lons[row, 1] > lons[row, 0], "lon should grow west -> east"
        assert lons[row, -1] > lons[row, 0]

    for col in range(size):
        col_lons = lons[:, col]
        spread = col_lons.max() - col_lons.min()
        assert spread < 1e-3, f"col {col} lon spread {spread} (UTM convergence)"


def test_latitude_varies_across_rows_decreasing_southward():
    size = 8
    matrix = GridMatrix.create(JERUSALEM, size=size, resolution_m=100.0)
    lats = matrix.node_fields.latitude

    for col in range(size):
        assert lats[0, col] > lats[1, col], "row 0 should be the north edge"
        assert lats[0, col] > lats[-1, col]


# --- altitude / elevation mirror --------------------------------------------


def _fake_terrain(size: int, elevation: np.ndarray) -> TerrainContext:
    z = np.zeros((size, size), dtype=np.float64)
    return TerrainContext(
        elevation=elevation,
        slope=z.copy(),
        aspect_n=z.copy(),
        aspect_e=z.copy(),
        road_proximity=z.copy(),
        is_land=np.ones((size, size), dtype=bool),
        road_tangent_e=z.copy(),
        road_tangent_n=z.copy(),
    )


def test_altitude_mirrors_elevation_when_terrain_present():
    build_node_fields = _import_build_node_fields()
    size = 8
    elevation = np.linspace(100.0, 900.0, size * size).reshape(size, size)
    terrain = _fake_terrain(size, elevation)
    fields = build_node_fields(terrain, size=size, weather_enabled=False)

    np.testing.assert_array_equal(fields.elevation, elevation)
    np.testing.assert_array_equal(fields.altitude, fields.elevation)


def test_altitude_is_a_copy_not_a_view():
    build_node_fields = _import_build_node_fields()
    size = 4
    elevation = np.full((size, size), 250.0)
    terrain = _fake_terrain(size, elevation)
    fields = build_node_fields(terrain, size=size, weather_enabled=False)

    fields.elevation[0, 0] = 9999.0
    assert fields.altitude[0, 0] == 250.0, "altitude must not be aliased to elevation"


def test_altitude_defaults_to_zero_with_no_terrain():
    build_node_fields = _import_build_node_fields()
    fields = build_node_fields(terrain=None, size=8, weather_enabled=False)
    assert np.all(fields.altitude == 0.0)
    assert np.all(fields.elevation == 0.0)


# --- GridMetadata.corners ---------------------------------------------------


def test_grid_metadata_includes_four_corners():
    metadata, _ = build_grid_metadata(HAIFA, resolution_m=50.0, grid_size=32)
    c = metadata.corners

    assert c.nw.lat > c.sw.lat, "NW above SW"
    assert c.ne.lat > c.se.lat, "NE above SE"
    assert c.ne.lon > c.nw.lon, "NE east of NW"
    assert c.se.lon > c.sw.lon, "SE east of SW"


def test_bounds_is_outer_bbox_of_corners():
    metadata, _ = build_grid_metadata(JERUSALEM, resolution_m=50.0, grid_size=64)
    c, b = metadata.corners, metadata.bounds

    assert b.north == max(c.nw.lat, c.ne.lat)
    assert b.south == min(c.sw.lat, c.se.lat)
    assert b.east == max(c.ne.lon, c.se.lon)
    assert b.west == min(c.nw.lon, c.sw.lon)


def test_corners_differ_from_bbox_near_utm_zone_edge():
    """If corners were just the bbox, the frontend wouldn't gain anything
    from receiving them. At Jerusalem (near UTM zone 36N edge), the actual
    corners must differ from the bbox by an amount that matters in meters."""
    metadata, _ = build_grid_metadata(JERUSALEM, resolution_m=50.0, grid_size=128)
    c, b = metadata.corners, metadata.bounds

    diffs_deg = [
        abs(c.nw.lat - b.north), abs(c.ne.lat - b.north),
        abs(c.sw.lat - b.south), abs(c.se.lat - b.south),
        abs(c.nw.lon - b.west),  abs(c.sw.lon - b.west),
        abs(c.ne.lon - b.east),  abs(c.se.lon - b.east),
    ]
    max_drift_deg = max(diffs_deg)
    max_drift_m = max_drift_deg * 111_000
    assert max_drift_m > 5.0, (
        f"corners barely differ from bbox ({max_drift_m:.2f} m); "
        f"frontend would gain nothing from receiving them"
    )


# --- The big contract: bilinear corners reproduces the backend truth --------


def _bilinear(u: float, v: float, c) -> tuple[float, float]:
    """Same formula the frontend's cellLatLon uses on the WGS84 corners."""
    top_lat = c.nw.lat * (1 - u) + c.ne.lat * u
    top_lon = c.nw.lon * (1 - u) + c.ne.lon * u
    bot_lat = c.sw.lat * (1 - u) + c.se.lat * u
    bot_lon = c.sw.lon * (1 - u) + c.se.lon * u
    return top_lat * (1 - v) + bot_lat * v, top_lon * (1 - v) + bot_lon * v


def _meters_between(lat1, lon1, lat2, lon2):
    mlat = 111_132.0
    mlon = 111_320.0 * math.cos(math.radians(lat1))
    return math.hypot((lat2 - lat1) * mlat, (lon2 - lon1) * mlon)


@pytest.mark.parametrize("lkp", [HAIFA, JERUSALEM])
def test_frontend_bilinear_matches_backend_within_one_meter(lkp):
    size = 128
    res = 50.0
    matrix = GridMatrix.create(lkp, size=size, resolution_m=res)
    corners = matrix.grid.metadata.corners

    worst = 0.0
    worst_cell = (-1, -1)
    for row in range(size):
        for col in range(size):
            truth_lat, truth_lon = cell_centroid_latlon(matrix.grid, row, col)
            u = (col + 0.5) / size
            v = (row + 0.5) / size
            approx_lat, approx_lon = _bilinear(u, v, corners)
            err = _meters_between(truth_lat, truth_lon, approx_lat, approx_lon)
            if err > worst:
                worst = err
                worst_cell = (row, col)
    assert worst < 1.0, (
        f"bilinear-in-corners drift {worst:.3f} m at cell {worst_cell} for LKP {lkp} "
        f"-- frontend cellLatLon contract broken"
    )


def test_bbox_only_interpolation_is_demonstrably_worse():
    """Sanity: confirms the bbox-based interpolation we replaced was bad.
    If this ever starts passing with a tiny error, UTM behavior changed
    and the corners field may no longer be necessary."""
    size = 128
    res = 50.0
    matrix = GridMatrix.create(JERUSALEM, size=size, resolution_m=res)
    b = matrix.grid.metadata.bounds

    worst = 0.0
    for row in range(size):
        for col in range(size):
            truth_lat, truth_lon = cell_centroid_latlon(matrix.grid, row, col)
            fy = (row + 0.5) / size
            fx = (col + 0.5) / size
            bbox_lat = b.north - fy * (b.north - b.south)
            bbox_lon = b.west + fx * (b.east - b.west)
            err = _meters_between(truth_lat, truth_lon, bbox_lat, bbox_lon)
            worst = max(worst, err)
    assert worst > 10.0, (
        f"bbox-only interp is now only {worst:.2f} m off, which contradicts "
        f"the premise for the corners field"
    )
