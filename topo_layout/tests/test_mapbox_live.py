from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

import topo_layout.mapbox_live as mapbox_live
from topo_layout.mapbox_live import (
    MapboxHeatmapRequest,
    _TileImage,
    _build_dem_from_mapbox_terrain,
    _choose_terrain_zoom,
    _elapsed_hours_since,
    _lonlat_extent_to_local_meters,
    _lonlat_to_local_xy,
    _lonlat_to_tile_xy,
    _lonlat_to_tile_xy_arrays,
    _parse_iso_timestamp,
)


class MapboxLiveTests(unittest.TestCase):
    def test_choose_terrain_zoom_is_clamped(self) -> None:
        self.assertEqual(_choose_terrain_zoom(2.0), 8)
        self.assertEqual(_choose_terrain_zoom(12.3), 12)
        self.assertEqual(_choose_terrain_zoom(18.9), 15)

    def test_lonlat_to_local_xy_maps_bbox_center_to_local_center(self) -> None:
        west, south, east, north = 34.95, 32.75, 35.10, 32.90
        width_m, height_m = _lonlat_extent_to_local_meters(
            west=west,
            south=south,
            east=east,
            north=north,
        )
        x, y = _lonlat_to_local_xy(
            lon=(west + east) / 2.0,
            lat=(south + north) / 2.0,
            west=west,
            south=south,
            east=east,
            north=north,
        )
        self.assertAlmostEqual(x, width_m / 2.0, places=6)
        self.assertAlmostEqual(y, height_m / 2.0, places=6)

    def test_lonlat_to_tile_xy_stays_inside_world_range(self) -> None:
        x, y = _lonlat_to_tile_xy(35.0, 32.8, 10)
        self.assertGreaterEqual(x, 0.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLess(x, 2**10)
        self.assertLess(y, 2**10)

    def test_lonlat_to_tile_xy_arrays_match_scalar_helper(self) -> None:
        lon = np.array([[34.95, 35.0], [35.05, 35.10]], dtype=np.float64)
        lat = np.array([[32.75, 32.80], [32.85, 32.90]], dtype=np.float64)

        x_array, y_array = _lonlat_to_tile_xy_arrays(lon, lat, 10)

        for row in range(lon.shape[0]):
            for col in range(lon.shape[1]):
                x_scalar, y_scalar = _lonlat_to_tile_xy(float(lon[row, col]), float(lat[row, col]), 10)
                self.assertAlmostEqual(float(x_array[row, col]), x_scalar, places=10)
                self.assertAlmostEqual(float(y_array[row, col]), y_scalar, places=10)

    def test_build_dem_fetches_each_tile_once_for_many_cells(self) -> None:
        request = MapboxHeatmapRequest(
            west=34.95,
            south=32.75,
            east=35.10,
            north=32.90,
            last_known_lon=35.0,
            last_known_lat=32.8,
            max_hours=1.0,
            grid_width=64,
            grid_height=64,
        )

        original_fetch = mapbox_live._fetch_terrain_tile
        original_cache = mapbox_live._tile_cache
        fetch_calls: list[tuple[int, int, int]] = []

        def fake_fetch(zoom: int, x: int, y: int, access_token: str) -> _TileImage:
            fetch_calls.append((zoom, x, y))
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[:, :, 2] = 100
            return _TileImage(rgb=rgb, width=256, height=256)

        mapbox_live._fetch_terrain_tile = fake_fetch
        mapbox_live._tile_cache = mapbox_live._TileCache()
        try:
            dem = _build_dem_from_mapbox_terrain(request, terrain_zoom=10, access_token="pk.test")
        finally:
            mapbox_live._fetch_terrain_tile = original_fetch
            mapbox_live._tile_cache = original_cache

        self.assertEqual(dem.shape, (64, 64))
        self.assertGreater(len(fetch_calls), 0)
        self.assertEqual(len(fetch_calls), len(set(fetch_calls)))
        self.assertTrue(np.isfinite(dem.elevation).all())

    def test_parse_iso_timestamp_accepts_utc_suffix(self) -> None:
        parsed = _parse_iso_timestamp("2026-06-09T10:15:00Z")
        self.assertEqual(parsed, datetime(2026, 6, 9, 10, 15, tzinfo=timezone.utc))

    def test_elapsed_hours_since_uses_backend_clock(self) -> None:
        ninety_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=90)
        elapsed_hours = _elapsed_hours_since(ninety_minutes_ago.isoformat())
        self.assertGreater(elapsed_hours, 1.49)
        self.assertLess(elapsed_hours, 1.51)


if __name__ == "__main__":
    unittest.main()
