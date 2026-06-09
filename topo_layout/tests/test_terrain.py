from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from topo_layout.preprocessing import DemRaster, GeoReference
from topo_layout.terrain import (
    TERRAIN_BIT_CLIFF_LIKE,
    TERRAIN_BIT_RIDGE,
    TERRAIN_BIT_STEEP,
    TERRAIN_BIT_VALLEY,
    classify_terrain,
    save_terrain_tiff,
)


class TerrainTests(unittest.TestCase):
    def test_peak_is_classified_as_ridge(self) -> None:
        dem = build_test_dem(
            np.array(
                [
                    [100.0, 100.0, 100.0],
                    [100.0, 120.0, 100.0],
                    [100.0, 100.0, 100.0],
                ],
                dtype=np.float32,
            ),
            span_x=300.0,
            span_y=300.0,
        )

        terrain = classify_terrain(dem, ridge_threshold_m=10.0, valley_threshold_m=10.0)

        self.assertEqual(int(terrain.ridge_mask[1, 1]), 1)
        self.assertEqual(int(terrain.valley_mask[1, 1]), 0)
        self.assertTrue(int(terrain.bitmask[1, 1]) & TERRAIN_BIT_RIDGE)

    def test_pit_is_classified_as_valley(self) -> None:
        dem = build_test_dem(
            np.array(
                [
                    [100.0, 100.0, 100.0],
                    [100.0, 80.0, 100.0],
                    [100.0, 100.0, 100.0],
                ],
                dtype=np.float32,
            ),
            span_x=300.0,
            span_y=300.0,
        )

        terrain = classify_terrain(dem, ridge_threshold_m=10.0, valley_threshold_m=10.0)

        self.assertEqual(int(terrain.valley_mask[1, 1]), 1)
        self.assertEqual(int(terrain.ridge_mask[1, 1]), 0)
        self.assertTrue(int(terrain.bitmask[1, 1]) & TERRAIN_BIT_VALLEY)

    def test_steep_ramp_sets_steep_and_cliff_bits(self) -> None:
        dem = build_test_dem(
            np.array(
                [
                    [0.0, 100.0, 200.0],
                    [0.0, 100.0, 200.0],
                    [0.0, 100.0, 200.0],
                ],
                dtype=np.float32,
            ),
            span_x=300.0,
            span_y=300.0,
        )

        terrain = classify_terrain(
            dem,
            steep_threshold_deg=20.0,
            cliff_threshold_deg=40.0,
            ridge_threshold_m=500.0,
            valley_threshold_m=500.0,
        )

        self.assertTrue(np.all(terrain.steep_mask == 1))
        self.assertTrue(np.all(terrain.cliff_like_mask == 1))
        self.assertTrue(np.all((terrain.bitmask & TERRAIN_BIT_STEEP) > 0))
        self.assertTrue(np.all((terrain.bitmask & TERRAIN_BIT_CLIFF_LIKE) > 0))

    def test_save_terrain_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "terrain.tif"
            dem = build_test_dem(np.zeros((3, 3), dtype=np.float32), span_x=300.0, span_y=300.0)
            terrain = classify_terrain(dem)

            raster_path, metadata_path = save_terrain_tiff(terrain, output_path, layer="bitmask")

            self.assertTrue(raster_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["saved_layer"], "bitmask")
            self.assertEqual(
                metadata["layers"]["bitmask"]["bits"][str(TERRAIN_BIT_RIDGE)],
                "ridge",
            )


def build_test_dem(elevation: np.ndarray, *, span_x: float, span_y: float) -> DemRaster:
    return DemRaster(
        elevation=elevation,
        georef=GeoReference(
            crs="EPSG:32636",
            min_x=0.0,
            min_y=0.0,
            max_x=span_x,
            max_y=span_y,
        ),
        elevation_min_m=float(np.min(elevation)),
        elevation_max_m=float(np.max(elevation)),
        source_image="test-heightmap.png",
        source_dem_path="test-dem.tif",
    )


if __name__ == "__main__":
    unittest.main()
