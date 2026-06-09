from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from topo_layout.mobility import (
    TerrainInfluenceConfig,
    compute_heatmap,
    compute_heatmap_from_heightmap,
    row_col_to_coordinate,
    save_heatmap_tiff,
    tobler_hiking_speed_kmh,
)
from topo_layout.preprocessing import DemRaster, GeoReference
from topo_layout.terrain import TerrainRaster


class MobilityTests(unittest.TestCase):
    def test_tobler_prefers_gentle_downhill(self) -> None:
        downhill_speed = float(tobler_hiking_speed_kmh(-0.05))
        flat_speed = float(tobler_hiking_speed_kmh(0.0))
        uphill_speed = float(tobler_hiking_speed_kmh(0.2))

        self.assertAlmostEqual(downhill_speed, 6.0, places=6)
        self.assertGreater(downhill_speed, flat_speed)
        self.assertGreater(flat_speed, uphill_speed)

    def test_heatmap_on_flat_dem_has_symmetric_travel_times(self) -> None:
        dem = build_test_dem(np.zeros((3, 3), dtype=np.float32), span_x=300.0, span_y=300.0)
        start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=1, col=1)

        heatmap = compute_heatmap(
            dem,
            start_x=start_x,
            start_y=start_y,
            max_hours=0.03,
            probability_method="exponential",
        )

        self.assertAlmostEqual(float(np.sum(heatmap.probability)), 1.0, places=6)
        self.assertAlmostEqual(float(heatmap.travel_time_hours[1, 1]), 0.0, places=6)
        np.testing.assert_allclose(
            heatmap.travel_time_hours[0, 1],
            heatmap.travel_time_hours[1, 0],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            heatmap.travel_time_hours[0, 0],
            heatmap.travel_time_hours[0, 2],
            rtol=1e-6,
        )
        self.assertGreater(float(heatmap.probability[1, 1]), float(heatmap.probability[0, 1]))

    def test_steep_uphill_cells_become_less_reachable(self) -> None:
        dem = build_test_dem(
            np.array(
                [
                    [0.0, 150.0, 300.0],
                    [0.0, 150.0, 300.0],
                    [0.0, 150.0, 300.0],
                ],
                dtype=np.float32,
            ),
            span_x=300.0,
            span_y=300.0,
        )
        start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=1, col=0)

        heatmap = compute_heatmap(
            dem,
            start_x=start_x,
            start_y=start_y,
            max_hours=0.02,
            probability_method="linear",
        )

        self.assertTrue(np.isfinite(heatmap.travel_time_hours[1, 0]))
        self.assertTrue(np.isinf(heatmap.travel_time_hours[1, 2]))
        self.assertEqual(float(heatmap.probability[1, 2]), 0.0)

    def test_save_heatmap_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap.tif"
            dem = build_test_dem(np.zeros((2, 2), dtype=np.float32), span_x=200.0, span_y=200.0)
            start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=0, col=0)
            heatmap = compute_heatmap(
                dem,
                start_x=start_x,
                start_y=start_y,
                max_hours=0.05,
            )

            raster_path, metadata_path = save_heatmap_tiff(heatmap, output_path)

            self.assertTrue(raster_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["saved_layer"], "probability")
            self.assertEqual(metadata["time_horizon_hours"], 0.05)

    def test_save_color_heatmap_creates_rgb_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap.png"
            dem = build_test_dem(np.zeros((3, 3), dtype=np.float32), span_x=300.0, span_y=300.0)
            start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=1, col=1)
            heatmap = compute_heatmap(
                dem,
                start_x=start_x,
                start_y=start_y,
                max_hours=0.005,
            )

            raster_path, metadata_path = save_heatmap_tiff(
                heatmap,
                output_path,
                layer="probability_color",
            )

            self.assertTrue(raster_path.exists())
            saved = np.asarray(Image.open(raster_path))
            self.assertEqual(saved.shape, (3, 3, 3))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["saved_layer"], "probability_color")
            self.assertEqual(metadata["render"]["palette"], "cool-to-hot")
            self.assertTrue(np.array_equal(saved[0, 0], np.array([0, 0, 255], dtype=np.uint8)))

    def test_terrain_classes_reweight_equal_travel_time_cells(self) -> None:
        dem = build_test_dem(np.zeros((3, 3), dtype=np.float32), span_x=300.0, span_y=300.0)
        terrain = build_test_terrain(
            dem,
            valley_cells=[(1, 0)],
            ridge_cells=[(1, 2)],
        )
        start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=1, col=1)

        heatmap = compute_heatmap(
            dem,
            start_x=start_x,
            start_y=start_y,
            max_hours=0.03,
            probability_method="linear",
            terrain=terrain,
            terrain_influence=TerrainInfluenceConfig(
                steep_weight=1.0,
                cliff_like_weight=1.0,
                valley_weight=2.0,
                ridge_weight=0.5,
            ),
        )

        self.assertAlmostEqual(
            float(heatmap.travel_time_hours[1, 0]),
            float(heatmap.travel_time_hours[1, 2]),
            places=6,
        )
        self.assertGreater(float(heatmap.probability[1, 0]), float(heatmap.probability[1, 2]))

    def test_terrain_influence_can_be_disabled(self) -> None:
        dem = build_test_dem(np.zeros((3, 3), dtype=np.float32), span_x=300.0, span_y=300.0)
        terrain = build_test_terrain(
            dem,
            cliff_cells=[(1, 0)],
            valley_cells=[(1, 0)],
            ridge_cells=[(1, 2)],
        )
        start_x, start_y = row_col_to_coordinate(dem.georef, dem.shape, row=1, col=1)

        heatmap = compute_heatmap(
            dem,
            start_x=start_x,
            start_y=start_y,
            max_hours=0.03,
            probability_method="linear",
            terrain=terrain,
            use_terrain_influence=False,
        )

        self.assertAlmostEqual(
            float(heatmap.probability[1, 0]),
            float(heatmap.probability[1, 2]),
            places=6,
        )

    def test_compute_heatmap_from_heightmap_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "heightmap.png"
            Image.fromarray(
                np.array(
                    [
                        [0, 64, 128],
                        [64, 128, 192],
                        [128, 192, 255],
                    ],
                    dtype=np.uint8,
                ),
                mode="L",
            ).save(image_path)

            dem, heatmap = compute_heatmap_from_heightmap(
                image_path=image_path,
                georef=GeoReference(
                    crs="LOCAL_TEST",
                    min_x=0.0,
                    min_y=0.0,
                    max_x=300.0,
                    max_y=300.0,
                ),
                start_x=150.0,
                start_y=150.0,
                max_hours=0.05,
                elevation_min_m=0.0,
                elevation_max_m=300.0,
            )

            self.assertEqual(dem.shape, (3, 3))
            self.assertEqual(heatmap.shape, (3, 3))
            self.assertAlmostEqual(float(np.sum(heatmap.probability)), 1.0, places=6)


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


def build_test_terrain(
    dem: DemRaster,
    *,
    steep_cells: list[tuple[int, int]] | None = None,
    cliff_cells: list[tuple[int, int]] | None = None,
    valley_cells: list[tuple[int, int]] | None = None,
    ridge_cells: list[tuple[int, int]] | None = None,
) -> TerrainRaster:
    steep_cells = steep_cells or []
    cliff_cells = cliff_cells or []
    valley_cells = valley_cells or []
    ridge_cells = ridge_cells or []

    shape = dem.shape
    steep_mask = np.zeros(shape, dtype=np.uint8)
    cliff_mask = np.zeros(shape, dtype=np.uint8)
    valley_mask = np.zeros(shape, dtype=np.uint8)
    ridge_mask = np.zeros(shape, dtype=np.uint8)
    tpi = np.zeros(shape, dtype=np.float32)

    for row, col in steep_cells:
        steep_mask[row, col] = 1
    for row, col in cliff_cells:
        cliff_mask[row, col] = 1
    for row, col in valley_cells:
        valley_mask[row, col] = 1
        tpi[row, col] = -10.0
    for row, col in ridge_cells:
        ridge_mask[row, col] = 1
        tpi[row, col] = 10.0

    bitmask = np.zeros(shape, dtype=np.uint8)
    bitmask[steep_mask > 0] |= 1
    bitmask[cliff_mask > 0] |= 2
    bitmask[valley_mask > 0] |= 4
    bitmask[ridge_mask > 0] |= 8

    return TerrainRaster(
        bitmask=bitmask,
        steep_mask=steep_mask,
        cliff_like_mask=cliff_mask,
        valley_mask=valley_mask,
        ridge_mask=ridge_mask,
        topographic_position_index=tpi,
        georef=dem.georef,
        source_dem=dem.source_dem_path or dem.source_image,
        steep_threshold_deg=30.0,
        cliff_threshold_deg=45.0,
        neighborhood_size=3,
        ridge_threshold_m=5.0,
        valley_threshold_m=5.0,
    )


if __name__ == "__main__":
    unittest.main()
