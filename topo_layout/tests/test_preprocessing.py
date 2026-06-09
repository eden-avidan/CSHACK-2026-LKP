from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from topo_layout.preprocessing import (
    DemRaster,
    GeoReference,
    UnsupportedTopographicImageError,
    compute_slope,
    convert_image_to_dem,
    import_geotiff_dem,
    import_worldfile_dem,
    load_dem_tiff,
    save_dem_tiff,
    save_slope_tiff,
)


class PreprocessingTests(unittest.TestCase):
    def test_heightmap_conversion_scales_pixel_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "heightmap.png"
            data = np.array([[0, 255], [128, 64]], dtype=np.uint8)
            Image.fromarray(data, mode="L").save(image_path)

            dem = convert_image_to_dem(
                image_path=image_path,
                georef=GeoReference(
                    crs="EPSG:32636",
                    min_x=0,
                    min_y=0,
                    max_x=2,
                    max_y=2,
                ),
                elevation_min_m=100,
                elevation_max_m=300,
            )

            self.assertEqual(dem.shape, (2, 2))
            self.assertAlmostEqual(float(dem.elevation[0, 0]), 100.0)
            self.assertAlmostEqual(float(dem.elevation[0, 1]), 300.0)
            self.assertAlmostEqual(float(dem.elevation[1, 0]), 200.39215, places=4)

    def test_scanned_topo_maps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "scan.png"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(image_path)

            with self.assertRaises(UnsupportedTopographicImageError):
                convert_image_to_dem(
                    image_path=image_path,
                    georef=GeoReference(
                        crs="EPSG:32636",
                        min_x=0,
                        min_y=0,
                        max_x=2,
                        max_y=2,
                    ),
                    mode="scanned_topo_map",
                )

    def test_save_writes_raster_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "heightmap.png"
            output_path = Path(tmpdir) / "dem.tif"
            Image.fromarray(np.full((3, 4), 255, dtype=np.uint8), mode="L").save(image_path)

            dem = convert_image_to_dem(
                image_path=image_path,
                georef=GeoReference(
                    crs="EPSG:32636",
                    min_x=10,
                    min_y=20,
                    max_x=14,
                    max_y=23,
                ),
                elevation_min_m=0,
                elevation_max_m=1000,
            )
            raster_path, metadata_path = save_dem_tiff(dem, output_path)

            self.assertTrue(raster_path.exists())
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["crs"], "EPSG:32636")
            self.assertEqual(metadata["raster"]["width"], 4)
            self.assertEqual(metadata["raster"]["height"], 3)

    def test_load_dem_restores_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "heightmap.png"
            output_path = Path(tmpdir) / "dem.tif"
            Image.fromarray(np.array([[0, 255], [64, 128]], dtype=np.uint8), mode="L").save(image_path)

            original_dem = convert_image_to_dem(
                image_path=image_path,
                georef=GeoReference(
                    crs="EPSG:32636",
                    min_x=100,
                    min_y=200,
                    max_x=104,
                    max_y=204,
                ),
                elevation_min_m=50,
                elevation_max_m=250,
            )
            save_dem_tiff(original_dem, output_path)
            loaded_dem = load_dem_tiff(output_path)

            np.testing.assert_allclose(loaded_dem.elevation, original_dem.elevation)
            self.assertEqual(loaded_dem.georef.crs, "EPSG:32636")
            self.assertEqual(loaded_dem.shape, (2, 2))

    def test_compute_slope_for_uniform_east_west_rise(self) -> None:
        elevation = np.array(
            [
                [0.0, 10.0, 20.0],
                [0.0, 10.0, 20.0],
                [0.0, 10.0, 20.0],
            ],
            dtype=np.float32,
        )
        dem = load_dem_for_test(elevation=elevation, span_x=3.0, span_y=3.0)

        slope = compute_slope(dem)

        np.testing.assert_allclose(slope.slope_grade, np.ones((3, 3), dtype=np.float32) * 10.0)
        expected_degrees = np.ones((3, 3), dtype=np.float32) * np.degrees(np.arctan(10.0))
        np.testing.assert_allclose(slope.slope_degrees, expected_degrees, rtol=1e-5)

    def test_save_slope_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "slope.tif"
            dem = load_dem_for_test(
                elevation=np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32),
                span_x=2.0,
                span_y=2.0,
            )

            slope = compute_slope(dem)
            raster_path, metadata_path = save_slope_tiff(slope, output_path, layer="grade")

            self.assertTrue(raster_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["saved_layer"], "grade")
            self.assertEqual(metadata["layers"]["slope_degrees"]["description"], "Slope angle in degrees.")

    def test_import_worldfile_dem_mosaics_tiles_and_can_shift_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            left_tile = tmpdir_path / "tile_left.tif"
            right_tile = tmpdir_path / "tile_right.tif"
            Image.fromarray(np.array([[1, 2], [3, 4]], dtype=np.float32), mode="F").save(left_tile)
            Image.fromarray(np.array([[5, 6], [7, 8]], dtype=np.float32), mode="F").save(right_tile)
            (tmpdir_path / "tile_left.tfw").write_text(
                "10\n0\n0\n-10\n5\n15\n",
                encoding="utf-8",
            )
            (tmpdir_path / "tile_right.tfw").write_text(
                "10\n0\n0\n-10\n25\n15\n",
                encoding="utf-8",
            )

            dem = import_worldfile_dem(
                str(tmpdir_path / "tile_*.tif"),
                crs="LOCAL_TEST",
                shift_to_origin=True,
            )

            expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
            np.testing.assert_allclose(dem.elevation, expected)
            self.assertEqual(dem.georef.min_x, 0.0)
            self.assertEqual(dem.georef.min_y, 0.0)
            self.assertEqual(dem.georef.max_x, 40.0)
            self.assertEqual(dem.georef.max_y, 20.0)

    def test_import_geotiff_dem_reads_geotags_and_converts_geographic_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir) / "geo.tif"
            image = Image.fromarray(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32), mode="F")
            ifd = TiffImagePlugin.ImageFileDirectory_v2()
            ifd[33550] = (0.1, 0.1, 0.0)
            ifd[33922] = (0.0, 0.0, 0.0, 34.0, 33.0, 0.0)
            ifd[34735] = (
                1, 1, 0, 1,
                2048, 0, 1, 4326,
            )
            image.save(tif_path, tiffinfo=ifd)

            dem = import_geotiff_dem(tif_path, shift_to_origin=True)

            np.testing.assert_allclose(dem.elevation, np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
            self.assertEqual(dem.georef.crs, "LOCAL_METERS_FROM_EPSG4326")
            self.assertEqual(dem.georef.min_x, 0.0)
            self.assertEqual(dem.georef.min_y, 0.0)
            self.assertGreater(dem.georef.max_x, 0.0)
            self.assertGreater(dem.georef.max_y, 0.0)


def load_dem_for_test(
    *,
    elevation: np.ndarray,
    span_x: float,
    span_y: float,
) -> DemRaster:
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
