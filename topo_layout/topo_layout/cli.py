from __future__ import annotations

import argparse
from pathlib import Path

from .mobility import (
    TerrainInfluenceConfig,
    compute_heatmap,
    compute_heatmap_from_heightmap,
    save_heatmap_tiff,
)
from .preprocessing import (
    GeoReference,
    compute_slope,
    convert_image_to_dem,
    import_geotiff_dem,
    import_worldfile_dem,
    load_dem_tiff,
    save_dem_tiff,
    save_slope_tiff,
)
from .terrain import classify_terrain, save_terrain_tiff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert supported terrain imagery into DEM raster artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert-image",
        help="Convert a heightmap-style image into a TIFF DEM plus JSON metadata.",
    )
    convert_parser.add_argument("--input", required=True, help="Path to the input PNG/JPG")
    convert_parser.add_argument("--output", required=True, help="Path to the output TIFF")
    convert_parser.add_argument("--crs", required=True, help="Coordinate reference system, e.g. EPSG:32636")
    convert_parser.add_argument("--min-x", required=True, type=float)
    convert_parser.add_argument("--min-y", required=True, type=float)
    convert_parser.add_argument("--max-x", required=True, type=float)
    convert_parser.add_argument("--max-y", required=True, type=float)
    convert_parser.add_argument("--elevation-min", required=True, type=float)
    convert_parser.add_argument("--elevation-max", required=True, type=float)
    convert_parser.add_argument(
        "--mode",
        default="heightmap",
        choices=("heightmap", "scanned_topo_map"),
        help="Use 'heightmap' when brightness encodes elevation directly.",
    )

    import_parser = subparsers.add_parser(
        "import-worldfile-dem",
        help="Import one or more TIFF tiles with adjacent TFW files into a DEM artifact.",
    )
    import_parser.add_argument("--input-glob", required=True, help="Glob for TIFF tiles, e.g. .\\tiles\\*.tif")
    import_parser.add_argument("--output", required=True, help="Path to the output DEM TIFF")
    import_parser.add_argument("--crs", required=True, help="Coordinate reference system label to store")
    import_parser.add_argument(
        "--shift-to-origin",
        action="store_true",
        help="Shift the DEM bounds so the lower-left corner becomes (0, 0).",
    )

    geotiff_parser = subparsers.add_parser(
        "import-geotiff-dem",
        help="Import a GeoTIFF DEM using embedded georeferencing tags.",
    )
    geotiff_parser.add_argument("--input", required=True, help="Path to the input GeoTIFF DEM")
    geotiff_parser.add_argument("--output", required=True, help="Path to the output DEM TIFF")
    geotiff_parser.add_argument(
        "--shift-to-origin",
        action="store_true",
        help="Shift the imported DEM bounds so the lower-left corner becomes (0, 0).",
    )
    geotiff_parser.add_argument(
        "--keep-geographic-coordinates",
        action="store_true",
        help="Keep geographic coordinates instead of converting EPSG:4326 bounds into local meters.",
    )

    slope_parser = subparsers.add_parser(
        "compute-slope",
        help="Compute a slope raster from a previously generated DEM TIFF and metadata.",
    )
    slope_parser.add_argument("--input-dem", required=True, help="Path to the DEM TIFF")
    slope_parser.add_argument(
        "--input-metadata",
        help="Optional path to the DEM metadata JSON. Defaults to <input-dem>.json",
    )
    slope_parser.add_argument("--output", required=True, help="Path to the output slope TIFF")
    slope_parser.add_argument(
        "--layer",
        default="degrees",
        choices=("degrees", "grade"),
        help="Which slope layer to save to the TIFF.",
    )

    heatmap_parser = subparsers.add_parser(
        "generate-heatmap",
        help="Generate a Tobler-based travel-time heatmap from a DEM and last known point.",
    )
    heatmap_parser.add_argument("--input-dem", required=True, help="Path to the DEM TIFF")
    heatmap_parser.add_argument(
        "--input-metadata",
        help="Optional path to the DEM metadata JSON. Defaults to <input-dem>.json",
    )
    heatmap_parser.add_argument("--start-x", required=True, type=float, help="Last known point X coordinate")
    heatmap_parser.add_argument("--start-y", required=True, type=float, help="Last known point Y coordinate")
    heatmap_parser.add_argument(
        "--max-hours",
        required=True,
        type=float,
        help="Time horizon in hours since the person was last seen.",
    )
    heatmap_parser.add_argument("--output", required=True, help="Path to the output heatmap TIFF")
    heatmap_parser.add_argument(
        "--layer",
        default="probability",
        choices=("probability", "travel_time_hours", "probability_color", "travel_time_color"),
        help="Which heatmap layer to save. Use a .png output path for the color layers.",
    )
    heatmap_parser.add_argument(
        "--probability-method",
        default="linear",
        choices=("linear", "exponential"),
        help="How to convert travel time into normalized heatmap weights.",
    )
    heatmap_parser.add_argument(
        "--disable-terrain-influence",
        action="store_true",
        help="Turn off terrain-class weighting and use only travel-time probability.",
    )
    heatmap_parser.add_argument("--steep-weight", default=0.7, type=float)
    heatmap_parser.add_argument("--cliff-like-weight", default=0.2, type=float)
    heatmap_parser.add_argument("--valley-weight", default=1.15, type=float)
    heatmap_parser.add_argument("--ridge-weight", default=0.9, type=float)
    heatmap_parser.add_argument("--terrain-steep-threshold-deg", default=30.0, type=float)
    heatmap_parser.add_argument("--terrain-cliff-threshold-deg", default=45.0, type=float)
    heatmap_parser.add_argument("--terrain-neighborhood-size", default=3, type=int)
    heatmap_parser.add_argument("--terrain-ridge-threshold-m", default=5.0, type=float)
    heatmap_parser.add_argument("--terrain-valley-threshold-m", default=5.0, type=float)

    heightmap_heatmap_parser = subparsers.add_parser(
        "generate-heatmap-from-image",
        help="Generate a heatmap directly from a heightmap-style PNG/JPG input.",
    )
    heightmap_heatmap_parser.add_argument("--input", required=True, help="Path to the input heightmap PNG/JPG")
    heightmap_heatmap_parser.add_argument("--crs", required=True, help="Coordinate reference system label")
    heightmap_heatmap_parser.add_argument("--min-x", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--min-y", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--max-x", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--max-y", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--elevation-min", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--elevation-max", required=True, type=float)
    heightmap_heatmap_parser.add_argument("--start-x", required=True, type=float, help="Last known point X coordinate")
    heightmap_heatmap_parser.add_argument("--start-y", required=True, type=float, help="Last known point Y coordinate")
    heightmap_heatmap_parser.add_argument(
        "--max-hours",
        required=True,
        type=float,
        help="Time horizon in hours since the person was last seen.",
    )
    heightmap_heatmap_parser.add_argument("--output", required=True, help="Path to the output heatmap file")
    heightmap_heatmap_parser.add_argument(
        "--layer",
        default="probability",
        choices=("probability", "travel_time_hours", "probability_color", "travel_time_color"),
        help="Which heatmap layer to save. Use a .png output path for the color layers.",
    )
    heightmap_heatmap_parser.add_argument(
        "--probability-method",
        default="linear",
        choices=("linear", "exponential"),
        help="How to convert travel time into normalized heatmap weights.",
    )
    heightmap_heatmap_parser.add_argument(
        "--disable-terrain-influence",
        action="store_true",
        help="Turn off terrain-class weighting and use only travel-time probability.",
    )
    heightmap_heatmap_parser.add_argument("--steep-weight", default=0.7, type=float)
    heightmap_heatmap_parser.add_argument("--cliff-like-weight", default=0.2, type=float)
    heightmap_heatmap_parser.add_argument("--valley-weight", default=1.15, type=float)
    heightmap_heatmap_parser.add_argument("--ridge-weight", default=0.9, type=float)
    heightmap_heatmap_parser.add_argument("--terrain-steep-threshold-deg", default=30.0, type=float)
    heightmap_heatmap_parser.add_argument("--terrain-cliff-threshold-deg", default=45.0, type=float)
    heightmap_heatmap_parser.add_argument("--terrain-neighborhood-size", default=3, type=int)
    heightmap_heatmap_parser.add_argument("--terrain-ridge-threshold-m", default=5.0, type=float)
    heightmap_heatmap_parser.add_argument("--terrain-valley-threshold-m", default=5.0, type=float)
    heightmap_heatmap_parser.add_argument(
        "--output-dem",
        help="Optional path to save the intermediate DEM artifact created from the heightmap.",
    )

    terrain_parser = subparsers.add_parser(
        "classify-terrain",
        help="Derive terrain classes like steep, cliff-like, valley, and ridge from a DEM.",
    )
    terrain_parser.add_argument("--input-dem", required=True, help="Path to the DEM TIFF")
    terrain_parser.add_argument(
        "--input-metadata",
        help="Optional path to the DEM metadata JSON. Defaults to <input-dem>.json",
    )
    terrain_parser.add_argument("--output", required=True, help="Path to the output terrain TIFF")
    terrain_parser.add_argument(
        "--layer",
        default="bitmask",
        choices=("bitmask", "steep", "cliff_like", "valley", "ridge", "tpi"),
        help="Which terrain layer to save to the TIFF.",
    )
    terrain_parser.add_argument("--steep-threshold-deg", default=30.0, type=float)
    terrain_parser.add_argument("--cliff-threshold-deg", default=45.0, type=float)
    terrain_parser.add_argument("--neighborhood-size", default=3, type=int)
    terrain_parser.add_argument("--ridge-threshold-m", default=5.0, type=float)
    terrain_parser.add_argument("--valley-threshold-m", default=5.0, type=float)

    serve_parser = subparsers.add_parser(
        "serve-mapbox-app",
        help="Run the live Mapbox-backed heatmap web app.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert-image":
        georef = GeoReference(
            crs=args.crs,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y,
        )
        dem = convert_image_to_dem(
            image_path=Path(args.input),
            georef=georef,
            mode=args.mode,
            elevation_min_m=args.elevation_min,
            elevation_max_m=args.elevation_max,
        )
        raster_path, metadata_path = save_dem_tiff(dem, Path(args.output))
        print(f"Saved raster: {raster_path}")
        print(f"Saved metadata: {metadata_path}")
        return 0

    if args.command == "import-worldfile-dem":
        dem = import_worldfile_dem(
            input_glob=args.input_glob,
            crs=args.crs,
            shift_to_origin=args.shift_to_origin,
        )
        raster_path, metadata_path = save_dem_tiff(dem, Path(args.output))
        print(f"Saved raster: {raster_path}")
        print(f"Saved metadata: {metadata_path}")
        return 0

    if args.command == "import-geotiff-dem":
        dem = import_geotiff_dem(
            input_path=Path(args.input),
            shift_to_origin=args.shift_to_origin,
            convert_geographic_to_local_meters=not args.keep_geographic_coordinates,
        )
        raster_path, metadata_path = save_dem_tiff(dem, Path(args.output))
        print(f"Saved raster: {raster_path}")
        print(f"Saved metadata: {metadata_path}")
        return 0

    if args.command == "compute-slope":
        dem = load_dem_tiff(
            dem_path=Path(args.input_dem),
            metadata_path=Path(args.input_metadata) if args.input_metadata else None,
        )
        slope = compute_slope(dem)
        raster_path, metadata_path = save_slope_tiff(
            slope,
            Path(args.output),
            layer=args.layer,
        )
        print(f"Saved slope raster: {raster_path}")
        print(f"Saved slope metadata: {metadata_path}")
        return 0

    if args.command == "generate-heatmap":
        dem = load_dem_tiff(
            dem_path=Path(args.input_dem),
            metadata_path=Path(args.input_metadata) if args.input_metadata else None,
        )
        terrain_influence = TerrainInfluenceConfig(
            steep_weight=args.steep_weight,
            cliff_like_weight=args.cliff_like_weight,
            valley_weight=args.valley_weight,
            ridge_weight=args.ridge_weight,
        )
        heatmap = compute_heatmap(
            dem,
            start_x=args.start_x,
            start_y=args.start_y,
            max_hours=args.max_hours,
            probability_method=args.probability_method,
            use_terrain_influence=not args.disable_terrain_influence,
            terrain_influence=terrain_influence,
            terrain_steep_threshold_deg=args.terrain_steep_threshold_deg,
            terrain_cliff_threshold_deg=args.terrain_cliff_threshold_deg,
            terrain_neighborhood_size=args.terrain_neighborhood_size,
            terrain_ridge_threshold_m=args.terrain_ridge_threshold_m,
            terrain_valley_threshold_m=args.terrain_valley_threshold_m,
        )
        raster_path, metadata_path = save_heatmap_tiff(
            heatmap,
            Path(args.output),
            layer=args.layer,
        )
        print(f"Saved heatmap raster: {raster_path}")
        print(f"Saved heatmap metadata: {metadata_path}")
        return 0

    if args.command == "generate-heatmap-from-image":
        georef = GeoReference(
            crs=args.crs,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y,
        )
        terrain_influence = TerrainInfluenceConfig(
            steep_weight=args.steep_weight,
            cliff_like_weight=args.cliff_like_weight,
            valley_weight=args.valley_weight,
            ridge_weight=args.ridge_weight,
        )
        dem, heatmap = compute_heatmap_from_heightmap(
            image_path=Path(args.input),
            georef=georef,
            start_x=args.start_x,
            start_y=args.start_y,
            max_hours=args.max_hours,
            elevation_min_m=args.elevation_min,
            elevation_max_m=args.elevation_max,
            probability_method=args.probability_method,
            use_terrain_influence=not args.disable_terrain_influence,
            terrain_influence=terrain_influence,
            terrain_steep_threshold_deg=args.terrain_steep_threshold_deg,
            terrain_cliff_threshold_deg=args.terrain_cliff_threshold_deg,
            terrain_neighborhood_size=args.terrain_neighborhood_size,
            terrain_ridge_threshold_m=args.terrain_ridge_threshold_m,
            terrain_valley_threshold_m=args.terrain_valley_threshold_m,
        )
        if args.output_dem:
            dem_path, dem_metadata_path = save_dem_tiff(dem, Path(args.output_dem))
            print(f"Saved intermediate DEM raster: {dem_path}")
            print(f"Saved intermediate DEM metadata: {dem_metadata_path}")
        raster_path, metadata_path = save_heatmap_tiff(
            heatmap,
            Path(args.output),
            layer=args.layer,
        )
        print(f"Saved heatmap raster: {raster_path}")
        print(f"Saved heatmap metadata: {metadata_path}")
        return 0

    if args.command == "classify-terrain":
        dem = load_dem_tiff(
            dem_path=Path(args.input_dem),
            metadata_path=Path(args.input_metadata) if args.input_metadata else None,
        )
        terrain = classify_terrain(
            dem,
            steep_threshold_deg=args.steep_threshold_deg,
            cliff_threshold_deg=args.cliff_threshold_deg,
            neighborhood_size=args.neighborhood_size,
            ridge_threshold_m=args.ridge_threshold_m,
            valley_threshold_m=args.valley_threshold_m,
        )
        raster_path, metadata_path = save_terrain_tiff(
            terrain,
            Path(args.output),
            layer=args.layer,
        )
        print(f"Saved terrain raster: {raster_path}")
        print(f"Saved terrain metadata: {metadata_path}")
        return 0

    if args.command == "serve-mapbox-app":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is required to run the Mapbox app. Install project dependencies first."
            ) from exc
        from .mapbox_live import _FASTAPI_AVAILABLE

        if not _FASTAPI_AVAILABLE:
            raise RuntimeError(
                "fastapi is required to run the Mapbox app. Install project dependencies first."
            )

        uvicorn.run("topo_layout.mapbox_live:app", host=args.host, port=args.port, reload=False)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
