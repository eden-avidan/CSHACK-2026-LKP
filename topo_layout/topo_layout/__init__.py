"""Terrain preprocessing utilities."""

from .mobility import (
    HeatmapRaster,
    TerrainInfluenceConfig,
    compute_heatmap,
    compute_heatmap_from_heightmap,
    coordinate_to_row_col,
    render_heatmap_image,
    save_heatmap_tiff,
    tobler_hiking_speed_kmh,
)
from .preprocessing import (
    DemRaster,
    GeoReference,
    SlopeRaster,
    UnsupportedTopographicImageError,
    compute_slope,
    convert_image_to_dem,
    import_geotiff_dem,
    import_worldfile_dem,
    load_dem_tiff,
    save_dem_tiff,
    save_slope_tiff,
)
from .terrain import TerrainRaster, classify_terrain, save_terrain_tiff

__all__ = [
    "HeatmapRaster",
    "TerrainInfluenceConfig",
    "DemRaster",
    "GeoReference",
    "SlopeRaster",
    "TerrainRaster",
    "compute_heatmap",
    "compute_heatmap_from_heightmap",
    "coordinate_to_row_col",
    "render_heatmap_image",
    "UnsupportedTopographicImageError",
    "classify_terrain",
    "save_heatmap_tiff",
    "save_terrain_tiff",
    "tobler_hiking_speed_kmh",
    "compute_slope",
    "convert_image_to_dem",
    "import_geotiff_dem",
    "import_worldfile_dem",
    "load_dem_tiff",
    "save_dem_tiff",
    "save_slope_tiff",
]
