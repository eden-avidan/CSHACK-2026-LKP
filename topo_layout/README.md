# topo-layout

Utilities for turning terrain inputs into machine-usable elevation data for later probability and movement modeling.

## Current scope

The implemented stages currently are:

- Accepts grayscale or RGB `png` / `jpg` inputs that act as heightmaps
- Converts pixel intensity into elevation values
- Stores the resulting DEM as a TIFF image plus a JSON sidecar with geospatial metadata
- Loads a previously generated DEM artifact back into Python
- Computes slope from the DEM as both:
  - `grade` as rise over run
  - `degrees` as slope angle
- Stores the slope raster as a TIFF plus JSON sidecar
- Classifies DEM-derived terrain features:
  - `steep`
  - `cliff-like`
  - `valley`
  - `ridge`
- Stores the terrain classification as a TIFF plus JSON sidecar
- Computes Tobler-based travel time from a last known point
- Reweights the heatmap using DEM-derived terrain classes
- Converts travel time into a normalized heatmap of probability-like weights
- Stores the heatmap raster as a TIFF plus JSON sidecar
- Rejects ordinary scanned topographic maps with a clear error because that requires a larger computer-vision pipeline

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m topo_layout.cli convert-image `
  --input .\example.png `
  --output .\artifacts\dem.tif `
  --crs EPSG:32636 `
  --min-x 1000 --min-y 2000 --max-x 2000 --max-y 3000 `
  --elevation-min 0 --elevation-max 1200
```

This writes:

- `dem.tif`: TIFF raster of the elevation values
- `dem.tif.json`: metadata needed to locate the raster on the map

You can also generate a heatmap directly from a heightmap input in one step:

```powershell
python -m topo_layout.cli generate-heatmap-from-image `
  --input .\example.png `
  --crs EPSG:32636 `
  --min-x 1000 --min-y 2000 --max-x 2000 --max-y 3000 `
  --elevation-min 0 --elevation-max 1200 `
  --start-x 1500 --start-y 2500 `
  --max-hours 6 `
  --output .\artifacts\heatmap_color.png `
  --layer probability_color `
  --output-dem .\artifacts\dem_from_heightmap.tif
```

You can also run a live Mapbox-backed web app:

```powershell
$env:MAPBOX_ACCESS_TOKEN="your_backend_token"
$env:MAPBOX_PUBLIC_TOKEN="your_public_browser_token"
python -m topo_layout.cli serve-mapbox-app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/`

The web app:

- shows a live Mapbox map on the frontend
- lets you pan/zoom to the search area
- lets you click to set the last known point
- lets you enter the timestamp of the last known point
- sends the visible bbox, point, and timestamp to the backend
- fetches Mapbox Terrain-RGB tiles in the backend
- computes the elapsed time since the last known timestamp on the backend
- computes the heatmap with the existing terrain model
- refreshes the heatmap automatically every 5 minutes
- overlays the returned PNG heatmap as a live raster layer on the map

Then compute slope from that DEM:

```powershell
python -m topo_layout.cli compute-slope `
  --input-dem .\artifacts\dem.tif `
  --output .\artifacts\slope_degrees.tif `
  --layer degrees
```

Then generate a heatmap from the DEM and last known point:

```powershell
python -m topo_layout.cli generate-heatmap `
  --input-dem .\artifacts\dem.tif `
  --start-x 1500 --start-y 2500 `
  --max-hours 6 `
  --output .\artifacts\heatmap.tif `
  --layer probability `
  --probability-method linear `
  --steep-weight 0.7 `
  --cliff-like-weight 0.2 `
  --valley-weight 1.15 `
  --ridge-weight 0.9
```

To export a human-readable color heatmap image:

```powershell
python -m topo_layout.cli generate-heatmap `
  --input-dem .\artifacts\dem.tif `
  --start-x 1500 --start-y 2500 `
  --max-hours 6 `
  --output .\artifacts\heatmap_color.png `
  --layer probability_color
```

To derive terrain classes from the DEM:

```powershell
python -m topo_layout.cli classify-terrain `
  --input-dem .\artifacts\dem.tif `
  --output .\artifacts\terrain.tif `
  --layer bitmask `
  --steep-threshold-deg 30 `
  --cliff-threshold-deg 45 `
  --neighborhood-size 3 `
  --ridge-threshold-m 5 `
  --valley-threshold-m 5
```

## Notes

- A plain TIFF is used by default to keep dependencies light.
- If you later want true GeoTIFF output, we can add `rasterio` or `GDAL` and write embedded georeferencing directly into the TIFF.
- The heatmap stage assumes the DEM coordinates are in a projected CRS with meter units, because travel distance is converted into walking time.
- For heightmap inputs, you must still provide the map bounds, CRS label, and an elevation range, because ordinary image files do not carry reliable terrain metadata by themselves.
- The terrain-classification stage only infers features available from elevation geometry. It can flag steep or cliff-like terrain and local ridges or valleys, but it cannot infer roads, trails, or vegetation from a DEM alone.
- In the current heatmap model, terrain classes affect probability weighting after travel time is computed. They do not yet add roads, trails, vegetation, or drone findings into the model.
- The color heatmap export is a visualization layer. The numeric probability raster remains the authoritative machine-readable output.
- The Mapbox live app uses Mapbox Terrain-RGB in the backend and samples it into a local meter-based DEM grid over the current map view before running the heatmap model.
