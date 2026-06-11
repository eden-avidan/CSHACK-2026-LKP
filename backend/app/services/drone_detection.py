from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.geospatial.grid import ProbabilityGrid, cell_centroid_utm, grid_extent_m

DETECTION_JSONL_NAME = Path("figure_recognition") / "results" / "person_detection_output.jsonl"
DRONE_TRACK_JSONL_NAME = Path("figure_recognition") / "results" / "synthetic_drone_track.jsonl"


@dataclass
class DetectionRecord:
    timestamp: datetime
    latitude: float | None
    longitude: float | None
    altitude: float | None
    person: bool = True
    confidence: float = 0.0
    confidence_percent: float = 0.0
    frame: int | None = None
    bbox: list[float] | None = None


def get_default_detection_jsonl_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / DETECTION_JSONL_NAME


def get_default_drone_track_jsonl_path() -> Path:
    """Synthetic 'drone searched here, found nobody' track (person_found=false)."""
    root = Path(__file__).resolve().parents[3]
    return root / DRONE_TRACK_JSONL_NAME


def get_drone_sortie_paths() -> list[Path]:
    """Ordered real-drone sortie files (played back one after another).

    Resolved from ``settings.drone_sortie_files`` relative to the repo root.
    """
    root = Path(__file__).resolve().parents[3]
    return [root / rel for rel in settings.drone_sortie_file_list]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_detection_records(path: Path) -> list[DetectionRecord]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(text)
        raw_records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        raw_records = []
        for line in text.splitlines():
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    records: list[DetectionRecord] = []
    for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            # Keep both outcomes: person-found rows drive detection events, while
            # person-not-found rows (with a GPS fix) mark "clean" cells the drone
            # overflew without spotting anyone.
            person_found = bool(raw.get("person_found", raw.get("person", False)))

            ts = _parse_timestamp(raw.get("timestamp") or raw.get("ts"))
            if ts is None:
                continue

            lat = raw.get("latitude")
            lon = raw.get("longitude")
            if lat is not None and lon is not None:
                try:
                    lat = float(lat)
                    lon = float(lon)
                except (ValueError, TypeError):
                    lat = lon = None
            else:
                lat = lon = None

            alt = raw.get("altitude")
            if alt is None:
                alt = raw.get("alt_m")
            if alt is not None:
                try:
                    alt = float(alt)
                except (ValueError, TypeError):
                    alt = None

            confidence = raw.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else 0.0
            except (ValueError, TypeError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            confidence_percent = raw.get("confidence_percent")
            try:
                confidence_percent = (
                    float(confidence_percent)
                    if confidence_percent is not None
                    else confidence * 100.0
                )
            except (ValueError, TypeError):
                confidence_percent = confidence * 100.0

            frame = raw.get("frame")
            try:
                frame = int(frame) if frame is not None else None
            except (ValueError, TypeError):
                frame = None

            bbox = raw.get("bbox")
            if isinstance(bbox, list):
                try:
                    bbox = [float(value) for value in bbox]
                except (ValueError, TypeError):
                    bbox = None
            else:
                bbox = None

            records.append(DetectionRecord(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                person=person_found,
                confidence=confidence,
                confidence_percent=max(0.0, min(100.0, confidence_percent)),
                frame=frame,
                bbox=bbox,
            ))
    return records


# Parsed-record cache keyed by file path, invalidated on mtime change. The real
# sortie files are large (tens of MB) and are read once per tick, so reparsing
# every tick would be wasteful.
_RECORD_CACHE: dict[str, tuple[float, list[DetectionRecord]]] = {}


def load_detection_records_cached(path: Path) -> list[DetectionRecord]:
    """Like :func:`load_detection_records` but cached by path + mtime and sorted by time."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    key = str(path)
    cached = _RECORD_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    records = load_detection_records(path)
    records = [r for r in records if r.timestamp is not None]
    records.sort(key=lambda r: r.timestamp)
    _RECORD_CACHE[key] = (mtime, records)
    return records


def subsample_records(
    records: list[DetectionRecord], max_points: int
) -> list[DetectionRecord]:
    """Thin a dense flight log down to ``max_points`` while keeping every
    person-found record (those drive detection events) and the endpoints."""
    if max_points <= 0 or len(records) <= max_points:
        return records
    found = [r for r in records if r.person]
    others = [r for r in records if not r.person]
    keep = max_points - len(found)
    if keep <= 0:
        sampled = list(found)
    else:
        step = len(others) / keep
        sampled = found + [others[min(len(others) - 1, int(i * step))] for i in range(keep)]
    sampled.sort(key=lambda r: r.timestamp)
    return sampled


def map_detection_to_grid_cell(
    grid: ProbabilityGrid,
    latitude: float,
    longitude: float,
) -> tuple[int, int]:
    """Map a WGS84 ``(latitude, longitude)`` fix to a grid ``(row, col)``.

    This is a purely 2D, planimetric lookup: the drone's flight altitude is
    irrelevant to *which ground cell it overflew* and must never be mixed in
    here. Internally we project lon/lat -> UTM (always_xy), so ``to_utm`` is
    called as ``(longitude, latitude)`` even though the public argument order
    is ``(latitude, longitude)``.
    """
    e, n = grid.crs.to_utm(longitude, latitude)
    west, _east, _south, north = grid_extent_m(grid.rows, grid.metadata.resolution_m)
    res = grid.metadata.resolution_m
    min_e = grid.crs.origin_e - west

    cols = (e - min_e) / res
    rows = ((grid.crs.origin_n + north) - n) / res
    row = int(np.floor(rows))
    col = int(np.floor(cols))

    if not (0 <= row < grid.rows and 0 <= col < grid.cols):
        raise ValueError("Detection outside grid bounds")
    return row, col


def cells_within_radius(
    grid: ProbabilityGrid,
    latitude: float,
    longitude: float,
    radius_m: float,
) -> list[tuple[int, int]]:
    """Grid cells whose centroid lies within ``radius_m`` of a WGS84 point.

    Models the drone's ground coverage footprint (camera swath) as a disc, so a
    single GPS fix marks a realistic area rather than one cell. Distances are
    measured in UTM meters (reusing the grid's CRS), never in raw degrees. The
    cell directly under the point is always included; cells outside the grid are
    skipped. Raises ``ValueError`` if the point itself falls outside the grid.
    """
    center = map_detection_to_grid_cell(grid, latitude, longitude)
    if radius_m <= 0.0:
        return [center]

    e, n = grid.crs.to_utm(longitude, latitude)
    res = grid.metadata.resolution_m
    reach = int(math.ceil(radius_m / res))
    r0, c0 = center
    radius_sq = radius_m * radius_m

    cells: list[tuple[int, int]] = []
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            row, col = r0 + dr, c0 + dc
            if not (0 <= row < grid.rows and 0 <= col < grid.cols):
                continue
            ce, cn = cell_centroid_utm(grid, row, col)
            if (ce - e) ** 2 + (cn - n) ** 2 <= radius_sq:
                cells.append((row, col))

    return cells or [center]
