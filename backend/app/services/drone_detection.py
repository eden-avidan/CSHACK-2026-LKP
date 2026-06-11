from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.geospatial.grid import ProbabilityGrid, grid_extent_m

DETECTION_JSONL_NAME = Path("figure_recognition") / "results" / "person_detection_output.jsonl"
ALTITUDE_MATCH_THRESHOLD_M = 20.0


@dataclass
class DetectionRecord:
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float | None
    person: bool = True


def get_default_detection_jsonl_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / DETECTION_JSONL_NAME


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

    records: list[DetectionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not raw.get("person", False):
                continue

            ts = _parse_timestamp(raw.get("timestamp") or raw.get("ts"))
            if ts is None:
                continue

            lat = raw.get("latitude")
            lon = raw.get("longitude")
            if lat is None or lon is None:
                continue

            try:
                lat = float(lat)
                lon = float(lon)
            except (ValueError, TypeError):
                continue

            alt = raw.get("altitude")
            if alt is None:
                alt = raw.get("alt_m")
            if alt is not None:
                try:
                    alt = float(alt)
                except (ValueError, TypeError):
                    alt = None

            records.append(DetectionRecord(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                altitude=alt,
                person=True,
            ))
    return records


def map_detection_to_grid_cell(
    grid: ProbabilityGrid,
    latitude: float,
    longitude: float,
) -> tuple[int, int]:
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


def altitude_matches_node(
    node_fields: Any,
    row: int,
    col: int,
    altitude: float | None,
) -> bool:
    if altitude is None:
        return True
    node_alt = float(node_fields.altitude[row, col])
    return abs(node_alt - altitude) <= ALTITUDE_MATCH_THRESHOLD_M
