from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.geospatial.grid import ProbabilityGrid, grid_extent_m

DETECTION_JSONL_NAME = Path("figure_recognition") / "results" / "person_detection_output.jsonl"
DRONE_TRACK_JSONL_NAME = Path("figure_recognition") / "results" / "synthetic_drone_track.jsonl"
ALTITUDE_MATCH_THRESHOLD_M = 20.0


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


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


<<<<<<< HEAD
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

            # Keep both outcomes: person-found rows drive detection events, while
            # person-not-found rows (with a GPS fix) mark "clean" cells the drone
            # overflew without spotting anyone.
            person_found = bool(raw.get("person_found", raw.get("person", False)))
=======
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
            if not raw.get("person_found", raw.get("person", False)):
                continue
>>>>>>> 49d662201c706af5a6f8b324353f75b011ecf7f3

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
