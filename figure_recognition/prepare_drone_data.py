"""Merge DJI CSV telemetry with timestamped person detections.

Run from the repository root:
    py -3.13 figure_recognition/prepare_drone_data.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = HERE / "samples" / "drone_find.csv"
DEFAULT_DETECTION_PATH = HERE / "samples" / "drone_find.json"
DEFAULT_NOT_FOUND_CSV_PATH = HERE / "samples" / "good_drone_not_found.csv"
DEFAULT_OUTPUT_DIR = HERE / "results"

CSV_LOCAL_TIMEZONE = ZoneInfo("Asia/Jerusalem")
MATCH_TOLERANCE_SECONDS = 0.15
SESSION_GAP_SECONDS = 10.0

DETECTION_DEFAULTS = {
    "person_found": False,
    "confidence": None,
    "confidence_percent": 0.0,
    "bbox": None,
    "frame": None,
}


@dataclass
class PreparedRow:
    timestamp: datetime
    record: dict[str, Any]
    source_index: int


@dataclass
class MergeSummary:
    total_json_records: int
    selected_json_records: int
    total_csv_rows: int
    not_found_csv_rows: int
    matched_records: int
    unmatched_json_records: int
    unmatched_selected_json_records: int
    unselected_json_records: int
    unmatched_csv_rows: int
    duplicate_json_timestamps: int
    duplicate_csv_timestamps: int
    malformed_json_timestamps: int
    malformed_csv_timestamps: int
    malformed_not_found_csv_timestamps: int
    selected_session_start: str | None
    clock_offset_seconds: float | None


def parse_iso_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_csv_timestamp(row: dict[str, str]) -> datetime | None:
    date_value = (row.get("CUSTOM.date [local]") or "").strip()
    time_value = (row.get("CUSTOM.updateTime [local]") or "").strip()
    if not date_value or not time_value:
        return None
    try:
        local = datetime.strptime(
            f"{date_value} {time_value}", "%m/%d/%Y %I:%M:%S.%f %p"
        ).replace(tzinfo=CSV_LOCAL_TIMEZONE)
    except ValueError:
        return None
    return local.astimezone(timezone.utc)


def load_csv_rows(path: Path) -> tuple[list[PreparedRow], int]:
    prepared: list[PreparedRow] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        if not first.lower().startswith("sep="):
            handle.seek(0)
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            timestamp = parse_csv_timestamp(row)
            if timestamp is None:
                malformed += 1
                continue
            prepared.append(PreparedRow(timestamp, dict(row), index))
    return prepared, malformed


def load_detection_rows(path: Path) -> tuple[list[PreparedRow], int]:
    text = path.read_text(encoding="utf-8-sig")
    raw_records: list[dict[str, Any]]
    try:
        parsed = json.loads(text)
        raw_records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        raw_records = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                raw_records.append({})
                continue
            raw_records.append(value if isinstance(value, dict) else {})

    prepared: list[PreparedRow] = []
    malformed = 0
    for index, record in enumerate(raw_records):
        timestamp = parse_iso_timestamp(record.get("timestamp") or record.get("ts"))
        if timestamp is None:
            malformed += 1
            continue
        prepared.append(PreparedRow(timestamp, record, index))
    return prepared, malformed


def duplicate_count(rows: list[PreparedRow]) -> int:
    seen: set[datetime] = set()
    duplicates = 0
    for row in rows:
        if row.timestamp in seen:
            duplicates += 1
        seen.add(row.timestamp)
    return duplicates


def split_sessions(rows: list[PreparedRow]) -> list[list[PreparedRow]]:
    sessions: list[list[PreparedRow]] = []
    current: list[PreparedRow] = []
    for row in sorted(rows, key=lambda item: item.timestamp):
        if current and (row.timestamp - current[-1].timestamp).total_seconds() > SESSION_GAP_SECONDS:
            sessions.append(current)
            current = []
        current.append(row)
    if current:
        sessions.append(current)
    return sessions


def select_matching_session(
    detections: list[PreparedRow], csv_rows: list[PreparedRow]
) -> list[PreparedRow]:
    if not detections or not csv_rows:
        return []
    csv_duration = (csv_rows[-1].timestamp - csv_rows[0].timestamp).total_seconds()
    return min(
        split_sessions(detections),
        key=lambda session: abs(
            (session[-1].timestamp - session[0].timestamp).total_seconds() - csv_duration
        ),
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _output_record(
    csv_row: PreparedRow, aligned_timestamp: datetime, detection: PreparedRow | None
) -> dict[str, Any]:
    altitude_ft = _float_or_none(csv_row.record.get("OSD.altitude [ft]"))
    record: dict[str, Any] = {
        "timestamp": aligned_timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "source_timestamp": csv_row.timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "latitude": _float_or_none(csv_row.record.get("OSD.latitude")),
        "longitude": _float_or_none(csv_row.record.get("OSD.longitude")),
        "altitude": altitude_ft * 0.3048 if altitude_ft is not None else None,
        "csv_row": csv_row.record,
        **DETECTION_DEFAULTS,
    }
    if detection is None:
        return record

    person_found = bool(detection.record.get("person_found", detection.record.get("person", False)))
    confidence = _float_or_none(detection.record.get("confidence")) if person_found else None
    record.update(
        {
            "person_found": person_found,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100.0, 4) if confidence is not None else 0.0,
            "bbox": detection.record.get("bbox") if person_found else None,
            "frame": detection.record.get("frame"),
        }
    )
    return record


def merge_records(
    csv_rows: list[PreparedRow],
    detections: list[PreparedRow],
    *,
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], MergeSummary]:
    session = select_matching_session(detections, csv_rows)
    if not csv_rows or not session:
        raise ValueError("Both CSV rows and a matching detection session are required")

    offset = session[0].timestamp - csv_rows[0].timestamp
    aligned_times = [row.timestamp + offset for row in csv_rows]
    matches: dict[int, PreparedRow] = {}
    matched_detection_indices: set[int] = set()

    for detection in session:
        position = bisect.bisect_left(aligned_times, detection.timestamp)
        candidates = [i for i in (position - 1, position) if 0 <= i < len(csv_rows)]
        if not candidates:
            continue
        best_index = min(
            candidates,
            key=lambda index: abs((aligned_times[index] - detection.timestamp).total_seconds()),
        )
        gap = abs((aligned_times[best_index] - detection.timestamp).total_seconds())
        if gap > tolerance_seconds or best_index in matches:
            continue
        matches[best_index] = detection
        matched_detection_indices.add(detection.source_index)

    merged = [
        _output_record(row, aligned_times[index], matches.get(index))
        for index, row in enumerate(csv_rows)
    ]
    not_found = [
        _output_record(row, aligned_times[index], None)
        for index, row in enumerate(csv_rows)
    ]
    summary = MergeSummary(
        total_json_records=len(detections),
        selected_json_records=len(session),
        total_csv_rows=len(csv_rows),
        not_found_csv_rows=0,
        matched_records=len(matches),
        unmatched_json_records=len(detections) - len(matched_detection_indices),
        unmatched_selected_json_records=len(session) - len(matched_detection_indices),
        unselected_json_records=len(detections) - len(session),
        unmatched_csv_rows=len(csv_rows) - len(matches),
        duplicate_json_timestamps=duplicate_count(detections),
        duplicate_csv_timestamps=duplicate_count(csv_rows),
        malformed_json_timestamps=0,
        malformed_csv_timestamps=0,
        malformed_not_found_csv_timestamps=0,
        selected_session_start=session[0].timestamp.isoformat(),
        clock_offset_seconds=offset.total_seconds(),
    )
    return merged, not_found, summary


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def generate(
    csv_path: Path,
    detection_path: Path,
    not_found_csv_path: Path,
    output_dir: Path,
    *,
    tolerance_seconds: float = MATCH_TOLERANCE_SECONDS,
) -> MergeSummary:
    csv_rows, malformed_csv = load_csv_rows(csv_path)
    not_found_csv_rows, malformed_not_found_csv = load_csv_rows(not_found_csv_path)
    detections, malformed_json = load_detection_rows(detection_path)
    merged, _, summary = merge_records(
        csv_rows, detections, tolerance_seconds=tolerance_seconds
    )
    not_found = [
        _output_record(row, row.timestamp, None)
        for row in not_found_csv_rows
    ]
    summary.not_found_csv_rows = len(not_found_csv_rows)
    summary.malformed_csv_timestamps = malformed_csv
    summary.malformed_not_found_csv_timestamps = malformed_not_found_csv
    summary.malformed_json_timestamps = malformed_json

    write_json(output_dir / "drone.merged.JSON", merged)
    write_json(output_dir / "drone.not_found.JSON", not_found)
    write_jsonl(output_dir / "drone.merged.jsonl", merged)
    (output_dir / "drone.merge_report.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTION_PATH)
    parser.add_argument("--not-found-csv", type=Path, default=DEFAULT_NOT_FOUND_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance-seconds", type=float, default=MATCH_TOLERANCE_SECONDS)
    args = parser.parse_args()

    summary = generate(
        args.csv,
        args.detections,
        args.not_found_csv,
        args.output_dir,
        tolerance_seconds=args.tolerance_seconds,
    )
    print(json.dumps(asdict(summary), indent=2))
    print(f"wrote {args.output_dir / 'drone.merged.JSON'}")
    print(f"wrote {args.output_dir / 'drone.not_found.JSON'}")
    print(f"wrote backend-compatible {args.output_dir / 'drone.merged.jsonl'}")


if __name__ == "__main__":
    main()
