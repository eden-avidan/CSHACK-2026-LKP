import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from figure_recognition.prepare_drone_data import (
    PreparedRow,
    generate,
    load_detection_rows,
    merge_records,
)


def _row(seconds: float) -> PreparedRow:
    start = datetime(2026, 6, 11, 10, 0, tzinfo=timezone.utc)
    return PreparedRow(
        timestamp=start + timedelta(seconds=seconds),
        source_index=int(seconds * 10),
        record={
            "CUSTOM.date [local]": "6/11/2026",
            "CUSTOM.updateTime [local]": "1:00:00.00 PM",
            "OSD.latitude": "32.7",
            "OSD.longitude": "35.0",
            "OSD.altitude [ft]": "100",
        },
    )


def _detection(seconds: float, index: int = 0) -> PreparedRow:
    start = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
    return PreparedRow(
        timestamp=start + timedelta(seconds=seconds),
        source_index=index,
        record={"person": True, "confidence": 0.8, "bbox": [1, 2, 3, 4], "frame": 7},
    )


def test_nearest_timestamp_matching_and_defaults():
    csv_rows = [_row(0.0), _row(0.1), _row(0.2)]
    merged, not_found, summary = merge_records(csv_rows, [_detection(0.11)])

    assert summary.matched_records == 1
    assert merged[0]["person_found"] is True
    assert merged[0]["confidence_percent"] == 80.0
    assert merged[1]["person_found"] is False
    assert all(record["person_found"] is False for record in not_found)


def test_outside_tolerance_is_reported_unmatched():
    csv_rows = [_row(0.0), _row(0.1), _row(0.2)]
    detections = [_detection(0.0, 0), _detection(0.5, 1)]
    _, _, summary = merge_records(csv_rows, detections, tolerance_seconds=0.05)

    assert summary.matched_records == 1
    assert summary.unmatched_json_records == 1
    assert summary.unmatched_selected_json_records == 1
    assert summary.unmatched_csv_rows == 2


def test_malformed_timestamp_is_counted(tmp_path: Path):
    path = tmp_path / "detections.json"
    path.write_text('{"ts":"bad","person":true}\n{"ts":"2026-06-11T10:00:00Z","person":true}\n')

    rows, malformed = load_detection_rows(path)

    assert len(rows) == 1
    assert malformed == 1


def test_generate_creates_json_and_not_found_outputs(tmp_path: Path):
    csv_path = tmp_path / "drone.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "CUSTOM.date [local]",
                "CUSTOM.updateTime [local]",
                "OSD.latitude",
                "OSD.longitude",
                "OSD.altitude [ft]",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "CUSTOM.date [local]": "6/11/2026",
                "CUSTOM.updateTime [local]": "1:00:00.00 PM",
                "OSD.latitude": "32.7",
                "OSD.longitude": "35.0",
                "OSD.altitude [ft]": "100",
            }
        )
    detection_path = tmp_path / "drone.json"
    detection_path.write_text(
        '{"ts":"2026-06-11T15:00:00Z","person":true,"confidence":0.9,"bbox":[1,2,3,4]}\n'
    )

    summary = generate(csv_path, detection_path, csv_path, tmp_path / "out")
    merged = json.loads((tmp_path / "out" / "drone.merged.JSON").read_text())
    not_found = json.loads((tmp_path / "out" / "drone.not_found.JSON").read_text())

    assert summary.matched_records == 1
    assert merged[0]["person_found"] is True
    assert not_found[0]["person_found"] is False
    assert (tmp_path / "out" / "drone.merged.jsonl").exists()
    assert (tmp_path / "out" / "drone.merge_report.json").exists()
