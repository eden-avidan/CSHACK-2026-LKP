# figure_recognition

Drone video person detection and data preparation tools.

## Prepare Drone Detection Data

```bash
py -3.13 figure_recognition/prepare_drone_data.py
```

This merges `samples/drone_find.csv` with the closest-duration detection session
inside `samples/drone_find.json`. The no-detection baseline is generated from
`samples/good_drone_not_found.csv`. Outputs are written to
`figure_recognition/results/`:

- `drone.merged.JSON`: valid JSON array with telemetry and detections
- `drone.not_found.JSON`: the same path with all detections set to not found
- `drone.merged.jsonl`: backend-compatible JSONL
- `drone.merge_report.json`: timestamp matching and mismatch summary

The named constants `MATCH_TOLERANCE_SECONDS`, `SESSION_GAP_SECONDS`, and
`CSV_LOCAL_TIMEZONE` control timestamp matching.

## Models

Downloaded model weights belong in `models/` and are gitignored.
