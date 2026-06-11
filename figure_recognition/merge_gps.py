"""Merge drone GPS (DJI flight CSV) into a person-detection JSONL.

In-place: overwrites JSONL_PATH (written via temp file + atomic rename).
Run:
    python figure_recognition/merge_gps.py
"""

import bisect
import csv
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE         = Path(__file__).resolve().parent
CSV_PATH     = HERE / "results" / "DJIFlightRecord_2026-06-09_[20-07-05].csv"
JSONL_PATH   = HERE / "results" / "person_detection_output.jsonl"

# Fixed offset for the CSV's local times. Israel observes IDT (UTC+3) from late
# March to late October, which covers 6/9/2026 — so +3 is correct here.
# If you ever process a CSV that straddles a DST switch, switch to zoneinfo.
CSV_LOCAL_TZ = timezone(timedelta(hours=3))
UTC          = timezone.utc

# ---- the time-base difference between the two sources -------------------------
# CSV timestamps = the moment the drone recorded a telemetry sample (flight time).
# JSONL timestamps = wall-clock UTC when detect_live.py processed a video frame
# (see utc_now_iso() in detect_live.py). The video was processed long after the
# flight, so the two clocks differ by a fixed offset.
#
# Empirically:
#   CSV row 1   = 6/9/2026 5:07:05.140 PM local (IDT) = 2026-06-09T14:07:05.140Z
#   JSONL row 1 = 2026-06-09T17:21:40.766Z
#   offset      = JSONL - CSV  =  3 h 14 m 35.626 s  =  11675.626 s
#
# To match a JSONL detection at wall-clock time T against the CSV, we shift the
# CSV's GPS samples forward by this offset (csv_aligned = csv_utc + offset),
# i.e. "treat each CSV sample as if it were recorded JSONL_TO_CSV_OFFSET_SECONDS
# after its real flight time".
JSONL_TO_CSV_OFFSET_SECONDS = 11675.626

# Max gap (seconds) between a (shifted) CSV row and a detection timestamp.
# Detections farther than this get latitude=longitude=None.
MAX_MATCH_SECONDS = 1.0

# DJI logs (0,0) before GPS lock — skip those rather than assign 0,0 to detections.
DROP_NO_FIX = True


def load_gps_track(csv_path: Path):
    """Return a time-sorted list of (utc_datetime, lat, lon)."""
    track = []
    with csv_path.open(newline="") as f:
        first = f.readline()
        if not first.lower().startswith("sep="):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            date_s = (row.get("CUSTOM.date [local]") or "").strip()
            time_s = (row.get("CUSTOM.updateTime [local]") or "").strip()
            lat_s  = (row.get("OSD.latitude") or "").strip()
            lon_s  = (row.get("OSD.longitude") or "").strip()
            if not (date_s and time_s and lat_s and lon_s):
                continue
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
                continue
            if DROP_NO_FIX and lat == 0.0 and lon == 0.0:
                continue
            try:
                local_dt = datetime.strptime(
                    f"{date_s} {time_s}", "%m/%d/%Y %I:%M:%S.%f %p"
                ).replace(tzinfo=CSV_LOCAL_TZ)
            except ValueError:
                continue
            utc_dt = local_dt.astimezone(UTC) + timedelta(seconds=JSONL_TO_CSV_OFFSET_SECONDS)
            track.append((utc_dt, lat, lon))
    track.sort(key=lambda t: t[0])
    return track


def nearest_gps(track, times, ts):
    """Nearest (lat, lon) to UTC datetime ts, or (None, None) if outside tolerance."""
    if not track:
        return None, None
    i = bisect.bisect_left(times, ts)
    candidates = []
    if i < len(track):
        candidates.append(track[i])
    if i > 0:
        candidates.append(track[i - 1])
    best = min(candidates, key=lambda t: abs((t[0] - ts).total_seconds()))
    if abs((best[0] - ts).total_seconds()) > MAX_MATCH_SECONDS:
        return None, None
    return best[1], best[2]


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    track = load_gps_track(CSV_PATH)
    times = [t[0] for t in track]
    print(f"loaded {len(track)} GPS samples from {CSV_PATH.name}")
    if track:
        print(f"  GPS UTC range: {track[0][0].isoformat()}  ->  {track[-1][0].isoformat()}")

    matched = total = 0
    fd, tmp_path = tempfile.mkstemp(
        prefix=JSONL_PATH.stem + ".", suffix=".tmp", dir=str(JSONL_PATH.parent)
    )
    try:
        with os.fdopen(fd, "w") as out, JSONL_PATH.open() as inp:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                ts_str = rec.get("timestamp") or rec.get("ts")
                lat, lon = (None, None)
                if ts_str:
                    lat, lon = nearest_gps(track, times, parse_utc(ts_str))
                if lat is not None:
                    matched += 1
                rec["latitude"]  = lat
                rec["longitude"] = lon
                out.write(json.dumps(rec) + "\n")
        os.replace(tmp_path, JSONL_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        finally:
            raise

    print(f"updated {JSONL_PATH.name}: {matched}/{total} detections got a GPS fix")


if __name__ == "__main__":
    main()
