"""Tests for drone last-seen updates during mission ticks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.core.config import settings
from app.engine.grid_matrix import GridMatrix
from app.models.mission import LatLon, MissionMode
from app.models.detection import DetectionEventMessage
from app.api.ws.mission import broadcast_tick_result
from app.geospatial.grid import lkp_cell_indices
from app.services.mission_store import MissionStore
from app.services.mission_store import TickResult
from app.services.drone_detection import (
    DetectionRecord,
    cells_within_radius,
    load_detection_records,
    map_detection_to_grid_cell,
)

HAIFA = LatLon(lat=32.7940, lon=34.9896)


def _terrain(size: int):
    from app.services.env_ingestion import TerrainContext

    return TerrainContext(
        elevation=np.zeros((size, size)),
        slope=np.zeros((size, size)),
        aspect_n=np.zeros((size, size)),
        aspect_e=np.zeros((size, size)),
        road_proximity=np.zeros((size, size)),
        is_land=np.ones((size, size), dtype=bool),
        road_tangent_e=np.zeros((size, size)),
        road_tangent_n=np.zeros((size, size)),
        reachability=None,
    )


def test_load_detection_records(tmp_path: Path):
    path = tmp_path / "detections.jsonl"
    path.write_text(
        "{\"ts\": \"2026-06-11T16:00:00.000Z\", \"person\": true, \"latitude\": 32.7940, \"longitude\": 34.9896, \"altitude\": 0.0}\n"
        "{\"ts\": \"2026-06-11T16:00:01.000Z\", \"person\": false, \"latitude\": 32.7950, \"longitude\": 34.9900, \"altitude\": 0.0}\n"
    )

    records = load_detection_records(path)
    # Both outcomes are kept: found rows drive detections, not-found rows mark clean cells.
    assert len(records) == 2
    assert records[0].person is True
    assert records[0].latitude == pytest.approx(32.7940)
    assert records[0].longitude == pytest.approx(34.9896)
    assert records[1].person is False
    assert records[1].latitude == pytest.approx(32.7950)


def test_load_person_found_detection_payload_without_gps(tmp_path: Path):
    path = tmp_path / "detections.jsonl"
    path.write_text(
        "{\"timestamp\": \"2026-06-11T16:00:00.000Z\", \"frame\": 42, "
        "\"person_found\": true, \"confidence\": 0.9386, "
        "\"confidence_percent\": 93.86, \"bbox\": [1, 2, 3, 4], "
        "\"latitude\": null, \"longitude\": null}\n"
    )

    records = load_detection_records(path)
    assert len(records) == 1
    assert records[0].latitude == pytest.approx(32.7940)
    assert records[0].longitude == pytest.approx(34.9896)


def test_load_person_found_detection_payload_without_gps(tmp_path: Path):
    path = tmp_path / "detections.jsonl"
    path.write_text(
        "{\"timestamp\": \"2026-06-11T16:00:00.000Z\", \"frame\": 42, "
        "\"person_found\": true, \"confidence\": 0.9386, "
        "\"confidence_percent\": 93.86, \"bbox\": [1, 2, 3, 4], "
        "\"latitude\": null, \"longitude\": null}\n"
    )

    records = load_detection_records(path)
    assert len(records) == 1
    assert records[0].confidence == pytest.approx(0.9386)
    assert records[0].confidence_percent == pytest.approx(93.86)
    assert records[0].frame == 42
    assert records[0].bbox == [1, 2, 3, 4]
    assert records[0].latitude is None


def test_load_detection_records_from_json_array(tmp_path: Path):
    path = tmp_path / "detections.JSON"
    path.write_text(
        '[{"timestamp":"2026-06-11T16:00:00Z","person_found":true,'
        '"confidence":0.8,"confidence_percent":80.0,"latitude":32.7,"longitude":35.0}]'
    )

    records = load_detection_records(path)

    assert len(records) == 1
    assert records[0].confidence_percent == pytest.approx(80.0)


# --- coordinate mapping: lat/lon order, altitude, fixture ------------------

# Known fixture: a 128x128 grid at 50 m resolution anchored on the Haifa LKP.
# The LKP sits at the matrix center; offsets below are hand-checked against the
# 111_320 m/deg latitude scale (~1 cell == 50 m == ~0.00045 deg lat).
_GRID_SIZE = 128
_GRID_RES_M = 50.0


def _haifa_grid() -> GridMatrix:
    return GridMatrix.create(HAIFA, size=_GRID_SIZE, resolution_m=_GRID_RES_M)


def test_lkp_coordinate_maps_to_center_cell():
    grid = _haifa_grid().grid
    row, col = map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon)
    assert (row, col) == lkp_cell_indices(_GRID_SIZE)


def test_known_coordinate_maps_to_expected_nearby_cell():
    """A point ~150 m north and ~150 m east of the LKP lands 3 cells away."""
    grid = _haifa_grid().grid
    center_row, center_col = lkp_cell_indices(_GRID_SIZE)
    # ~150 m north, ~150 m east of the LKP.
    lat = HAIFA.lat + 150.0 / 111_320.0
    import math

    lon = HAIFA.lon + 150.0 / (111_320.0 * math.cos(math.radians(HAIFA.lat)))
    row, col = map_detection_to_grid_cell(grid, lat, lon)
    # North -> smaller row (row 0 is the north edge); East -> larger col.
    assert row == center_row - 3
    assert col == center_col + 3


def test_latlon_order_is_not_swapped():
    """Latitude drives the row axis, longitude the column axis."""
    grid = _haifa_grid().grid
    center_row, center_col = map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon)

    north = map_detection_to_grid_cell(grid, HAIFA.lat + 0.001, HAIFA.lon)
    east = map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon + 0.001)

    # Moving north changes only the row (and decreases it); moving east changes
    # only the column (and increases it). If lat/lon were swapped these axes
    # would be exchanged.
    assert north[0] < center_row and north[1] == center_col
    assert east[1] > center_col and east[0] == center_row


def test_swapped_latlon_lands_in_a_different_place():
    """Feeding (lon, lat) instead of (lat, lon) must NOT resolve to the LKP."""
    grid = _haifa_grid().grid
    correct = map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon)
    with pytest.raises(ValueError):
        # 34.98.. as a latitude / 32.79.. as a longitude is far outside the grid.
        map_detection_to_grid_cell(grid, HAIFA.lon, HAIFA.lat)
    assert correct == lkp_cell_indices(_GRID_SIZE)


def test_altitude_does_not_affect_2d_cell_mapping():
    """Cell mapping is planimetric: it takes no altitude and is unchanged by it.

    Regression guard for the bug where a drone's flight altitude (e.g. ~505 m
    ASL) was compared against the node's ground DEM elevation, causing real
    overflights to be silently dropped.
    """
    store = MissionStore()
    matrix = _haifa_grid()
    clean = matrix.node_fields.searched_clean
    # Give the LKP cell a realistic ground elevation so a naive altitude check
    # against the drone's flight altitude would fail.
    center_row, center_col = lkp_cell_indices(_GRID_SIZE)
    matrix.node_fields.altitude[center_row, center_col] = 150.0

    state = type("S", (), {})()
    state.grid_matrix = matrix

    high_altitude_record = DetectionRecord(
        timestamp=datetime.now(timezone.utc),
        latitude=HAIFA.lat,
        longitude=HAIFA.lon,
        altitude=505.0,  # ~1658 ft, a real DJI flight altitude
        person=False,
    )
    position = LatLon(lat=HAIFA.lat, lon=HAIFA.lon)
    store._mark_clean_cell(state, clean, high_altitude_record, position)

    assert clean[center_row, center_col] == pytest.approx(1.0)


# --- coverage footprint: a fix clears a disc, not a single cell -------------


def test_zero_radius_clears_only_the_single_cell():
    grid = _haifa_grid().grid
    cells = cells_within_radius(grid, HAIFA.lat, HAIFA.lon, 0.0)
    assert cells == [map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon)]


def test_coverage_footprint_clears_disc_of_cells():
    grid = _haifa_grid().grid
    center = map_detection_to_grid_cell(grid, HAIFA.lat, HAIFA.lon)
    radius = 80.0
    cells = cells_within_radius(grid, HAIFA.lat, HAIFA.lon, radius)

    # More than one cell, and the cell under the point is included.
    assert len(cells) > 1
    assert center in cells

    # Every returned cell's centroid is genuinely within the radius (UTM meters).
    from app.geospatial.grid import cell_centroid_utm

    e, n = grid.crs.to_utm(HAIFA.lon, HAIFA.lat)
    for row, col in cells:
        ce, cn = cell_centroid_utm(grid, row, col)
        assert ((ce - e) ** 2 + (cn - n) ** 2) ** 0.5 <= radius + 1e-9

    # An 80 m disc on a 50 m grid is the 3x3 block around the center.
    assert len(cells) == 9


def test_larger_radius_clears_more_cells():
    grid = _haifa_grid().grid
    small = cells_within_radius(grid, HAIFA.lat, HAIFA.lon, 80.0)
    large = cells_within_radius(grid, HAIFA.lat, HAIFA.lon, 200.0)
    assert len(large) > len(small)


def test_mark_clean_cell_uses_coverage_footprint():
    """A 'no person' fix clears its whole footprint; a detection stays point-like."""
    store = MissionStore()
    center_row, center_col = lkp_cell_indices(_GRID_SIZE)

    # No-person sweep -> footprint disc.
    matrix = _haifa_grid()
    clean = matrix.node_fields.searched_clean
    state = type("S", (), {})()
    state.grid_matrix = matrix
    sweep = DetectionRecord(
        timestamp=datetime.now(timezone.utc),
        latitude=HAIFA.lat,
        longitude=HAIFA.lon,
        altitude=505.0,
        person=False,
    )
    store._mark_clean_cell(state, clean, sweep, LatLon(lat=HAIFA.lat, lon=HAIFA.lon))
    assert int((clean >= 1.0 - 1e-9).sum()) == 9
    assert clean[center_row, center_col] == pytest.approx(1.0)

    # Positive detection -> only its own cell is touched (set back to 0.0).
    matrix2 = _haifa_grid()
    clean2 = matrix2.node_fields.searched_clean
    clean2.fill(1.0)
    state2 = type("S", (), {})()
    state2.grid_matrix = matrix2
    found = DetectionRecord(
        timestamp=datetime.now(timezone.utc),
        latitude=HAIFA.lat,
        longitude=HAIFA.lon,
        altitude=505.0,
        person=True,
    )
    store._mark_clean_cell(state2, clean2, found, LatLon(lat=HAIFA.lat, lon=HAIFA.lon))
    assert clean2[center_row, center_col] == pytest.approx(0.0)
    assert int((clean2 <= 1e-9).sum()) == 1


def test_tick_marks_searched_clean(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_ts = two_hours_ago + timedelta(seconds=5)
        path = tmp_path / "person_detection_output.jsonl"
        # Drone overflew this spot and found no one -> the cell becomes "clean".
        path.write_text(
            f"{{\"ts\": \"{sample_ts.isoformat().replace('+00:00','Z')}\", "
            "\"person\": false, \"latitude\": 32.7940, \"longitude\": 34.9896, \"altitude\": 0.0}\n"
        )

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=path), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[]), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            await store.tick(state.mission_id)

            clean = state.grid_matrix.node_fields.searched_clean
            assert clean.dtype == np.float64
            assert clean.max() == pytest.approx(1.0)
            assert clean.sum() >= 1.0

    asyncio.run(run())


def test_tick_clean_suppresses_probability(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_ts = two_hours_ago + timedelta(seconds=5)
        path = tmp_path / "person_detection_output.jsonl"
        # A fully-clean (no person) cell with full suppression strength should be
        # driven toward zero probability after the tick.
        path.write_text(
            f"{{\"ts\": \"{sample_ts.isoformat().replace('+00:00','Z')}\", "
            "\"person\": false, \"latitude\": 32.7940, \"longitude\": 34.9896, \"altitude\": 0.0}\n"
        )

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=path), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[]), \
             patch.object(settings, "drone_clean_suppression_strength", 1.0), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            await store.tick(state.mission_id)

            probs = state.grid_matrix.probabilities
            clean = state.grid_matrix.node_fields.searched_clean
            cleared = clean >= 1.0 - 1e-9
            assert cleared.any()
            assert float(probs[cleared].max()) == pytest.approx(0.0, abs=1e-9)
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    asyncio.run(run())


def test_tick_with_high_altitude_drone_reduces_local_probability(tmp_path: Path):
    """End-to-end regression: a real-style record carrying a high flight
    altitude must still suppress probability around the drone's cell.

    Before the fix, the altitude gate (drone ~505 m ASL vs ground DEM) dropped
    the clear, so probability was never reduced along the actual drone path.
    """

    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_ts = two_hours_ago + timedelta(seconds=5)
        path = tmp_path / "person_detection_output.jsonl"
        # person=false overflight WITH a realistic flight altitude (~1658 ft).
        path.write_text(
            f"{{\"ts\": \"{sample_ts.isoformat().replace('+00:00','Z')}\", "
            "\"person\": false, \"latitude\": 32.7940, \"longitude\": 34.9896, "
            "\"altitude\": 505.0}\n"
        )

        # Non-flat terrain so the LKP cell has a real ground elevation that a
        # naive altitude check would compare against (and fail).
        terrain = _terrain(settings.grid_size)
        terrain.elevation[:] = 150.0

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=path), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[]), \
             patch.object(settings, "drone_clean_suppression_strength", 1.0), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = terrain
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            await store.tick(state.mission_id)

            probs = state.grid_matrix.probabilities
            clean = state.grid_matrix.node_fields.searched_clean
            cleared = clean >= 1.0 - 1e-9

            # The high-altitude overflight still clears its cell...
            assert cleared.any()
            # ...and the clear is local, not smeared across the whole grid.
            assert int(cleared.sum()) < 0.01 * probs.size
            # ...and drives probability there to zero (full suppression strength).
            assert float(probs[cleared].max()) == pytest.approx(0.0, abs=1e-9)
            # The reduction is local to the drone path: probability survives
            # elsewhere, so the global peak is NOT in a cleared cell.
            peak = np.unravel_index(int(np.argmax(probs)), probs.shape)
            assert not cleared[peak]
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    asyncio.run(run())


def test_tick_returns_detection_event(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        start = datetime.now(timezone.utc) - timedelta(seconds=120)
        sample_ts = start + timedelta(seconds=5)
        path = tmp_path / "person_detection_output.jsonl"
        path.write_text(
            f"{{\"timestamp\": \"{sample_ts.isoformat().replace('+00:00','Z')}\", "
            "\"frame\": 17, \"person_found\": true, \"confidence\": 0.91, "
            "\"confidence_percent\": 91.0, \"bbox\": [10, 20, 30, 40], "
            "\"latitude\": null, \"longitude\": null}\n"
        )

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=path), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[]), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=start
            )
            result = await store.tick(state.mission_id)

            assert len(result.detection_events) == 1
            event = result.detection_events[0]
            assert event.person_found is True
            assert event.confidence_percent == pytest.approx(91.0)
            assert event.frame == 17
            assert event.position is None

    asyncio.run(run())


def test_tick_broadcasts_detection_event():
    async def run() -> None:
        mission_id = uuid4()
        event = DetectionEventMessage(
            mission_id=mission_id,
            timestamp=datetime.now(timezone.utc),
            confidence=0.87,
            confidence_percent=87.0,
            frame=12,
        )
        result = TickResult(deltas=[], engine_tick=None, detection_events=[event])

        with patch("app.api.ws.mission.mission_store.broadcast", new_callable=AsyncMock) as broadcast:
            await broadcast_tick_result(mission_id, result)

        broadcast.assert_awaited_once()
        payload = broadcast.await_args.args[1]
        assert payload["type"] == "detection_event"
        assert payload["confidence_percent"] == pytest.approx(87.0)

    asyncio.run(run())


def test_tick_emits_drone_track(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        # Synthetic track: person_found=false points near the LKP, with timestamps
        # interpreted relative to mission start (base = first record).
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        track_path = tmp_path / "synthetic_drone_track.jsonl"
        empty_feed = tmp_path / "person_detection_output.jsonl"
        empty_feed.write_text("")
        lines = []
        for i, off in enumerate((0, 8, 16)):
            ts = (base + timedelta(seconds=off)).isoformat().replace("+00:00", "Z")
            lat = HAIFA.lat + 0.0001 * i
            lon = HAIFA.lon + 0.0001 * i
            lines.append(
                f'{{"timestamp": "{ts}", "person_found": false, '
                f'"latitude": {lat}, "longitude": {lon}}}'
            )
        track_path.write_text("\n".join(lines) + "\n")

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=empty_feed), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[track_path]), \
             patch.object(settings, "drone_track_launch_delay_sec", 0.0), \
             patch.object(settings, "drone_sortie_north_offsets_m", "0"), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            # step_sec defaults to 10s; tick 1 reveals offsets in [0, 10) -> 2 points.
            result = await store.tick(state.mission_id)

            assert result.drone_track is not None
            assert result.drone_track.position is not None
            assert len(result.drone_track.path) == 2
            # Each path point is [lon, lat].
            assert result.drone_track.path[0] == pytest.approx([HAIFA.lon, HAIFA.lat])
            # A single sortie surfaces as one drone in the list.
            assert len(result.drone_track.drones) == 1
            assert result.drone_track.drones[0].asset_id == "drone-1"
            clean = state.grid_matrix.node_fields.searched_clean
            assert clean.max() == pytest.approx(1.0)

    asyncio.run(run())


def test_drone_track_launch_delay(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        track_path = tmp_path / "synthetic_drone_track.jsonl"
        empty_feed = tmp_path / "person_detection_output.jsonl"
        empty_feed.write_text("")
        ts = base.isoformat().replace("+00:00", "Z")
        track_path.write_text(
            f'{{"timestamp": "{ts}", "person_found": false, '
            f'"latitude": {HAIFA.lat}, "longitude": {HAIFA.lon}}}\n'
        )

        # Delay of 15s with step_sec=10s: tick 1 (elapsed 10s) is still grounded,
        # tick 2 (elapsed 20s) launches the drone.
        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=empty_feed), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[track_path]), \
             patch.object(settings, "drone_track_launch_delay_sec", 15.0), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            first = await store.tick(state.mission_id)
            assert first.drone_track is None

            second = await store.tick(state.mission_id)
            assert second.drone_track is not None
            assert second.drone_track.position is not None

    asyncio.run(run())


def _write_track(path: Path, points: list[tuple[float, bool]], base: datetime) -> None:
    """points: list of (offset_sec, person_found)."""
    lines = []
    for off, found in points:
        ts = (base + timedelta(seconds=off)).isoformat().replace("+00:00", "Z")
        extra = ', "confidence": 0.8, "confidence_percent": 80.0' if found else ""
        lines.append(
            f'{{"timestamp": "{ts}", "person_found": {str(found).lower()}, '
            f'"latitude": {HAIFA.lat}, "longitude": {HAIFA.lon}{extra}}}'
        )
    path.write_text("\n".join(lines) + "\n")


def test_two_sorties_play_one_after_another(tmp_path: Path):
    """Drone 1 (no-one) sweeps first; drone 2 (finds the person) launches after."""

    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        empty_feed = tmp_path / "person_detection_output.jsonl"
        empty_feed.write_text("")

        # Sortie 1: 2 not-found points (offsets 0, 5 -> 5 s flight).
        f1 = tmp_path / "drone1_not_found.jsonl"
        _write_track(f1, [(0, False), (5, False)], base)
        # Sortie 2: a sweep point then a detection (offsets 0, 4 -> 4 s flight).
        f2 = tmp_path / "drone2_found.jsonl"
        _write_track(f2, [(0, False), (4, True)], base)

        # delay 0, gap 10, step 10s -> sortie 2 launches at 0+5+10 = 15 s.
        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=empty_feed), \
             patch("app.services.mission_store.get_drone_sortie_paths", return_value=[f1, f2]), \
             patch.object(settings, "drone_track_launch_delay_sec", 0.0), \
             patch.object(settings, "drone_sortie_gap_sec", 10.0), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(settings.grid_size)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )

            # Tick 1 (mission elapsed [0, 10)): only drone 1 has flown.
            first = await store.tick(state.mission_id)
            assert first.drone_track is not None
            assert len(first.drone_track.drones) == 1
            assert first.drone_track.drones[0].asset_id == "drone-1"
            assert first.drone_track.drones[0].found is False
            assert len(first.detection_events) == 0

            # Tick 2 (mission elapsed [10, 20)): drone 2 launches and finds the person.
            second = await store.tick(state.mission_id)
            assert second.drone_track is not None
            ids = [d.asset_id for d in second.drone_track.drones]
            assert ids == ["drone-1", "drone-2"]
            found_item = second.drone_track.drones[1]
            assert found_item.found is True
            assert len(second.detection_events) >= 1
            event = second.detection_events[-1]
            assert event.person_found is True
            assert event.asset_id == "drone-2"

    asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__])
