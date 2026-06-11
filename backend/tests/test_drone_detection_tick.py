"""Tests for drone last-seen updates during mission ticks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest

from app.models.mission import LatLon, MissionMode
from app.models.detection import DetectionEventMessage
from app.api.ws.mission import broadcast_tick_result
from app.services.mission_store import MissionStore
from app.services.mission_store import TickResult
from app.services.drone_detection import load_detection_records

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
        "{\"ts\": \"2026-06-11T16:00:01.000Z\", \"person\": false}\n"
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


def test_tick_updates_drone_last_seen(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_ts = two_hours_ago + timedelta(seconds=5)
        path = tmp_path / "person_detection_output.jsonl"
        path.write_text(
            f"{{\"ts\": \"{sample_ts.isoformat().replace('+00:00','Z')}\", "
            "\"person\": true, \"latitude\": 32.7940, \"longitude\": 34.9896, \"altitude\": 0.0}\n"
        )

        with patch("app.services.mission_store.get_default_detection_jsonl_path", return_value=path), \
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(128)
            state = await store.create(
                HAIFA, mode=MissionMode.LIVE, lkp_timestamp=two_hours_ago
            )
            await store.tick(state.mission_id)

            seen = state.grid_matrix.node_fields.drone_last_seen
            assert seen.dtype == bool
            assert seen.sum() >= 1

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
             patch("app.services.mission_store.build_terrain_context", new_callable=AsyncMock) as mock_tc:
            mock_tc.return_value = _terrain(128)
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


if __name__ == "__main__":
    pytest.main([__file__])
