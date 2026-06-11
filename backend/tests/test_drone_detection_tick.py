"""Tests for drone last-seen updates during mission ticks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.models.mission import LatLon, MissionMode
from app.services.mission_store import MissionStore
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


def test_tick_updates_drone_last_seen(tmp_path: Path):
    async def run() -> None:
        store = MissionStore()
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_ts = two_hours_ago + timedelta(seconds=30)
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


if __name__ == "__main__":
    pytest.main([__file__])
