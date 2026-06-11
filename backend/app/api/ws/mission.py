from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.heatmap import HeatmapDeltaMessage, HeatmapFullMessage
from app.services.mission_store import TickResult, mission_store

router = APIRouter()

_tick_loops: dict[UUID, asyncio.Task] = {}


def stop_tick_loop(mission_id: UUID) -> None:
    task = _tick_loops.pop(mission_id, None)
    if task and not task.done():
        task.cancel()


async def broadcast_tick_result(mission_id: UUID, result: TickResult) -> None:
    """Push heatmap + engine state to all WS subscribers."""
    if result.full_refresh:
        state = mission_store.get(mission_id)
        if state:
            full = HeatmapFullMessage(
                mission_id=mission_id,
                timestamp=datetime.now(timezone.utc),
                metadata=state.grid.metadata,
                probabilities=state.grid.probabilities.flatten(order="C").tolist(),
            )
            await mission_store.broadcast(mission_id, full.model_dump(mode="json"))
    elif result.deltas:
        msg = HeatmapDeltaMessage(
            mission_id=mission_id,
            timestamp=datetime.now(timezone.utc),
            cells=result.deltas,
        )
        await mission_store.broadcast(mission_id, msg.model_dump(mode="json"))
    elif result.engine_tick:
        # Fallback: send full grid when delta compression yields nothing
        state = mission_store.get(mission_id)
        if state:
            full = HeatmapFullMessage(
                mission_id=mission_id,
                timestamp=datetime.now(timezone.utc),
                metadata=state.grid.metadata,
                probabilities=state.grid.probabilities.flatten(order="C").tolist(),
            )
            await mission_store.broadcast(mission_id, full.model_dump(mode="json"))

    if result.engine_tick:
        await mission_store.broadcast(
            mission_id, result.engine_tick.model_dump(mode="json")
        )

    for detection in result.detection_events:
        await mission_store.broadcast(
            mission_id, detection.model_dump(mode="json")
        )


def start_tick_loop(mission_id: UUID) -> None:
    if mission_id in _tick_loops and not _tick_loops[mission_id].done():
        return

    async def loop() -> None:
        while True:
            state = mission_store.get(mission_id)
            if not state:
                break
            interval = state.update_interval_sec
            await asyncio.sleep(interval)

            state = mission_store.get(mission_id)
            if not state:
                break
            if not state.simulation_running:
                continue
            result = await mission_store.tick(mission_id)
            await broadcast_tick_result(mission_id, result)

    _tick_loops[mission_id] = asyncio.create_task(loop())


@router.websocket("/ws/mission/{mission_id}")
async def mission_ws(websocket: WebSocket, mission_id: UUID) -> None:
    state = mission_store.get(mission_id)
    if not state:
        await websocket.close(code=4004, reason="Mission not found")
        return

    await websocket.accept()
    queue = mission_store.subscribe(mission_id)

    full = HeatmapFullMessage(
        mission_id=mission_id,
        timestamp=datetime.now(timezone.utc),
        metadata=state.grid.metadata,
        probabilities=state.grid.probabilities.flatten(order="C").tolist(),
    )
    await websocket.send_json(full.model_dump(mode="json"))

    engine_tick = mission_store.build_engine_tick(mission_id)
    if engine_tick:
        await websocket.send_json(engine_tick.model_dump(mode="json"))

    async def sender() -> None:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("event") == "update_layers" and isinstance(data.get("layers"), dict):
                await mission_store.update_layers(mission_id, data["layers"])
                engine_tick = mission_store.build_engine_tick(mission_id)
                result = TickResult(deltas=[], engine_tick=engine_tick, full_refresh=True)
                await broadcast_tick_result(mission_id, result)
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        mission_store.unsubscribe(mission_id, queue)
