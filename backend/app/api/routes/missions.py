from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.models.heatmap import HeatmapResponse
from app.models.mission import (
    CreateMissionRequest,
    CreateMissionResponse,
    MissionMode,
    MissionResponse,
    MissionStatus,
    UpdatePaceRequest,
)
from app.services.mission_store import mission_store
from app.api.ws.mission import start_tick_loop, stop_tick_loop, broadcast_tick_result

router = APIRouter(prefix="/missions", tags=["missions"])


def _mission_response(state) -> MissionResponse:
    return MissionResponse(
        mission_id=state.mission_id,
        status=state.status,
        mode=state.mode,
        lkp=state.lkp,
        lkp_timestamp=state.lkp_timestamp,
        created_at=state.created_at,
        tick_count=state.tick_count,
        pace=state.pace,
        step_sec=state.step_sec,
        update_interval_sec=state.update_interval_sec,
        simulation_running=state.simulation_running,
    )


@router.post("", response_model=CreateMissionResponse)
async def create_mission(body: CreateMissionRequest) -> CreateMissionResponse:
    try:
        state = await mission_store.create(
            body.lkp,
            body.sigma_0_m,
            mode=body.mode,
            lkp_timestamp=body.lkp_timestamp,
            pace=body.pace,
            step_sec=body.step_sec,
            update_interval_sec=body.update_interval_sec,
            layers=body.layers,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if state.mode == MissionMode.LIVE:
        start_tick_loop(state.mission_id)
    return CreateMissionResponse(mission_id=state.mission_id, status=MissionStatus.SEARCHING)


@router.get("/{mission_id}", response_model=MissionResponse)
async def get_mission(mission_id: UUID) -> MissionResponse:
    state = mission_store.get(mission_id)
    if not state:
        raise HTTPException(status_code=404, detail="Mission not found")
    return _mission_response(state)


@router.patch("/{mission_id}/pace", response_model=MissionResponse)
async def update_pace(mission_id: UUID, body: UpdatePaceRequest) -> MissionResponse:
    try:
        state = await mission_store.update_pace(
            mission_id, body.pace, body.step_sec, body.update_interval_sec
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None
    return _mission_response(state)


@router.post("/{mission_id}/pause", response_model=MissionResponse)
async def pause_mission(mission_id: UUID) -> MissionResponse:
    try:
        state = await mission_store.pause(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None
    return _mission_response(state)


@router.post("/{mission_id}/resume", response_model=MissionResponse)
async def resume_mission(mission_id: UUID) -> MissionResponse:
    try:
        state = await mission_store.resume(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None
    if state.mode == MissionMode.LIVE:
        start_tick_loop(mission_id)
    return _mission_response(state)


@router.delete("/{mission_id}")
async def delete_mission(mission_id: UUID) -> dict:
    try:
        stop_tick_loop(mission_id)
        await mission_store.delete(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None
    return {"deleted": True}


@router.post("/{mission_id}/tick")
async def manual_tick(mission_id: UUID) -> dict:
    try:
        result = await mission_store.tick(mission_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None
    await broadcast_tick_result(mission_id, result)
    return {"cells_updated": len(result.deltas)}


@router.get("/{mission_id}/heatmap", response_model=HeatmapResponse)
async def get_mission_heatmap(mission_id: UUID) -> HeatmapResponse:
    state = mission_store.get(mission_id)
    if not state:
        raise HTTPException(status_code=404, detail="Mission not found")
    flat = state.grid.probabilities.flatten(order="C").tolist()
    return HeatmapResponse(
        mission_id=mission_id,
        metadata=state.grid.metadata,
        probabilities=flat,
    )
