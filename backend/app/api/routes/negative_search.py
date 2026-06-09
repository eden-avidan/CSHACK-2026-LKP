from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models.heatmap import HeatmapDeltaMessage, NegativeSearchRequest, NegativeSearchResponse
from app.services.mission_store import mission_store

router = APIRouter(tags=["negative-search"])


@router.post("/negative-search", response_model=NegativeSearchResponse)
async def negative_search(body: NegativeSearchRequest) -> NegativeSearchResponse:
    try:
        cells = await mission_store.negative_search(body.mission_id, body.polygon, body.pod)
    except KeyError:
        raise HTTPException(status_code=404, detail="Mission not found") from None

    msg = HeatmapDeltaMessage(
        mission_id=body.mission_id,
        timestamp=datetime.now(timezone.utc),
        cells=cells,
    )
    await mission_store.broadcast(body.mission_id, msg.model_dump(mode="json"))

    return NegativeSearchResponse(
        mission_id=body.mission_id,
        cells_updated=len(cells),
        cells=cells,
    )
