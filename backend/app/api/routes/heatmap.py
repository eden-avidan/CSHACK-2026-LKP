from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.models.heatmap import HeatmapResponse, NegativeSearchRequest, NegativeSearchResponse
from app.services.mission_store import mission_store

router = APIRouter(tags=["heatmap"])


@router.get("/heatmap/{mission_id}", response_model=HeatmapResponse)
async def get_heatmap(mission_id: UUID) -> HeatmapResponse:
    state = mission_store.get(mission_id)
    if not state:
        raise HTTPException(status_code=404, detail="Mission not found")
    flat = state.grid.probabilities.flatten(order="C").tolist()
    return HeatmapResponse(
        mission_id=mission_id,
        metadata=state.grid.metadata,
        probabilities=flat,
    )
