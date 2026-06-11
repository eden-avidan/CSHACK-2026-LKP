from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import heatmap, missions, negative_search, terrain
from app.api.ws import mission as mission_ws
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="RescuEdge API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(missions.router)
app.include_router(heatmap.router)
app.include_router(negative_search.router)
app.include_router(terrain.router)
app.include_router(mission_ws.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
