# ADAR (CSHACK-2026-LKP)

**Predictive Search & Rescue End-to-End Platform for Missing Persons**

> A stale last known point becomes **live search intelligence**.
>
> One pin and a timestamp become a live probability map — then drones sweep the hot zones. Each pass feeds back: searched ground cools, a sighting drops a pin, the search converges.

---

## Why Every Minute Counts

The search area grows with the square of elapsed time. Survival falls sharply with every hour lost — cutting search time is the difference between a rescue and a recovery. ADAR spends the first minutes narrowing where to look.

| Stat | Source |
|---|---|
| **4,500** SAR incidents a year | U.S. National Parks alone |
| **50–100K** personnel-hours a year | spent searching |
| **up to 16%** of drowning victims | never recovered (in some studies) |

---

## How It Works

From a single pin to a confirmed sighting:

| # | Stage | What happens |
|---|---|---|
| 01 | **Last known point** | Operator drops the LKP radius + time. Seeds an impulse at the grid center. |
| 02 | **Environment model** | Features of land and sea combine with subject information to produce the probability map. |
| 03 | **Probability heatmap** | Cost-surface diffusion + layer physics, refreshed each tick. |
| 04 | **Mission routes** | Rank unscanned cells by probability; route drones / teams over the hot sectors. |
| 05 | **Live detection** | Automated drone sweep with CV (or any applied mechanism) feeds back live evidence. |
| 06 | **Found** | Detections re-weight the map (Bayes); the loop sharpens until the subject is located. |

---

## The Models — Modular Environment Framework

One layer = one physics. Combine freely per mission.

| Layer | Module | Physics |
|---|---|---|
| **Probability field & diffusion** | `grid_engine` | Tick keeps mass on-cell, leaks a baseline share to its 8 neighbours, adds per-layer terms |
| **Topography** | `topography` | Tobler's hiking function turns slope into walking speed; Dijkstra gives travel time from the LKP, projected onto the reachable land |
| **Roads** | `roads` | L2 diffusion over a friction map — roads fast, off-road / water slow. Probability forms road "fingers" with a trail-magnetism bonus |
| **Sea drift** | `sea_drift` | Mass advects along a drift vector (current + leeway); diffusion grows the widening downstream search cone |
| **Weather** | `weather` | Wind shifts probability downwind, scaled by wind speed and the simulated time step |
| **Subject mobility** | `personality` | A mobility scalar from age, fitness and injury widens or contracts the field around the LKP |

### The loop — re-weight & search again

**Negative-search feedback (Bayesian update).** After a sweep, a sector's probability drops by its Probability of Detection (POD) and the grid renormalizes — cleared ground cools, freed mass flows elsewhere. A hit collapses the field onto the sighting. Every sweep sharpens the map.

**Automated drone detection.** Autonomous drones sweep the highest-probability sectors and locate people by whatever sensing fits the scene — RGB, thermal, motion, or RF. Each geotagged hit feeds back to the grid as evidence. The demo runs **Ultralytics YOLO26**, but the detector is pluggable — not the core of the system.

---

## Three Subsystems

| Subsystem | Stack | Default port | Responsibility |
|---|---|---|---|
| [`backend/`](backend/) | FastAPI · NumPy · SciPy · GeoPandas · Pydantic v2 | **8000** | Particle filter, layer pipeline, negative search, path optimizer, REST + WebSocket API |
| [`frontend/`](frontend/) | React 18 · Vite · TypeScript · Mapbox GL JS · Zustand | **5173** | Map UI, heatmap renderer, mission/layer controls, fleet display |
| [`figure_recognition/`](figure_recognition/) | Python 3.11 (conda `cshack`) · Ultralytics YOLO26 · OpenCV · Flask | **8000** (use 8001 if backend also running) | Drone video → person detection → JSONL + MJPEG dashboard or annotated MP4 |

`topo_layout/` is an early prototype (Tobler hiking isochrones); its heuristic is now integrated into the backend's topography layer. Don't run it for the demo.

---

## System Architecture

```mermaid
flowchart TB
    subgraph drone [figure_recognition]
        Video[Video source RTSP webcam or file]
        YOLO[YOLO26 detector]
        FlaskUI[Flask MJPEG plus JSONL log]
        Video --> YOLO --> FlaskUI
    end

    subgraph backend_svc [backend FastAPI]
        PF[ParticleFilter]
        Layers[Layer pipeline topo roads weather personality sea drift]
        NS[NegativeSearch updater]
        Path[Multi drone path optimizer]
        WS[WebSocket hub]
        Layers --> PF --> NS --> WS
        PF --> Path --> WS
    end

    subgraph frontend_ui [frontend React]
        Map[Mapbox map]
        Heat[Canvas heatmap]
        Controls[Mission and layer controls]
        Fleet[Drone fleet display]
        Map --> Heat
        WS --> Heat
        WS --> Controls
        WS --> Fleet
    end

    EnvData[Elevation OSM roads marine current] --> Layers
    frontend_ui -->|REST plus WS mission| backend_svc
    drone -->|detections JSONL or telemetry| Operator
```

### Data flow

| Step | Source → Destination | Payload |
|---|---|---|
| 1 | Operator → Backend | `POST /missions` — LKP, layer flags, personality, mode (live or offline), drone sortie launch delays |
| 2 | Env APIs → Backend | Open-Elevation tiles, OSM road graph, marine current data |
| 3 | Backend → Frontend | `engine_tick`, `heatmap_full`, `heatmap_delta`, `drone_track`, `detection_flash` (WebSocket) |
| 4 | Operator → Backend | `POST /negative-search` — cleared polygon + POD |
| 5 | Operator → Backend | `POST /missions/{id}/drone-route` — request optimal multi-drone search plan |
| 6 | Drone video → `figure_recognition` | YOLO detections, MJPEG stream, `detections.jsonl` |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | Backend (`backend/.venv`) |
| Python + conda | 3.11 | `figure_recognition` (Ultralytics + PyTorch) |
| Node.js | 20+ | Frontend build |
| Mapbox token | — | Frontend map (set `VITE_MAPBOX_TOKEN` in `frontend/.env`) |

---

## Quick Start (three terminals)

> Backend and `detect_live.py` both default to port 8000. Run backend on 8000 and change `PORT = 8001` at the top of `figure_recognition/detect_live.py` if running both locally.

### Terminal 1 — Backend (port 8000)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # tweak particle count, grid size, etc. (optional)
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

API docs: <http://localhost:8000/docs>

### Terminal 2 — Frontend (port 5173)

```bash
cd frontend
npm install
cp .env.example .env       # put your VITE_MAPBOX_TOKEN here
npm run dev
```

Open <http://localhost:5173>, click the map to place LKP, then **Run Heatmap**.

### Terminal 3 — Drone detection (`figure_recognition`)

```bash
conda env create -f environment.macos.yaml   # Mac Apple Silicon (MPS)
# or: conda env create -f environment.yaml     # Linux + CUDA
conda activate cshack
python figure_recognition/detect_live.py
```

Open <http://localhost:8000/> (or whichever PORT you set). `F11` for fullscreen.

**Source options** — edit the `CONFIG` block at the top of `detect_live.py`:

- `SOURCE = 0` — Mac webcam
- `SOURCE = HERE / "samples" / "drone_test.mp4"` — bundled sample clip
- `SOURCE = "rtsp://localhost:8554/live/drone"` — RTSP from MediaMTX (see below)

YOLO weights auto-download on first run into `figure_recognition/models/`.

### Optional — Live drone via RTSP

The `server/` directory bundles **MediaMTX** for ingesting a DJI Mini 4 Pro RTMP push and restreaming it as RTSP:

```bash
cd server && ./mediamtx       # opens :1935 (RTMP) and :8554 (RTSP)
```

In DJI Fly: **Profile → Live Stream → Custom RTMP → `rtmp://<your-mac-ip>:1935/live/drone`**. Then point `detect_live.py` at `rtsp://localhost:8554/live/drone`.

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Default |
|---|---|---|
| `PARTICLE_COUNT` | Monte Carlo particles per mission | `5000` |
| `GRID_SIZE` | Heatmap grid rows/cols | `128` |
| `GRID_RESOLUTION_M` | Cell size, meters | `25` |
| `FILTER_HZ` | Particle filter tick frequency | `1` |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:5173,http://127.0.0.1:5173` |
| `ROADS_DATA_SOURCE` | Where to fetch OSM roads (`auto`, `local`, `osm`) | `auto` |
| `TERRAIN_INSPECT_RESOLUTION_M` | Resolution for terrain inspection endpoint | `25` |

### Frontend (`frontend/.env`)

| Variable | Description | Example |
|---|---|---|
| `VITE_MAPBOX_TOKEN` | Mapbox public token | `pk.eyJ1...` |
| `VITE_BACKEND_URL` | REST base URL | `http://localhost:8000` |
| `VITE_BACKEND_WS_URL` | Mission WebSocket base | `ws://localhost:8000/ws/mission` |

`.env` files are gitignored. Use the bundled `.env.example` as your template. Never commit a real Mapbox token.

---

## API Surface

| Method | Path | Description |
|---|---|---|
| `POST` | `/missions` | Create mission (LKP, layers, mode, sortie delays, personality) |
| `GET` | `/missions/{id}` | Mission status |
| `DELETE` | `/missions/{id}` | Tear down mission |
| `POST` | `/missions/{id}/pause` · `resume` | Simulation control |
| `PATCH` | `/missions/{id}/pace` | Change simulated-time pace |
| `GET` | `/missions/{id}/heatmap` | Current probability grid |
| `GET` | `/missions/{id}/node-fields` | Per-cell terrain data |
| `POST` | `/missions/{id}/drone-route` | Run multi-drone path optimizer |
| `POST` | `/negative-search` | Apply cleared-area Bayesian update |
| `GET` | `/terrain/inspect` | Inspect terrain at a coordinate |
| `WS` | `/ws/mission/{id}` | Live engine ticks, heatmap deltas, drone tracks, detection flashes |

---

## Repo Layout

```
.
├── README.md
├── AGENT.md                       Global contributor / agent rules
├── environment.yaml               conda env (Linux + CUDA)
├── environment.macos.yaml         conda env (Mac Apple Silicon, MPS)
├── backend/                       FastAPI SAR core
│   ├── app/
│   │   ├── api/                   REST + WebSocket routes
│   │   ├── core/                  Settings, logging
│   │   ├── layers/                topography, roads, weather, personality, sea drift
│   │   ├── models/                Pydantic schemas
│   │   └── services/              particle_filter, negative_search, path_optimizer, drone_detection, marine_current, ...
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/                      React + Mapbox command center
│   ├── src/components/            map, panels, ui
│   ├── src/stores/                Zustand mission store
│   ├── src/hooks/                 useWebSocket, useHeatmap, useMission
│   ├── package.json
│   └── .env.example
├── figure_recognition/            Drone CV
│   ├── detect_live.py             RTSP/webcam/file → Flask MJPEG dashboard + JSONL
│   ├── detect_offline.py          Video file → JSONL + annotated MP4 (CLI)
│   ├── prepare_drone_data.py      Merge CSV telemetry with detection JSON
│   ├── merge_gps.py               Join detection timestamps with GPS
│   ├── samples/                   Test clips (bundled)
│   ├── models/                    YOLO weights (gitignored)
│   └── results/                   JSONL output (gitignored)
├── server/                        MediaMTX RTMP/RTSP relay (binary gitignored)
├── scripts/                       slurm setup, deploy helpers
└── topo_layout/                   prototype only (not run for demo)
```

---

## Documentation Index

| File | Purpose |
|---|---|
| [AGENT.md](AGENT.md) | Project scope, cross-cutting standards, Git conventions |
| [backend/README.md](backend/README.md) | Backend setup, module map, particle filter notes |
| [backend/AGENT.md](backend/AGENT.md) | Backend-specific contracts |
| [backend/LAYERS.md](backend/LAYERS.md) | Layer pipeline reference |
| [backend/INTERACTIVE_LAYERS.md](backend/INTERACTIVE_LAYERS.md) | Layer interaction semantics |
| [frontend/README.md](frontend/README.md) | Frontend setup, UI structure |
| [frontend/AGENT.md](frontend/AGENT.md) | Canvas heatmap and WebSocket rules |
| [figure_recognition/README.md](figure_recognition/README.md) | Detection module layout |

---

## License

Hackathon project — internal use during CSHACK 2026. License TBD post-event.

---

## In Memory

**Adar — Narrow the search. Bring them back.**

ADAR is named in memory of **Lieutenant Adar Ben Simon**, who fell in battle while defending Kibbutz Zikim and her platoon on October 7th.

— Eden Avidan · Roy Carmel · Shir Belson · Naor Shoyhat · Linoy Geva
