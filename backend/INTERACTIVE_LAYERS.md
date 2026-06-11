# Interactive heatmap — per-layer field rules

The grid starts as an **impulse at the LKP** (center cell = 1, all others = 0). Each enabled layer then **independently transforms** the full matrix using its static per-cell fields (`NodeFields`). Values are **not** re-normalized to sum to 1 — the frontend scales by peak for display.

## Pipeline

```
t=0:  P[lkp] = 1,  P[elsewhere] = 0
      ↓
for each active layer (registry order):
      P ← layer.apply_field(P, node_fields, weight)
      ↓
optional land/sea mask (zero land or zero water)
      ↓
return P   (raw cell values, any total mass)
```

Layers run in **registry order** in `app/engine/layers/registry.py`. Each layer receives the output of the previous layer.

## Layer catalog

### 1. Topography (`topography`) — **implemented**

**Fields used:** `reachability_score`, `is_land`, `slope`

**Rule:** Spread mass from the LKP anchor along the Tobler/Dijkstra walking reach field.

```
anchor = P[lkp_row, lkp_col]
target[r,c] = anchor × reachability_score[r,c]
target[water] = 0
target[steep] ×= topo_steep_weight     (slope ≥ topo_steep_threshold_deg)
P ← (1 − w) × P + w × target
P[water] = 0
```

**Intuition:** If `P` is the initial impulse, the result is the **reachable walking envelope** from the pin — high near LKP, fading with travel time, zero on water and reduced on steep terrain.

**Reachability refresh:** `reachability_score` is recomputed each tick as simulated time grows (`mission_store._update_reachability`). Horizon advances by one `step_sec` per tick (**default 10 s** simulated time per tick at pace 1×; see `BASE_STEP_SEC` in `app/models/mission.py`).

**Walking speed:** Tobler function on the elevation grid; flat-ground speed **`tobler_flat_speed_kmh` (default 4.7 km/h)** in `config.py`.

---

### 2. Roads (`roads`) — **implemented**

**Fields used:** `is_road`, `is_land`, `slope`, `road_proximity`

**Rule:** Continuous **cost-surface diffusion** — blends L2 (Euclidean) neighbor distance with terrain friction. No hard walls; roads form fast "fingers" with soft forest bleed.

**Traversal cost map** (per cell):

| Terrain | Cost (default) |
|---------|----------------|
| road | `cost_road` (1.0) |
| off-road / light brush | `cost_offroad` (4.0) |
| steep slope (≥ threshold) | `cost_steep_slope` (8.0) |
| water | `cost_water` (20.0) |

**Transition weight** A → neighbor B (L2 + topology blend):

```
L2 = 1.0 (cardinal) or √2 (diagonal)
w = scale × (road_l2_weight/L2 + road_topology_weight/(L2 × terrain_cost[B]))
if off-road A and road B:  w ×= (1 + trail_magnetism_bonus)
```

Default blend: **28% pure L2** (Euclidean intent) + **72% cost-weighted** (terrain/road friction).

Each tick, N diffusion steps (default **6**, ramped over `road_warmup_ticks`; **0 on tick 0**):

```
P' ← cost_surface_diffusion(P, terrain_cost, is_road)
P'[lkp] ← max(P'[lkp], anchor × boost[lkp])
P ← (1 − w_roads) × P + w_roads × P' × (1 + road_kde_bonus × road_proximity)
```

`w_roads` defaults to **0.68** — topography envelope remains, roads add clear trail channeling.

**Intuition:** L2 keeps natural radial uncertainty; topology (cost map) channels mass toward trails and away from steep/water cells without hard walls.

---

### 3. Weather (`weather`) — *planned*

**Fields used:** `wind_u`, `wind_v` (per-cell; filled from env at create)

**Planned rule:** Shift mass one step downwind (discrete advection).

```
Δrow, Δcol from wind vector and dt
P' ← advect(P, wind, dt_sec)
P ← (1 − w) × P + w × P'
```

---

### 4. Personality (`personality`) — **implemented**

**Fields used:** none per-cell — global subject profile set at mission create (`age`, `fitness`, `injured`).

**Mobility heuristic** (combined multiplier `M`):

```
age_factor     = clamp(1.20 − (age − 10) / 100,  0.35 … 1.20)   # ↓ with age
fitness_factor = 0.85 + 0.10 × (fitness − 1)                     # 1→0.85, 5→1.25 (>1 when fit)
injured_factor = 0.45  if injured else 1.0                       # <1 when injured

M = age_factor × fitness_factor × injured_factor
```

**Rule:** Distance-weighted scale from the LKP (anchor cell unchanged):

```
dist_norm[r,c] = min(1, hypot(r−lkp, c−lkp) / (size/2))
scale[r,c]     = M ** (1 + dist_norm)        # scale[lkp] = 1
target         = P × scale
P ← (1 − w) × P + w × target
```

**Intuition:** Young, fit, uninjured subjects (`M > 1`) push probability farther from the pin; older, unfit, or injured subjects (`M < 1`) keep mass closer to the LKP. Set profile in the UI before **Run Heatmap** — locked for the mission run.

---

### 5. Sea drift (`sea_drift`)

**Fields used:** `current_u`, `current_v`, `is_land`

**Data source:** Open-Meteo Marine API — fetched once at mission create when `sea_drift` is enabled (LKP on water). Cached for the run; 3 s timeout with fallback vector from config.

**Rule:** Advect probability on water cells along the live `[u_east, v_north]` current; land cells are zeroed by the engine land mask.

```
current ← fetch_marine_current(LKP) once
P' ← advect(P, current_u/v, dt) on water cells (aligned neighbors get higher weight)
P[land] = 0
P ← (1 − w) × P + w × P'
```

Auto-enabled when LKP is on water (`mission_store.create`).

---

## Configuration knobs

| Layer | Config keys |
|-------|-------------|
| Topography | `tobler_flat_speed_kmh`, `topo_steep_threshold_deg`, `topo_steep_weight`, reachability horizon via `step_sec` × tick count |
| Roads | `road_l2_weight`, `road_topology_weight`, `cost_road`, `cost_offroad`, `trail_magnetism_bonus`, `diffusion_steps`, `road_kde_bonus`, `momentum_reference_dt_sec` (diffusion dt scale) |
| Weather | `momentum_reference_dt_sec`, wind from env |
| Personality | `age`, `fitness` (1–5), `injured` — see heuristic above |
| Sea drift | `marine_api_timeout_sec`, `marine_current_fallback_u/v_mps`, `marine_drift_advection_strength`, `marine_drift_steps` |
| **Engine (all ticks)** | `heatmap_history_decay` (default **0.86** — lower = less cumulative spread), `BASE_STEP_SEC` (**10 s**), `momentum_reference_dt_sec` (**60 s** reference for road/weather step scaling) |

### `momentum_reference_dt_sec` (default 60)

Not the simulation clock. It scales **how many road/weather diffusion steps** run per tick:

`diffusion_steps_effective ≈ diffusion_steps × layer_weight × (dt_sec / momentum_reference_dt_sec)`

With `BASE_STEP_SEC = 10` and reference 60, each tick runs ~⅙ of the “design” diffusion vs a 60 s tick. **`dt_sec`** comes from pace (`BASE_STEP_SEC × pace`).

### `heatmap_history_decay` (default 0.86)

Each tick after layers: `P ← decay × P_prior + (1 − decay) × P_current`. Lower decay = replace more of the old cloud each tick = **tighter** heatmap over long runs.

## Adding a layer

1. Subclass `BaseProbabilityLayer` in `app/engine/layers/<name>.py`
2. Implement `apply_field(ctx, weight) -> np.ndarray`
3. Register in `registry.py`
4. Add toggle to `LayerFlags` in `app/models/layers.py`
5. Document the rule in this file

Legacy `transition_weights()` remains on the base class for reference but is **not** used by the interactive pipeline.

See also: [LAYERS.md](LAYERS.md) for grid architecture and node field definitions.
