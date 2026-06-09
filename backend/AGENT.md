# RescuEdge Backend — Agent Instructions

AI agents and developers implementing backend services **must** follow these rules. Mathematical formulas below are authoritative — do not substitute alternate models without updating this document.

Global conventions: [../AGENT.md](../AGENT.md)

---

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `particle_filter.py` | Initialize, predict, resample, KDE rasterization |
| `negative_search.py` | Bayesian probability reduction in searched polygons |
| `path_optimizer.py` | Multi-agent route scoring and GeoJSON output |
| `env_ingestion.py` | Fetch and interpolate wind, current, elevation |
| `geospatial/grid.py` | Cell indexing, bbox, polygon extraction |
| `geospatial/crs.py` | WGS84 ↔ UTM transforms anchored at LKP |

---

## Particle Filter — State and Notation

### State Vector

Each particle $i$ at time $t$ carries:

$$
\mathbf{x}_t^{(i)} = \begin{bmatrix} \text{lat} \\ \text{lon} \\ v_n \\ v_e \end{bmatrix}
$$

- $v_n, v_e$: velocity components in local North-East (m/s), computed in UTM
- Particle weight: $w_t^{(i)}$, normalized such that $\sum_{i=1}^{N} w_t^{(i)} = 1$

### Environmental Inputs

At each timestep $t$, query environmental fields at particle positions:

| Symbol | Description | Units |
|--------|-------------|-------|
| $(u_w, v_w)$ | Wind velocity (north, east) | m/s |
| $(u_c, v_c)$ | Water current velocity | m/s |
| $\mathbf{s}$ | Terrain slope vector (optional drift bias) | unitless direction |

### Initialization

Given LKP $(lat_0, lon_0)$ and uncertainty radius $\sigma_0$ (meters):

1. Project LKP to UTM
2. Sample $N$ positions: $(x_n^{(i)}, x_e^{(i)}) \sim \mathcal{N}(\mu_0, \sigma_0^2 I)$
3. Initialize velocities: $v_n^{(i)}, v_e^{(i)} \sim \mathcal{N}(0, \sigma_v^2)$
4. Set uniform weights: $w^{(i)} = 1/N$

---

## State Transition (Prediction Step)

Propagate each particle in **projected meters (UTM)**, then convert back to WGS84.

### Velocity Update

$$
\begin{aligned}
v_n^{(i)} &\leftarrow \alpha \, v_n^{(i)} + (1 - \alpha)(u_w + u_c) + \sigma_v \, \eta_1 \\
v_e^{(i)} &\leftarrow \alpha \, v_e^{(i)} + (1 - \alpha)(v_w + v_c) + \sigma_v \, \eta_2
\end{aligned}
$$

### Position Update

$$
\begin{aligned}
x_n^{(i)} &\leftarrow x_n^{(i)} + v_n^{(i)} \, \Delta t + \sigma_x \, \eta_3 \\
x_e^{(i)} &\leftarrow x_e^{(i)} + v_e^{(i)} \, \Delta t + \sigma_x \, \eta_4
\end{aligned}
$$

Where:
- $\eta_k \sim \mathcal{N}(0, 1)$ — independent standard normal noise
- $\alpha \in [0, 1]$ — velocity persistence (default: `0.85`)
- $\sigma_v$ — velocity process noise scale (m/s), derived from weather severity
- $\sigma_x$ — position diffusion scale (m), derived from terrain roughness
- $\Delta t$ — filter timestep (seconds), default `1.0`

### Optional Terrain Bias

If slope vector $\mathbf{s} = (s_n, s_e)$ is available:

$$
\begin{aligned}
v_n^{(i)} &\leftarrow v_n^{(i)} + \beta \, s_n \\
v_e^{(i)} &\leftarrow v_e^{(i)} + \beta \, s_e
\end{aligned}
$$

Where $\beta$ is terrain drift coefficient (default: `0.1` m/s).

### Implementation Notes

- Vectorize over all $N$ particles with NumPy — no per-particle Python loops
- Convert UTM → WGS84 after position update
- Clamp particles to mission bounding box; reflect or redistribute weight at boundaries

---

## Resampling

Compute effective sample size:

$$
N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N} \left(w^{(i)}\right)^2}
$$

When $N_{\text{eff}} < N/2$, apply **systematic resampling**:

1. Draw $u \sim \mathcal{U}(0, 1/N)$
2. For $j = 0, \ldots, N-1$, target $t_j = u + j/N$
3. Walk cumulative weight array to select ancestor particles
4. Reset weights to $1/N$

Do not resample on every step — only when degeneracy threshold is breached.

---

## Heatmap Output (KDE Rasterization)

Convert particles to a regular probability grid via kernel density estimation:

$$
P(x, y) = \sum_{i=1}^{N} w^{(i)} \cdot K_h\big((x, y) - (x^{(i)}, y^{(i)})\big)
$$

Where:
- $K_h$ — isotropic Gaussian kernel with bandwidth $h$ equal to grid cell size
- $(x^{(i)}, y^{(i)})$ — particle position in projected coordinates
- Grid origin anchored at LKP UTM coordinates

### Normalization

After KDE, normalize grid cells:

$$
P(g) \leftarrow \frac{P(g)}{\sum_{g'} P(g')} \quad \forall g
$$

Store grid as 2D NumPy array with metadata: `{ origin_x, origin_y, resolution_m, crs_epsg }`.

---

## Bayesian Negative Search Update

When drone $d$ searches polygon $A$ with Probability of Detection $\text{POD}_d$ and reports **no detection**:

### Likelihood

$$
P(\text{not found in } A \mid \text{target in } A) = 1 - \text{POD}_d
$$

### Grid Cell Update

For each grid cell $g$ whose centroid or area fraction lies inside $A$:

$$
P_t(g) \leftarrow P_{t-1}(g) \cdot (1 - \text{POD}_d)
$$

For cells outside $A$: $P_t(g) = P_{t-1}(g)$ (unchanged).

### Renormalization

$$
P_t(g) \leftarrow \frac{P_t(g)}{\sum_{g'} P_{g'}} \quad \forall g
$$

### Positive Detection Override

When `scan_swath.result == "positive"` or a `detection` event arrives:

1. Collapse probability mass into a Gaussian kernel centered at detection coordinates
2. $\sigma_{\text{det}}$ = 25 m (configurable)
3. Renormalize

Alternatively, for `detection` with high confidence ($> 0.9$):

$$
P_t(g) \leftarrow \begin{cases} 1 & \text{if } g = g_{\text{det}} \\ 0 & \text{otherwise} \end{cases}
$$

Use soft kernel for demo smoothness; hard collapse for final lock.

---

## Numerical Stability Rules

| Rule | Implementation |
|------|----------------|
| Log-space updates | When $P(g) < 10^{-6}$, compute in $\log P$ domain |
| Probability floor | Never set $P(g) = 0$; floor at $\epsilon = 10^{-8}$ |
| Renormalization guard | If sum $< \epsilon$, reinitialize uniform over unsearched cells |
| POD bounds | Clamp $\text{POD}_d \in [0.01, 0.99]$ |

### Log-Space Negative Search

$$
\log P_t(g) \leftarrow \log P_{t-1}(g) + \log(1 - \text{POD}_d) \quad \text{for } g \in A
$$

Then log-sum-exp normalize:

$$
\log P_t(g) \leftarrow \log P_t(g) - \log \sum_{g'} \exp(\log P_t(g'))
$$

---

## WebSocket Emission After Update

After any grid mutation, emit sparse delta:

```json
{
  "type": "heatmap_delta",
  "mission_id": "uuid",
  "timestamp": "2026-06-02T12:00:00.000Z",
  "cells": [
    { "row": 42, "col": 87, "probability": 0.0031 },
    { "row": 42, "col": 88, "probability": 0.0028 }
  ]
}
```

Only include cells that changed above a threshold ($|\Delta P| > 10^{-7}$) or all cells in the searched polygon.

---

## Path Optimizer Rules

### Objective

Maximize expected Probability of Detection along each asset's path, subject to endurance constraint $L_{\max}$ (meters).

### Greedy Baseline (Hackathon)

Acceptable for demo — implement before full TSP/VRP solver:

1. Start at asset's current position
2. Score candidate edges $(i \to j)$:

$$
\text{score}(i, j) = \sum_{g \in \text{swath}(i, j)} P(g) \cdot \text{POD}_{\text{asset}}
$$

3. Greedily select highest-scoring edge until $L_{\max}$ exhausted
4. Return GeoJSON `LineString` per asset in WGS84

### Swath Model

For drone at altitude $h$ with camera FOV $\phi$:

$$
\text{swath\_width} \approx 2 \cdot h \cdot \tan(\phi / 2)
$$

Buffer the path line by `swath_width / 2` to produce search polygon.

### Output Schema

```json
{
  "type": "route_update",
  "mission_id": "uuid",
  "asset_id": "uuid",
  "route": {
    "type": "LineString",
    "coordinates": [[lon, lat], ...]
  },
  "expected_pod": 0.73,
  "length_m": 4200
}
```

---

## Testing Requirements

| Test | Assertion |
|------|-----------|
| `test_initialization` | Weights sum to 1.0; all particles within bbox |
| `test_prediction_step` | Particles displace in wind direction (statistical) |
| `test_resampling` | Post-resample weights uniform; particle count preserved |
| `test_negative_search_monotonicity` | $P(g)$ in searched area decreases; total sum = 1 |
| `test_negative_search_redistribution` | Unsearched cell probability increases after renormalize |
| `test_kde_normalization` | Grid sums to 1.0 ± floating point tolerance |

---

## Anti-Patterns

- Do **not** hand-roll haversine in particle propagation loops — use UTM projection
- Do **not** mutate grid in WebSocket handler — call service layer, then broadcast
- Do **not** zero-out searched cells without renormalization
- Do **not** block the event loop — offload heavy KDE to thread pool if > 100 ms

---

## Related Documentation

- [README.md](README.md) — Setup and endpoints
- [../AGENT.md](../AGENT.md) — Global conventions
- [../edge_drone/AGENT.md](../edge_drone/AGENT.md) — Incoming telemetry schema
