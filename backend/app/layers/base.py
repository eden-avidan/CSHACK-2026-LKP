from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from app.geospatial.grid import ProbabilityGrid
    from app.services.env_ingestion import TerrainContext
    from app.services.particle_types import EnvForcing, Particles


@dataclass
class LayerConfig:
    id: str
    default_enabled: bool
    default_weight: float = 1.0


@dataclass
class LayerContext:
    terrain: Optional["TerrainContext"] = None
    grid: Optional["ProbabilityGrid"] = None
    terrain_grid: Optional["ProbabilityGrid"] = None
    env: Optional["EnvForcing"] = None
    road_proximity: Optional[np.ndarray] = None
    dt: float = 1.0
    dt_eff: float = 1.0
    pos_noise_std: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


@dataclass
class PredictState:
    particles: "Particles"
    v_n: np.ndarray
    v_e: np.ndarray
    de: np.ndarray
    dn: np.ndarray
    eastings: np.ndarray
    northings: np.ndarray
    sigma_v: float
    sigma_x: float


class HeatmapLayer(Protocol):
    config: LayerConfig

    def adjust_sigmas(self, sigma_v: float, sigma_x: float, weight: float) -> tuple[float, float]: ...

    def apply_velocity(
        self, state: PredictState, ctx: LayerContext, weight: float
    ) -> PredictState: ...

    def apply_displacement(
        self, state: PredictState, ctx: LayerContext, weight: float
    ) -> PredictState: ...

    def apply_post_step(
        self, state: PredictState, ctx: LayerContext, weight: float
    ) -> PredictState: ...

    def kde_road_factor(
        self, row: int, col: int, ctx: LayerContext, weight: float
    ) -> float: ...
