from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import settings


@dataclass
class EnvForcing:
    u_w: float = 2.0
    v_w: float = 1.0
    u_c: float = 0.0
    v_c: float = 0.0


@dataclass
class Particles:
    eastings: np.ndarray
    northings: np.ndarray
    v_n: np.ndarray
    v_e: np.ndarray
    weights: np.ndarray

    @property
    def count(self) -> int:
        return len(self.weights)


def momentum_scales(dt: float, sigma_v: float, sigma_x: float) -> tuple[float, float, float, float]:
    dt_eff = max(float(dt), 0.1)
    ref_dt = settings.momentum_reference_dt_sec
    tau = settings.momentum_tau_sec
    alpha_dt = float(np.exp(-dt_eff / tau))
    time_scale = np.sqrt(dt_eff / ref_dt)
    return alpha_dt, sigma_v * time_scale, sigma_x * time_scale, dt_eff


def apply_edge_fade(grid: np.ndarray, fade_cells: int) -> np.ndarray:
    if fade_cells <= 0:
        return grid
    rows, cols = grid.shape
    r_idx = np.arange(rows, dtype=np.float64)[:, None]
    c_idx = np.arange(cols, dtype=np.float64)[None, :]
    dist = np.minimum(
        np.minimum(r_idx, rows - 1 - r_idx),
        np.minimum(c_idx, cols - 1 - c_idx),
    )
    t = np.clip(dist / fade_cells, 0.0, 1.0)
    fade = t * t * (3.0 - 2.0 * t)
    out = grid * fade
    total = out.sum()
    if total > 0:
        out /= total
    return out


def apply_radial_fade(grid: np.ndarray, fade_end: float | None = None) -> np.ndarray:
    """Circular vignette from grid center so the map overlay has no square clip."""
    end = fade_end if fade_end is not None else settings.kde_radial_fade_end
    if end >= 1.0:
        return grid
    rows, cols = grid.shape
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    if cy <= 0 or cx <= 0:
        return grid
    r_idx = np.arange(rows, dtype=np.float64)[:, None]
    c_idx = np.arange(cols, dtype=np.float64)[None, :]
    dist = np.sqrt(((r_idx - cy) / cy) ** 2 + ((c_idx - cx) / cx) ** 2)
    span = max(1.0 - end, 1e-3)
    t = np.clip((dist - end) / span, 0.0, 1.0)
    fade = 1.0 - t * t * (3.0 - 2.0 * t)
    out = grid * fade
    total = out.sum()
    if total > 0:
        out /= total
    return out
