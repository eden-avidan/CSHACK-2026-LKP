"""Spatial mock wind field for the interactive grid engine."""

from __future__ import annotations

import numpy as np

from app.core.config import settings


def fill_spatial_mock_wind(wind_u: np.ndarray, wind_v: np.ndarray) -> None:
    """
    Per-cell wind (m/s, east / north) twisting from W at grid bottom-left
    to N at top-right. Row 0 is north; bottom-left = south-west corner.
    """
    size = wind_u.shape[0]
    if size <= 1:
        wind_u.fill(settings.mock_wind_u_w_mps)
        wind_v.fill(settings.mock_wind_v_w_mps)
        return

    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    row_south = rows / (size - 1)
    col_west = 1.0 - cols / (size - 1)
    t = np.clip(1.0 - (row_south + col_west) / 2.0, 0.0, 1.0)

    u0 = settings.mock_wind_u_w_mps
    v0 = settings.mock_wind_v_w_mps
    u1 = settings.mock_wind_u_n_mps
    v1 = settings.mock_wind_v_n_mps

    wind_u[:] = u0 + t * (u1 - u0)
    wind_v[:] = v0 + t * (v1 - v0)
