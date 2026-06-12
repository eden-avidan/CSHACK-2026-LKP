"""Tests for spatial mock wind and cone advection."""

from __future__ import annotations

import numpy as np

from app.engine.flow_advection import flow_cone_advect_step
from app.engine.grid_matrix import NodeFields
from app.engine.wind_field import fill_spatial_mock_wind


def test_mock_wind_twists_from_w_to_n():
    fields = NodeFields.zeros(32)
    fill_spatial_mock_wind(fields.wind_u, fields.wind_v)

    bl = fields.wind_u[-1, 0], fields.wind_v[-1, 0]
    tr = fields.wind_u[0, -1], fields.wind_v[0, -1]

    assert bl[0] < 0 and abs(bl[1]) < 0.01
    assert abs(tr[0]) < 0.01 and tr[1] > 0
    assert tr[1] > bl[1]


def test_cone_advection_moves_mass_downstream_not_radial():
    size = 9
    p = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    p[center, center] = 1.0

    u = np.zeros((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    u[:, :] = 2.0
    v[:, :] = 0.0
    active = np.ones((size, size), dtype=bool)

    out = flow_cone_advect_step(
        p,
        u,
        v,
        active,
        dt_sec=60.0,
        strength=1.0,
        resolution_m=10.0,
    )

    east_mass = out[center, center + 1 :].sum()
    west_mass = out[center, :center].sum()
    assert east_mass > west_mass * 3
