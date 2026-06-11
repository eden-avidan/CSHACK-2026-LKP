from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.grid_matrix import NodeFields
from app.services.env_ingestion import TerrainContext
from app.services.particle_filter import get_mock_env, zero_env
from app.services.particle_types import EnvForcing


def build_node_fields(
    terrain: TerrainContext | None,
    size: int,
    *,
    weather_enabled: bool = False,
) -> NodeFields:
    fields = NodeFields.zeros(size)
    if terrain is None:
        return fields

    fields.elevation = terrain.elevation.astype(np.float64, copy=True)
    fields.altitude = terrain.elevation.astype(np.float64, copy=True)
    fields.slope = terrain.slope.astype(np.float64, copy=True)
    fields.is_land = terrain.is_land.astype(bool, copy=True)
    fields.road_proximity = terrain.road_proximity.astype(np.float64, copy=True)
    fields.road_tangent_e = terrain.road_tangent_e.astype(np.float64, copy=True)
    fields.road_tangent_n = terrain.road_tangent_n.astype(np.float64, copy=True)

    snap_threshold = np.exp(
        -settings.road_snap_radius_m / settings.road_proximity_decay_m
    )
    fields.is_road = fields.road_proximity >= snap_threshold

    if terrain.reachability is not None:
        fields.reachability = terrain.reachability.astype(np.float64, copy=True)

    env = get_mock_env() if weather_enabled else zero_env()
    fields.wind_u.fill(env.u_w)
    fields.wind_v.fill(env.v_w)
    return fields


def env_for_layers(weather_enabled: bool) -> EnvForcing:
    return get_mock_env() if weather_enabled else zero_env()
