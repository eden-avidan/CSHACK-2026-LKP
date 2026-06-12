from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.engine.grid_matrix import NodeFields
from app.services.env_ingestion import TerrainContext
from app.services.particle_filter import get_mock_env, zero_env
from app.services.particle_types import EnvForcing


from app.engine.wind_field import fill_spatial_mock_wind
from app.services.marine_current import MarineCurrent


def _populate_sea_current(
    fields: NodeFields,
    marine: MarineCurrent | None = None,
) -> None:
    """Fill water cells with live marine current or configured fallback."""
    if marine is not None:
        u = marine.u_east_mps
        v = marine.v_north_mps
    else:
        heading = np.radians(settings.sea_drift_heading_deg)
        speed = settings.sea_drift_speed_mps
        u = speed * np.sin(heading)
        v = speed * np.cos(heading)
    water = ~fields.is_land
    fields.current_u[water] = u
    fields.current_v[water] = v


def build_node_fields(
    terrain: TerrainContext | None,
    size: int,
    *,
    weather_enabled: bool = False,
    marine_current: MarineCurrent | None = None,
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
    if terrain.reachability_score is not None:
        fields.reachability_score = terrain.reachability_score.astype(np.float64, copy=True)

    fill_spatial_mock_wind(fields.wind_u, fields.wind_v)
    _populate_sea_current(fields, marine_current)
    return fields


def copy_node_fields(fields: NodeFields) -> NodeFields:
    """Deep copy of per-cell static inputs (snapshot at mission create)."""
    return NodeFields(
        elevation=fields.elevation.copy(),
        slope=fields.slope.copy(),
        is_land=fields.is_land.copy(),
        is_road=fields.is_road.copy(),
        road_proximity=fields.road_proximity.copy(),
        road_tangent_e=fields.road_tangent_e.copy(),
        road_tangent_n=fields.road_tangent_n.copy(),
        wind_u=fields.wind_u.copy(),
        wind_v=fields.wind_v.copy(),
        current_u=fields.current_u.copy(),
        current_v=fields.current_v.copy(),
        reachability=fields.reachability.copy(),
        reachability_score=fields.reachability_score.copy(),
        latitude=fields.latitude.copy(),
        longitude=fields.longitude.copy(),
        altitude=fields.altitude.copy(),
        searched_clean=fields.searched_clean.copy(),
    )


def env_for_layers(weather_enabled: bool) -> EnvForcing:
    return get_mock_env() if weather_enabled else zero_env()
