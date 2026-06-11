from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    grid_size: int = 128
    grid_resolution_m: float = 50.0
    filter_hz: float = 1.0  # live heatmap broadcast rate (Hz); 1.0 = one update per second
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    alpha: float = 0.85
    sigma_v: float = 0.25
    sigma_x: float = 3.0
    sigma_0_m: float = 200.0
    momentum_tau_sec: float = 600.0
    momentum_reference_dt_sec: float = 60.0

    uphill_factor: float = 0.25
    downhill_factor: float = 1.3
    terrain_beta: float = 0.15
    road_kde_bonus: float = 1.2
    road_proximity_decay_m: float = 20.0
    boundary_reflect_damping: float = 0.7
    boundary_soft_margin_frac: float = 0.1

    road_snap_radius_m: float = 80.0
    road_snap_strength: float = 0.85
    road_displacement_pull: float = 0.35
    injured_velocity_factor: float = 0.25
    land_elevation_threshold_m: float = 1.0
    engine_tick_particle_limit: int = 200

    # Sea drift: used automatically when the LKP falls on water. The object is
    # assumed to drift at a constant velocity (surface current + leeway).
    sea_drift_speed_mps: float = 0.4          # constant drift speed (m/s)
    sea_drift_heading_deg: float = 90.0       # compass heading drift moves TOWARD (0=N, 90=E)
    sea_drift_strength: float = 1.2           # directional bias multiplier vs. baseline diffusion

    kde_edge_fade_cells: int = 0
    kde_bandwidth_factor: float = 1.1
    kde_radial_fade_end: float = 1.0
    heatmap_history_decay: float = 0.94
    topo_reachability_floor_frac: float = 0.12
    grid_base_outflow: float = 0.22

    # Tobler/Dijkstra topography (topo_layout parity)
    topo_probability_method: str = "linear"
    topo_steep_threshold_deg: float = 30.0
    topo_cliff_threshold_deg: float = 45.0
    topo_neighborhood_size: int = 3
    topo_ridge_threshold_m: float = 5.0
    topo_valley_threshold_m: float = 5.0
    topo_steep_weight: float = 0.7
    topo_cliff_like_weight: float = 0.2
    topo_valley_weight: float = 1.15
    topo_ridge_weight: float = 0.9

    env_fetch_timeout_sec: float = 8.0
    roads_data_source: str = "auto"  # auto | overpass | osm_map

    # Terrain inspection: meters per cell (smaller = finer detail, same coverage).
    # Coverage stays grid_size * grid_resolution_m; cell count scales up accordingly.
    terrain_inspect_resolution_m: float = 25.0
    # Tobler/Dijkstra horizon used by terrain debug inspect (hours).
    terrain_inspect_reachability_hours: float = 2.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
