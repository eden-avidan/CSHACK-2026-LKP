from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    grid_size: int = 128
    grid_resolution_m: float = 1
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
    road_kde_bonus: float = 0.9
    road_proximity_decay_m: float = 20.0

    # Continuous cost-surface diffusion (RoadMagnetismLayer)
    cost_road: float = 1.0
    cost_offroad: float = 4.0
    cost_steep_slope: float = 8.0
    cost_water: float = 20.0
    cost_floor: float = 0.5
    trail_magnetism_bonus: float = 0.28
    transition_weight_scale: float = 1.0
    road_l2_weight: float = 0.28
    road_topology_weight: float = 0.72
    diffusion_self_weight: float = 0.11
    diffusion_steps: int = 6
    diffusion_steps_max: int = 18
    road_initial_diffusion_steps: int = 0
    road_warmup_ticks: int = 3
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

    # Open-Meteo Marine API (SeaDriftLayer)
    marine_api_timeout_sec: float = 3.0
    marine_current_fallback_u_mps: float = 1.0
    marine_current_fallback_v_mps: float = 0.5
    marine_drift_advection_strength: float = 2.5
    marine_drift_self_weight: float = 0.15
    marine_drift_steps: int = 2          # advection steps added per simulated tick
    marine_drift_max_steps: int = 96     # cap on cumulative drift steps (≈ grid half-width)

    kde_edge_fade_cells: int = 0
    kde_bandwidth_factor: float = 1.1
    kde_radial_fade_end: float = 1.0
    heatmap_history_decay: float = 0.94

    # Drone "clean" coverage: cells the drone overflew while detecting no person.
    # A freshly-cleared cell scores 1.0 and decays by `drone_clean_decay` each tick
    # (the bigger the score, the less likely the person is there). Each tick the
    # probability is suppressed by (1 - drone_clean_suppression_strength * clean)
    # and renormalized, so cleared areas lose mass and slowly recover as the clean
    # score decays and probability re-diffuses.
    drone_clean_decay: float = 0.95
    drone_clean_suppression_strength: float = 0.7
    # Ground coverage footprint (camera swath) of a single drone GPS fix, in
    # meters. Each "no person" fix clears every cell whose centroid falls within
    # this radius, so a sweep suppresses a realistic area instead of one cell.
    # 0 -> legacy single-cell behavior.
    drone_coverage_radius_m: float = 80.0
    # Delay (mission-time seconds) before the first drone "launches" and starts
    # sweeping, so it doesn't move the instant the mission begins.
    drone_track_launch_delay_sec: float = 60.0
    # Sequential drone sorties: comma-separated paths (relative to the repo root)
    # played back one after another, each relative to mission start. The first
    # launches after drone_track_launch_delay_sec; each subsequent one launches
    # drone_sortie_gap_sec after the previous one lands. Default: first the
    # "found nobody" sweep, then the drone that locates the person.
    drone_sortie_files: str = (
        "figure_recognition/results/drone.not_found.JSON,"
        "figure_recognition/results/drone.merged.jsonl"
    )
    drone_sortie_gap_sec: float = 15.0
    # Per-sortie north shift (meters), comma-separated and aligned to
    # drone_sortie_files, to nudge a flight onto the search area without editing
    # the raw telemetry. Positive = north. Missing/blank entries default to 0.
    drone_sortie_north_offsets_m: str = "60,0"
    # Subsample each flight to at most this many points (person-found points are
    # always kept) so per-tick marking and the broadcast path stay lightweight.
    drone_sortie_max_points: int = 300
    topo_reachability_floor_frac: float = 0.12
    grid_base_outflow: float = 0.22

    # Tobler/Dijkstra topography (topo_layout parity)
    tobler_flat_speed_kmh: float = 3.5  # hiking speed on flat ground (km/h)
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

    @property
    def drone_sortie_file_list(self) -> list[str]:
        return [p.strip() for p in self.drone_sortie_files.split(",") if p.strip()]

    @property
    def drone_sortie_north_offset_list(self) -> list[float]:
        offsets: list[float] = []
        for part in self.drone_sortie_north_offsets_m.split(","):
            part = part.strip()
            try:
                offsets.append(float(part) if part else 0.0)
            except ValueError:
                offsets.append(0.0)
        return offsets


settings = Settings()
