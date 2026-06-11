"""Open-Meteo Marine API — fetch and cache surface current at the LKP."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"

_cached_current: MarineCurrent | None = None
_cache_key: tuple[float, float] | None = None


@dataclass(frozen=True)
class MarineCurrent:
    """Surface current in grid coordinates (east, north) m/s."""

    u_east_mps: float
    v_north_mps: float
    speed_mps: float
    direction_deg: float
    source: str  # "open_meteo" | "fallback"


def polar_current_to_cartesian_mps(
    speed: float,
    direction_deg: float,
    *,
    speed_unit: str = "kmh",
) -> tuple[float, float]:
    """
    Convert polar current to Cartesian [u_east, v_north] in m/s.

    ``direction_deg`` is the compass bearing the current flows *toward*
    (0° = north, 90° = east), matching Open-Meteo ``ocean_current_direction``.
    """
    if speed_unit == "kmh":
        speed_mps = speed / 3.6
    elif speed_unit == "ms":
        speed_mps = speed
    else:
        speed_mps = speed / 3.6

    rad = math.radians(direction_deg)
    u_east = speed_mps * math.sin(rad)
    v_north = speed_mps * math.cos(rad)
    return u_east, v_north


def fallback_marine_current() -> MarineCurrent:
    u = settings.marine_current_fallback_u_mps
    v = settings.marine_current_fallback_v_mps
    speed = math.hypot(u, v)
    direction = math.degrees(math.atan2(u, v)) if speed > 1e-9 else 0.0
    if direction < 0:
        direction += 360.0
    return MarineCurrent(
        u_east_mps=u,
        v_north_mps=v,
        speed_mps=speed,
        direction_deg=direction,
        source="fallback",
    )


def clear_marine_current_cache() -> None:
    """Test helper — reset cached vector between missions."""
    global _cached_current, _cache_key
    _cached_current = None
    _cache_key = None


async def fetch_marine_current(lat: float, lon: float) -> MarineCurrent:
    """
    Fetch current once per LKP (cached for the simulation run).

    On network/parse failure, returns ``marine_current_fallback_*`` from config.
    """
    global _cached_current, _cache_key

    key = (round(lat, 4), round(lon, 4))
    if _cache_key == key and _cached_current is not None:
        return _cached_current

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "ocean_current_velocity,ocean_current_direction",
        }
        async with httpx.AsyncClient(
            timeout=settings.marine_api_timeout_sec,
        ) as client:
            response = await client.get(MARINE_API_URL, params=params)
            response.raise_for_status()
            payload = response.json()

        if payload.get("error"):
            raise ValueError(payload.get("reason", "Open-Meteo marine API error"))

        current = payload.get("current") or {}
        speed = float(current["ocean_current_velocity"])
        direction = float(current["ocean_current_direction"])
        u_east, v_north = polar_current_to_cartesian_mps(speed, direction, speed_unit="kmh")
        result = MarineCurrent(
            u_east_mps=u_east,
            v_north_mps=v_north,
            speed_mps=math.hypot(u_east, v_north),
            direction_deg=direction,
            source="open_meteo",
        )
        logger.info(
            "Marine current at (%.4f, %.4f): %.2f km/h @ %.0f° → u=%.3f v=%.3f m/s",
            lat,
            lon,
            speed,
            direction,
            u_east,
            v_north,
        )
    except Exception as exc:  # noqa: BLE001 — hackathon fallback
        logger.warning("Marine API failed (%s); using fallback current", exc)
        result = fallback_marine_current()

    _cached_current = result
    _cache_key = key
    return result
