"""Tests for Open-Meteo marine current fetch and polar→Cartesian conversion."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import settings
from app.services.marine_current import (
    clear_marine_current_cache,
    fallback_marine_current,
    fetch_marine_current,
    polar_current_to_cartesian_mps,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_marine_current_cache()
    yield
    clear_marine_current_cache()


def _mock_client(*, json_payload: dict | None = None, exc: Exception | None = None):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    if json_payload is not None:
        mock_response.json.return_value = json_payload

    mock_client = AsyncMock()
    if exc is not None:
        mock_client.get = AsyncMock(side_effect=exc)
    else:
        mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def test_polar_to_cartesian_east():
    u, v = polar_current_to_cartesian_mps(3.6, 90.0, speed_unit="kmh")
    assert u == pytest.approx(1.0)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_polar_to_cartesian_north():
    u, v = polar_current_to_cartesian_mps(3.6, 0.0, speed_unit="kmh")
    assert u == pytest.approx(0.0, abs=1e-9)
    assert v == pytest.approx(1.0)


def test_fallback_matches_config():
    fc = fallback_marine_current()
    assert fc.u_east_mps == settings.marine_current_fallback_u_mps
    assert fc.v_north_mps == settings.marine_current_fallback_v_mps
    assert fc.source == "fallback"


def test_fetch_marine_current_success():
    payload = {
        "current": {
            "ocean_current_velocity": 7.2,
            "ocean_current_direction": 90.0,
        }
    }
    with patch(
        "app.services.marine_current.httpx.AsyncClient",
        return_value=_mock_client(json_payload=payload),
    ):
        result = asyncio.run(fetch_marine_current(32.8, 34.5))
    assert result.source == "open_meteo"
    assert result.u_east_mps == pytest.approx(2.0)
    assert result.v_north_mps == pytest.approx(0.0, abs=1e-9)


def test_fetch_marine_current_caches():
    payload = {
        "current": {
            "ocean_current_velocity": 3.6,
            "ocean_current_direction": 0.0,
        }
    }
    with patch(
        "app.services.marine_current.httpx.AsyncClient",
        return_value=_mock_client(json_payload=payload),
    ) as client_cls:
        first = asyncio.run(fetch_marine_current(32.8, 34.5))
        second = asyncio.run(fetch_marine_current(32.8, 34.5))
    assert first is second
    assert client_cls.call_count == 1


def test_fetch_marine_current_fallback_on_error():
    with patch(
        "app.services.marine_current.httpx.AsyncClient",
        return_value=_mock_client(exc=httpx.ConnectError("offline")),
    ):
        result = asyncio.run(fetch_marine_current(32.8, 34.5))
    assert result.source == "fallback"
    assert result.u_east_mps == settings.marine_current_fallback_u_mps
    assert result.v_north_mps == settings.marine_current_fallback_v_mps


def test_fetch_marine_current_fallback_on_missing_fields():
    with patch(
        "app.services.marine_current.httpx.AsyncClient",
        return_value=_mock_client(json_payload={"current": {}}),
    ):
        result = asyncio.run(fetch_marine_current(32.8, 34.5))
    assert result.source == "fallback"
