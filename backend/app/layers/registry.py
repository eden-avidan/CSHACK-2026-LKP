from __future__ import annotations

from app.layers.base import HeatmapLayer
from app.layers.personality import personality_layer
from app.layers.roads import roads_layer
from app.layers.topography import topography_layer
from app.layers.weather import weather_layer
from app.models.layers import LayerFlags

LAYER_REGISTRY: list[HeatmapLayer] = [
    topography_layer,
    personality_layer,
    weather_layer,
    roads_layer,
]

_LAYER_BY_ID: dict[str, HeatmapLayer] = {layer.config.id: layer for layer in LAYER_REGISTRY}

FALLBACK_LAYER_ID = "topography"


def ensure_min_one_layer(flags: LayerFlags) -> LayerFlags:
    if flags.topography or flags.roads or flags.personality or flags.weather:
        return flags
    flags.topography = True
    return flags


def ensure_min_one_dict(layers: dict[str, bool]) -> dict[str, bool]:
    known = {k: v for k, v in layers.items() if k in _LAYER_BY_ID}
    if not any(known.values()):
        known[FALLBACK_LAYER_ID] = True
    return known


def get_layer_weights(flags: LayerFlags) -> dict[str, float]:
    weights: dict[str, float] = {}
    for layer in LAYER_REGISTRY:
        lid = layer.config.id
        enabled = getattr(flags, lid, False)
        weights[lid] = layer.config.default_weight if enabled else 0.0
    return weights


def is_layer_enabled(flags: LayerFlags, layer_id: str) -> bool:
    return bool(getattr(flags, layer_id, False))
