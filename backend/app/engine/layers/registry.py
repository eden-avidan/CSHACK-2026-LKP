from __future__ import annotations

from app.engine.layers.base import BaseProbabilityLayer
from app.engine.layers.personality import PersonalityLayer
from app.engine.layers.road_magnetism import RoadMagnetismLayer
from app.engine.layers.sea_drift import SeaDriftLayer
from app.engine.layers.topography import TopographyLayer
from app.engine.layers.weather import WeatherLayer
from app.models.layers import LayerFlags

PROBABILITY_LAYER_REGISTRY: list[BaseProbabilityLayer] = [
    TopographyLayer(),
    PersonalityLayer(),
    WeatherLayer(),
    RoadMagnetismLayer(),
    SeaDriftLayer(),
]

_LAYER_BY_ID: dict[str, BaseProbabilityLayer] = {
    layer.layer_id: layer for layer in PROBABILITY_LAYER_REGISTRY
}

FALLBACK_LAYER_ID = "topography"


def register_layer(layer: BaseProbabilityLayer) -> None:
    """Register a new layer at runtime (or append for team plugins)."""
    if layer.layer_id in _LAYER_BY_ID:
        raise ValueError(f"Layer {layer.layer_id!r} already registered")
    PROBABILITY_LAYER_REGISTRY.append(layer)
    _LAYER_BY_ID[layer.layer_id] = layer


def ensure_min_one_layer(flags: LayerFlags) -> LayerFlags:
    if flags.any_enabled():
        return flags
    flags.topography = True
    return flags


def ensure_min_one_dict(layers: dict[str, bool]) -> dict[str, bool]:
    known = {k: v for k, v in layers.items() if k in _LAYER_BY_ID}
    if not any(known.values()):
        known[FALLBACK_LAYER_ID] = True
    return known


def get_layer_weight(flags: LayerFlags, layer_id: str) -> float:
    layer = _LAYER_BY_ID.get(layer_id)
    if layer is None:
        return 0.0
    enabled = bool(getattr(flags, layer_id, False))
    return layer.default_weight if enabled else 0.0


def get_active_layers(flags: LayerFlags) -> list[tuple[BaseProbabilityLayer, float]]:
    active: list[tuple[BaseProbabilityLayer, float]] = []
    for layer in PROBABILITY_LAYER_REGISTRY:
        w = get_layer_weight(flags, layer.layer_id)
        if w > 0:
            active.append((layer, w))
    return active
