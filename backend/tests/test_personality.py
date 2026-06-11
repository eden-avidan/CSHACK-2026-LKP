"""Tests for personality mobility heuristic."""

from __future__ import annotations

import numpy as np

from app.engine.personality_heuristic import apply_mobility_scale, mobility_multiplier
from app.models.personality import PersonalityProfile


def test_mobility_decreases_with_age():
    young = mobility_multiplier(PersonalityProfile(age=25, fitness=3, injured=False))
    old = mobility_multiplier(PersonalityProfile(age=75, fitness=3, injured=False))
    assert young > old


def test_toddler_mobility_is_lower_than_young_adult():
    toddler = mobility_multiplier(PersonalityProfile(age=2, fitness=3, injured=False))
    adult = mobility_multiplier(PersonalityProfile(age=20, fitness=3, injured=False))
    assert toddler < adult


def test_mobility_peaks_before_declining_in_older_age():
    teen = mobility_multiplier(PersonalityProfile(age=16, fitness=3, injured=False))
    adult = mobility_multiplier(PersonalityProfile(age=25, fitness=3, injured=False))
    older = mobility_multiplier(PersonalityProfile(age=70, fitness=3, injured=False))
    assert adult > teen
    assert adult > older


def test_fitness_boosts_mobility_above_one():
    low = mobility_multiplier(PersonalityProfile(age=35, fitness=1, injured=False))
    high = mobility_multiplier(PersonalityProfile(age=35, fitness=5, injured=False))
    assert high > 1.0
    assert low < high


def test_injured_reduces_mobility():
    healthy = mobility_multiplier(PersonalityProfile(age=35, fitness=3, injured=False))
    injured = mobility_multiplier(PersonalityProfile(age=35, fitness=3, injured=True))
    assert injured < healthy
    assert injured < 1.0


def test_apply_mobility_scale_keeps_lkp_unchanged():
    p = np.zeros((16, 16))
    p[8, 8] = 1.0
    p[8, 9] = 0.5
    out = apply_mobility_scale(p, lkp_row=8, lkp_col=8, mobility=0.5)
    assert out[8, 8] == 1.0
    assert out[8, 9] < 0.5
