"""Tests for validity-span derivation (buffer-fill / dead-window / imag-ratio)."""

import math

from cumobs_vis.data_loader import RunData
from cumobs_vis.validity_spans import compute_spans


def _run_with(steps, series, health, config=None):
    return RunData(name="r", color="#000", steps=list(steps),
                   series=series, health=health, config=config or {})


def test_buffer_fill_ends_at_first_finite_measurement():
    steps = [100, 110, 120]
    # first finite value is at step 110.
    series = {"spectral_heat_capacity": [math.nan, 1.0, 2.0]}
    run = _run_with(steps, series, {}, {"window_size": 20})
    spans = compute_spans(run)
    buf = [s for s in spans if s.kind == "buffer_fill"]
    assert len(buf) == 1
    assert buf[0].start is None          # extends to the left edge
    assert buf[0].end == 110.0
    assert "window_size = 20" in buf[0].reason


def test_dead_window_spans_cover_constant_window_stretch():
    steps = [100, 110, 120, 130, 140]
    series = {"spectral_heat_capacity": [1.0, 2.0, 1.5, 1.2, 1.1]}
    # constant_window == 1.0 over steps 110..120.
    health = {"constant_window": [0.0, 1.0, 1.0, 0.0, 0.0]}
    run = _run_with(steps, series, health)
    spans = compute_spans(run)
    dead = [s for s in spans if s.kind == "dead_window"]
    assert len(dead) == 1
    # spacing is 10 -> pad 5; region covers [105, 125].
    assert dead[0].start == 105.0
    assert dead[0].end == 125.0


def test_two_separate_dead_windows():
    steps = [0, 10, 20, 30, 40, 50]
    series = {"participation_ratio": [1.0] * 6}
    health = {"constant_window": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]}
    run = _run_with(steps, series, health)
    dead = [s for s in compute_spans(run) if s.kind == "dead_window"]
    assert len(dead) == 2


def test_imag_ratio_off_by_default_on_by_flag():
    steps = [0, 10, 20]
    series = {"spectral_gap": [0.1, 0.2, 0.3]}
    health = {"contracted_fourth_cumulant_imag_ratio": [0.0, 1.0, 0.0]}
    run = _run_with(steps, series, health, {"imag_ratio_warn": 1e-3})
    assert not any(s.kind == "imag_ratio" for s in compute_spans(run))
    with_imag = compute_spans(run, include_imag_ratio=True)
    assert any(s.kind == "imag_ratio" for s in with_imag)
