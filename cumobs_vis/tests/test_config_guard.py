"""Tests for the config-compatibility guard."""

from cumobs_vis.config_guard import check_configs
from cumobs_vis.data_loader import RunData


def _run(name, config):
    return RunData(name=name, color="#000", config=config)


BASE = dict(window_size=20, ddof=1, detrend_order=0,
            cumulant_estimator="unbiased", shc_mode="rank")


def test_no_warnings_for_identical_configs():
    a = _run("a", dict(BASE))
    b = _run("b", dict(BASE))
    assert check_configs([a, b]) == []


def test_single_configured_run_has_no_warnings():
    a = _run("a", dict(BASE))
    b = RunData(name="b", color="#000", config=None)  # no config -> excluded
    assert check_configs([a, b]) == []


def test_ddof_and_shc_mode_mismatch():
    a = _run("a", dict(BASE))
    b = _run("b", {**BASE, "ddof": 2, "shc_mode": "full"})
    warns = check_configs([a, b])
    fields = {w.field for w in warns}
    assert "ddof" in fields
    assert "shc_mode" in fields
    # affected keys are surfaced for the banner.
    ddof_w = next(w for w in warns if w.field == "ddof")
    assert "spectral_heat_capacity" in ddof_w.affected_keys
    assert ddof_w.values == {"a": 1, "b": 2}


def test_estimator_and_detrend_and_window():
    a = _run("a", dict(BASE))
    b = _run("b", {**BASE, "cumulant_estimator": "biased",
                   "detrend_order": 1, "window_size": 30})
    fields = {w.field for w in check_configs([a, b])}
    assert {"cumulant_estimator", "detrend_order", "window_size"} <= fields


def test_missing_field_in_one_config_is_skipped():
    a = _run("a", {"ddof": 1})               # only has ddof
    b = _run("b", {"shc_mode": "full"})      # only has shc_mode
    # neither field is present in *both*, so nothing is comparable.
    assert check_configs([a, b]) == []
