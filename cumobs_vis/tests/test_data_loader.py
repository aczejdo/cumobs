"""Tests for the dataLoader (manifest + JSONL + metrics + npz sidecar)."""

import json
import math

import numpy as np

from cumobs_vis.data_loader import (
    load_manifest,
    load_run,
    load_run_from_text,
)


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_parse_scalars_se_health_groups(tmp_path):
    obs = tmp_path / "run.jsonl"
    _write(obs, [
        {"step": 10, "spectral_heat_capacity": 1.0, "spectral_heat_capacity/se": 0.1,
         "health/constant_window": 0.0, "group_shc/A": 0.5, "participation_ratio": 4.0},
        # step 20 is missing participation_ratio -> should NaN-fill.
        {"step": 20, "spectral_heat_capacity": 2.0, "spectral_heat_capacity/se": 0.2,
         "health/constant_window": 1.0, "group_shc/A": 0.7},
    ])
    run = load_run(str(obs), name="r", color="#000")

    assert run.steps == [10, 20]
    # scalar series aligned, NaN where missing.
    assert run.series["spectral_heat_capacity"] == [1.0, 2.0]
    assert run.series["participation_ratio"][0] == 4.0
    assert math.isnan(run.series["participation_ratio"][1])
    # se is a sidecar, not a trace.
    assert run.se["spectral_heat_capacity"] == [0.1, 0.2]
    assert "spectral_heat_capacity/se" not in run.series
    # health parsed under stripped key.
    assert run.health["constant_window"] == [0.0, 1.0]
    # per-group series parsed separately.
    assert run.groups["group_shc"]["A"] == [0.5, 0.7]
    # primary scalar keys exclude se / health / group / slash keys.
    prim = run.primary_scalar_keys()
    assert "spectral_heat_capacity" in prim and "participation_ratio" in prim
    assert "group_shc" not in prim


def test_tolerates_missing_and_bad_lines(tmp_path):
    obs = tmp_path / "run.jsonl"
    with open(obs, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": 5, "trace_distance": 0.1}) + "\n")
        fh.write("not valid json\n")           # tolerated
        fh.write("\n")                          # blank tolerated
        fh.write(json.dumps({"step": 15}) + "\n")  # no scalars
    run = load_run(str(obs), name="r", color="#000")
    assert run.steps == [5, 15]
    assert run.series["trace_distance"][0] == 0.1
    assert math.isnan(run.series["trace_distance"][1])


def test_load_run_from_text():
    text = "\n".join([
        json.dumps({"step": 0, "total_second_cumulant": 3.0}),
        json.dumps({"step": 10, "total_second_cumulant": 2.0}),
    ])
    run = load_run_from_text(text, name="dropped", color="#123")
    assert run.steps == [0, 10]
    assert run.series["total_second_cumulant"] == [3.0, 2.0]


def test_metrics_and_config_and_sidecar(tmp_path):
    obs = tmp_path / "train.jsonl"
    _write(obs, [{"step": 10, "spectral_heat_capacity": 1.0}])
    metrics = tmp_path / "train.metrics.jsonl"
    _write(metrics, [{"step": 0, "loss": 2.0, "acc": 0.1},
                     {"step": 5, "loss": 1.0, "acc": 0.5}])
    cfg = tmp_path / "train.config.json"
    cfg.write_text(json.dumps({"window_size": 20, "ddof": 1}), encoding="utf-8")
    # npz sidecar with a covariance spectrum.
    np.savez(str(obs) + ".npz", **{"10/covariance_spectrum": np.array([1.0, 0.5, 0.0])})

    run = load_run(str(obs), name="train", color="#000",
                   config_path=str(cfg), metrics_path=str(metrics))
    assert run.has_metrics()
    assert run.metrics_steps == [0, 5]
    assert run.metrics["loss"] == [2.0, 1.0]
    assert run.config["window_size"] == 20
    assert run.has_covariance_spectrum
    spectra = run.load_covariance_spectra()
    assert spectra[10] == [1.0, 0.5, 0.0]


def test_manifest_relative_paths_and_default_colors(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"step": 1, "spectral_gap": 0.2}) + "\n",
                                      encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps({"step": 1, "spectral_gap": 0.3}) + "\n",
                                      encoding="utf-8")
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps({"runs": [
        {"name": "A", "observables": "a.jsonl"},          # no color -> palette
        {"name": "B", "color": "#ff0000", "observables": "b.jsonl"},
    ]}), encoding="utf-8")

    runs = load_manifest(str(manifest))
    assert [r.name for r in runs] == ["A", "B"]
    assert runs[1].color == "#ff0000"
    assert runs[0].color.startswith("#")   # assigned from palette
    assert runs[0].series["spectral_gap"] == [0.2]
