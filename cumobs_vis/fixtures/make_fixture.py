"""Synthetic JSONL fixture generator (plan §6).

Writes a self-consistent fake run set so the visualizer can be developed and
demoed with **no training run**. It injects a transition and exercises every
feature the app cares about:

* a rising SHC, a ``total_fourth_cumulant`` spike, and a leading-mode
  condensation (PR down, spectral_gap / top_eigenvalue_fraction up) at the
  transition, all derived from a per-step covariance spectrum so the classical
  distances are consistent with the spectrum;
* a ``constant_window == 1.0`` dead-window stretch (validity shading);
* ``<scalar>/se`` sidecar keys on several scalars (uncertainty bands);
* per-group ``group_shc`` / ``group_pr`` / ``group_total_variance`` sub-series
  (the transition localises on group A);
* **two probe-set runs** (``train-probes`` / ``test-probes``) with *incompatible*
  configs (different ddof + shc_mode) to trip the config-compatibility guard;
* a metrics file (``step, loss, acc``) for the train run's reference panel;
* a ``.npz`` covariance-spectrum sidecar for the train run, which enables the
  §4i click-to-set-reference feature.

Usage::

    python fixtures/make_fixture.py --out _fixture_run
    python launch.py --manifest _fixture_run/runs.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

N_PROBES = 24
N_GROUP_A = 8  # group A is where the transition localises


def _spectrum(rng: np.random.Generator, step: int, transition: int,
              amp_scale: float) -> np.ndarray:
    """A length-n covariance spectrum: isotropic bulk + a detaching leading mode."""
    overall = 0.3 + 0.7 * math.exp(-step / 500.0)          # loss converging
    bulk = overall * (0.05 + 0.01 * rng.standard_normal(N_PROBES))
    bulk = np.abs(bulk)
    bulk *= np.linspace(1.2, 0.6, N_PROBES)                 # gentle decreasing shape

    rise = 1.0 / (1.0 + math.exp(-(step - transition) / 15.0))
    decay = math.exp(-max(0, step - (transition + 80)) / 120.0)
    amp = amp_scale * rise * decay                          # detached-mode energy
    lam = bulk.copy()
    lam[0] += amp
    lam = np.sort(np.clip(lam, 0.0, None))[::-1]
    return lam


def _spectral_scalars(lam: np.ndarray) -> dict[str, float]:
    total = float(lam.sum())
    l2 = float(np.square(lam).sum())
    nonzero = lam[lam > 1e-9]
    out: dict[str, float] = {}
    out["total_second_cumulant"] = total
    out["spectral_heat_capacity"] = float(np.var(nonzero)) if nonzero.size else float("nan")
    out["participation_ratio"] = (total * total / l2) if l2 > 0 else float("nan")
    out["spectral_gap"] = float(lam[0] - lam[1]) if lam.size > 1 else float("nan")
    out["top_eigenvalue_fraction"] = float(lam[0] / total) if total > 0 else float("nan")
    if total > 0:
        p = nonzero / nonzero.sum()
        ent = float(-(p * np.log(p)).sum())
        out["von_neumann_entropy"] = ent
        out["effective_rank"] = float(math.exp(ent))
    return out


def _classical_distances(ref: np.ndarray, cur: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    sr, sc = ref.sum(), cur.sum()
    if sr > 0 and sc > 0:
        p, q = ref / sr, cur / sc
        out["classical_fidelity"] = float(np.square(np.sqrt(p * q).sum()))
        out["trace_distance"] = 0.5 * float(np.abs(p - q).sum())
    denom = max(float(np.linalg.norm(ref)), float(np.linalg.norm(cur)))
    if denom > 0:
        out["spectral_drift"] = float(np.linalg.norm(ref - cur) / denom)
    return out


def _group_spectral(rng, step, transition, amp_scale, idxs):
    """SHC / PR / total-variance for a probe subgroup (its own mini-spectrum)."""
    lam = _spectrum(rng, step, transition, amp_scale)[: len(idxs)]
    lam = np.sort(np.clip(lam, 0, None))[::-1]
    total = float(lam.sum()); l2 = float(np.square(lam).sum())
    shc = float(np.var(lam[lam > 1e-9])) if np.any(lam > 1e-9) else float("nan")
    pr = (total * total / l2) if l2 > 0 else float("nan")
    return shc, pr, total


def generate_run(path_jsonl: str, *, seed: int, transition: int,
                 amp_scale: float, dead_range: tuple[int, int] | None,
                 with_sidecar: bool) -> None:
    rng = np.random.default_rng(seed)
    steps = list(range(100, 800, 10))  # observables start at 100 => buffer-fill 0..100
    rows: list[dict] = []
    spectra: dict[int, np.ndarray] = {}
    prev: np.ndarray | None = None

    se_keys = ("spectral_heat_capacity", "participation_ratio",
               "total_second_cumulant", "total_fourth_cumulant")

    for step in steps:
        dead = dead_range is not None and dead_range[0] <= step <= dead_range[1]
        lam = _spectrum(rng, step, transition, amp_scale)
        spectra[step] = lam
        row: dict[str, float | int] = {"step": step}

        # first cumulant (mean probe loss) — always present, decays.
        row["total_first_cumulant"] = 2.0 * math.exp(-step / 400.0) + 0.02 * rng.standard_normal()
        row["excess_loss"] = max(0.0, row["total_first_cumulant"] - 0.1)

        # health
        row["health/use_gram"] = 1.0
        row["health/constant_window"] = 1.0 if dead else 0.0
        row["health/rank"] = 0.0 if dead else float(int(np.sum(lam > 1e-9)))

        if not dead:
            sc = _spectral_scalars(lam)
            row.update(sc)
            # non-Gaussianity: ~0 with a spike at the transition.
            g = math.exp(-((step - transition) / 25.0) ** 2)
            row["total_fourth_cumulant"] = 40.0 * amp_scale * g + 0.5 * rng.standard_normal()
            row["total_third_cumulant"] = 8.0 * amp_scale * g + 0.3 * rng.standard_normal()
            # early-warning: autocorr rises through the transition.
            row["autocorr_lag1"] = float(np.clip(0.2 + 0.6 * (1 / (1 + math.exp(-(step - transition) / 20))) * math.exp(-max(0, step - transition) / 200), -1, 1))
            # distances vs previous measurement.
            if prev is not None:
                row.update(_classical_distances(prev, lam))
                # a synthetic representation drift (eigenvector-sensitive proxy).
                drift = row.get("spectral_drift", 0.0)
                row["cka_distance"] = float(np.clip(drift * 1.3, 0, 1))
                row["cka_similarity"] = 1.0 - row["cka_distance"]

            # per-group sub-series (A localises the transition, B stays quiet).
            a_shc, a_pr, a_var = _group_spectral(rng, step, transition, amp_scale, range(N_GROUP_A))
            b_shc, b_pr, b_var = _group_spectral(rng, step, transition, 0.05, range(N_GROUP_A, N_PROBES))
            row["group_shc/A"] = a_shc; row["group_pr/A"] = a_pr; row["group_total_variance/A"] = a_var
            row["group_shc/B"] = b_shc; row["group_pr/B"] = b_pr; row["group_total_variance/B"] = b_var

            # jackknife SE sidecars for a few scalars.
            for k in se_keys:
                v = row.get(k)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    row[f"{k}/se"] = abs(v) * 0.12 + 0.01 * abs(rng.standard_normal())

        prev = lam
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(path_jsonl)), exist_ok=True)
    with open(path_jsonl, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    if with_sidecar:
        arrays = {f"{s}/covariance_spectrum": spectra[s] for s in spectra}
        np.savez(path_jsonl + ".npz", **arrays)


def write_metrics(path: str) -> None:
    rng = np.random.default_rng(7)
    with open(path, "w", encoding="utf-8") as fh:
        for step in range(0, 800, 5):
            loss = 2.2 * math.exp(-step / 300.0) + 0.03 * rng.standard_normal()
            # accuracy climbs, with a little kink near the transition.
            acc = float(np.clip(1.0 - math.exp(-step / 220.0) + 0.01 * rng.standard_normal(), 0, 1))
            fh.write(json.dumps({"step": step, "loss": round(loss, 5),
                                 "acc": round(acc, 5)}) + "\n")


def write_config(path: str, **overrides) -> None:
    cfg = dict(window_size=20, eval_interval=10, update_interval=10, ddof=1,
               detrend_order=0, cumulant_estimator="unbiased", shc_mode="rank",
               order=4, top_k=3)
    cfg.update(overrides)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic cumobs fixture.")
    ap.add_argument("--out", default="_fixture_run", help="output directory")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    # train-probes: strong transition, sidecar + metrics, config A.
    train_jsonl = os.path.join(out, "train.jsonl")
    generate_run(train_jsonl, seed=0, transition=400, amp_scale=1.0,
                 dead_range=(650, 690), with_sidecar=True)
    write_metrics(os.path.join(out, "train.metrics.jsonl"))
    write_config(os.path.join(out, "train.config.json"))  # ddof=1, shc_mode=rank

    # test-probes: weaker/later transition, NO sidecar, INCOMPATIBLE config B.
    test_jsonl = os.path.join(out, "test.jsonl")
    generate_run(test_jsonl, seed=1, transition=470, amp_scale=0.55,
                 dead_range=None, with_sidecar=False)
    write_config(os.path.join(out, "test.config.json"), ddof=2, shc_mode="full")

    manifest = {
        "runs": [
            {"name": "train-probes", "color": "#3b82f6",
             "observables": "train.jsonl", "config": "train.config.json",
             "metrics": "train.metrics.jsonl"},
            {"name": "test-probes", "color": "#ef4444",
             "observables": "test.jsonl", "config": "test.config.json"},
        ]
    }
    manifest_path = os.path.join(out, "runs.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"wrote fixture to {out}")
    print(f"manifest: {manifest_path}")
    print(f"launch:   python launch.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
