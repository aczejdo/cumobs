"""End-to-end: train a small model with ``cumobs``, then open the visualizer.

This is the "final testing" script. It:

1. trains a small MLP on a regression task with an **injected curriculum
   transition** — at ``--transition`` the training target switches (w1 -> w2),
   so the *fixed* probe losses reorganise: SHC spikes, the participation ratio
   drops (one mode condenses), the connected 4th cumulant blips, and the
   inter-measurement distances jump;
2. watches the model with **two probe sets** (``train-probes`` = in-distribution,
   ``test-probes`` = harder / larger-norm), each via its own
   :class:`cumobs.CumulantMonitor` + :class:`cumobs.JSONLLogger`, producing two
   separate observables JSONL files = two runs;
3. logs jackknife SE (uncertainty bands), per-group observables (sub-series), and
   a full ``covariance_spectrum`` ``.npz`` sidecar for ``train-probes`` (enables
   the §4i click-to-set-reference);
4. writes a training ``metrics`` file (step, loss, acc) for the reference panel;
5. writes a ``runs.json`` manifest and launches the Bokeh visualizer.

The ``cumobs`` package is used read-only (imported, never modified).

Run::

    python scripts/train_and_visualize.py                 # train + open the app
    python scripts/train_and_visualize.py --no-serve      # just write artifacts
    python scripts/train_and_visualize.py --mismatch-configs   # also demo the guard
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
from torch import nn

# --- make the sibling cumobs package importable (read-only) ----------------
_SCRIPT = os.path.abspath(__file__)
_VIS_DIR = os.path.dirname(os.path.dirname(_SCRIPT))            # .../cumobs_vis
_REPO_ROOT = os.path.dirname(_VIS_DIR)                          # .../cumobs_v3
_CUMOBS_PKG = os.path.join(_REPO_ROOT, "cumobs_pkg")
if os.path.isdir(_CUMOBS_PKG) and _CUMOBS_PKG not in sys.path:
    sys.path.insert(0, _CUMOBS_PKG)

from cumobs import (  # noqa: E402
    CumulantConfig,
    CumulantMonitor,
    JSONLLogger,
    LoggingConfig,
    ProbeSet,
)


def _make_config(shc_mode: str = "rank") -> CumulantConfig:
    return CumulantConfig(
        window_size=20,
        eval_interval=1,        # append a probe-loss row every step
        update_interval=10,     # compute observables every 10 steps
        order=4,
        top_k=3,
        shc_mode=shc_mode,
        compute_eigenvectors=True,     # IPR / block energy / CKA / quantum
        per_group=True,                # group_shc / group_pr / group_total_variance
        materialize_matrices=True,     # quantum distances
        distance_flavor="both",        # classical + quantum distances
        compute_cka_distance=True,     # cka_similarity / cka_distance
        error_bars="jackknife",        # <scalar>/se  -> uncertainty bands
    )


def _config_to_json(cfg: CumulantConfig) -> dict:
    """The JSON-safe subset the visualizer's config guard / validity spans read."""
    return {
        "window_size": cfg.window_size,
        "eval_interval": cfg.eval_interval,
        "update_interval": cfg.update_interval,
        "ddof": cfg.ddof,
        "detrend_order": cfg.detrend_order,
        "cumulant_estimator": cfg.cumulant_estimator,
        "shc_mode": cfg.shc_mode,
        "order": cfg.order,
        "top_k": cfg.top_k,
        "imag_ratio_warn": cfg.imag_ratio_warn,
    }


def train(out: str, *, steps: int, transition: int, seed: int,
          mismatch_configs: bool) -> str:
    torch.manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    d = 8

    # two linear teacher targets; training switches w1 -> w2 at the transition.
    w1 = torch.randn(d, dtype=torch.float64)
    w2 = torch.randn(d, dtype=torch.float64)

    X = torch.randn(1024, d, dtype=torch.float64)

    def target(w, x):
        return x @ w

    # --- probe sets -----------------------------------------------------------
    # train-probes: in-distribution; targets fixed to the ORIGINAL task (w1), so
    # the switch to w2 shows up as a real probe-loss transition.
    n_tr = 40
    ptr_x = torch.randn(n_tr, d, dtype=torch.float64)
    ptr_y = target(w1, ptr_x)
    tr_labels = ["A"] * (n_tr // 2) + ["B"] * (n_tr - n_tr // 2)

    # test-probes: harder / larger-norm (OOD), also referenced to w1.
    n_te = 30
    pte_x = 2.5 * torch.randn(n_te, d, dtype=torch.float64)
    pte_y = target(w1, pte_x)
    te_labels = ["A"] * (n_te // 2) + ["B"] * (n_te - n_te // 2)

    def loss_fn_tr(model, idx):
        pred = model(ptr_x[idx]).squeeze(-1)
        return (pred - ptr_y[idx]) ** 2

    def loss_fn_te(model, idx):
        pred = model(pte_x[idx]).squeeze(-1)
        return (pred - pte_y[idx]) ** 2

    probes_tr = ProbeSet(torch.arange(n_tr), loss_fn_tr,
                         group_labels=tr_labels, name="train-probes")
    probes_te = ProbeSet(torch.arange(n_te), loss_fn_te,
                         group_labels=te_labels, name="test-probes")

    # --- model + monitors -----------------------------------------------------
    model = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1)).double()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)

    cfg_tr = _make_config()
    cfg_te = _make_config(shc_mode="full" if mismatch_configs else "rank")

    obs_tr = os.path.join(out, "train.jsonl")
    obs_te = os.path.join(out, "test.jsonl")
    logger_tr = JSONLLogger(obs_tr)
    logger_te = JSONLLogger(obs_te)

    # full covariance_spectrum -> .npz sidecar for train-probes (enables §4i).
    lc_tr = LoggingConfig(vector_policy="summary",
                          vector_overrides={"covariance_spectrum": "full"})
    lc_te = LoggingConfig(vector_policy="summary")

    mon_tr = CumulantMonitor(model, probes_tr, cfg_tr, logger=logger_tr,
                             logging_config=lc_tr)
    mon_te = CumulantMonitor(model, probes_te, cfg_te, logger=logger_te,
                             logging_config=lc_te)

    # --- training loop --------------------------------------------------------
    metrics_path = os.path.join(out, "train.metrics.jsonl")
    y_var = float(torch.var(target(w1, X)))
    print(f"training {steps} steps; target switches w1->w2 at step {transition}")
    with open(metrics_path, "w", encoding="utf-8") as mfh:
        for step in range(steps):
            w = w1 if step < transition else w2
            idx = torch.randint(0, len(X), (64,))
            xb = X[idx]
            yb = target(w, xb)
            pred = model(xb).squeeze(-1)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            # training metrics for the reference panel.
            with torch.no_grad():
                acc = float(max(0.0, 1.0 - float(loss.detach()) / (y_var + 1e-12)))
            mfh.write(json.dumps({"step": step, "loss": round(float(loss.detach()), 6),
                                  "acc": round(acc, 6)}) + "\n")

            mon_tr.step(step)
            mon_te.step(step)

            if step % max(1, steps // 10) == 0:
                print(f"  step {step:>4}  train_loss={float(loss.detach()):.4e}")

    # flush the covariance-spectrum arrays for train-probes.
    npz = logger_tr.flush_arrays()
    print(f"wrote covariance-spectrum sidecar: {npz}")

    # --- config sidecars ------------------------------------------------------
    with open(os.path.join(out, "train.config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config_to_json(cfg_tr), fh, indent=2)
    with open(os.path.join(out, "test.config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config_to_json(cfg_te), fh, indent=2)

    # --- manifest -------------------------------------------------------------
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
    print(f"wrote manifest: {manifest_path}")
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(_VIS_DIR, "_train_run"),
                    help="output directory for the run artifacts")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--transition", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mismatch-configs", action="store_true",
                    help="give the two runs different shc_mode to demo the config guard")
    ap.add_argument("--no-serve", action="store_true",
                    help="write artifacts but do not launch the visualizer")
    ap.add_argument("--port", type=int, default=5006)
    args = ap.parse_args()

    manifest = train(os.path.abspath(args.out), steps=args.steps,
                     transition=args.transition, seed=args.seed,
                     mismatch_configs=args.mismatch_configs)

    if args.no_serve:
        print("\nartifacts written. To view them, run:")
        print(f"  python launch.py --manifest {manifest}")
        return 0

    launcher = os.path.join(_VIS_DIR, "launch.py")
    cmd = [sys.executable, launcher, "--manifest", manifest, "--port", str(args.port)]
    print("\nlaunching visualizer:", " ".join(cmd))
    print("(Ctrl-C to stop the server)")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
