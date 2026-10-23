# cumobs · scalar-observable visualizer

A local, interactive dashboard for the **scalar** observables produced by the
[`cumobs`](../cumobs_pkg) monitor. Clean scalar line plots stacked on a locked
shared step axis, a loss/accuracy reference curve pinned on top, toggle-able
runs / probe-set subsets, uncertainty bands, validity shading, a
config-compatibility guard, and event markers.

It is a **scoped** tool: vector/matrix observables (spectra, per-mode quantities,
heatmaps) are out of scope by design. Built from
[`scalar_observable_visualizer_plan.md`](scalar_observable_visualizer_plan.md).

---

## Tech stack (a note on the deviation)

The plan's default recommendation is a Vite + React + uPlot single-page app, with
a **Bokeh** app as the stated fallback "if the repo already has a Python-frontend
toolchain / the user prefers Python." This machine has **no Node toolchain**, and
the whole project is Python (training is `cumobs` + torch), so the Bokeh fallback
is used. Bokeh provides everything the plan asks for natively — linked crosshairs,
`Span` markers / validity shading, `varea` bands, locked shared ranges, and
server-side file access for the §4i spectrum recompute — in one `pip install` and
one `bokeh serve`.

The **visualizer itself depends only on `bokeh` and `numpy`** (not torch, not
`cumobs`), so it stays a lightweight, standalone analysis tool. `cumobs` is used
read-only by the training demo script.

---

## Install

From the repo root (a `uv`-managed project with a `.venv`):

```powershell
uv pip install --python .venv\Scripts\python.exe bokeh pytest
```

or, in any environment:

```bash
pip install -r cumobs_vis/requirements.txt
```

---

## Quick start (no training needed)

A synthetic fixture with an injected transition ships for instant exploration:

```powershell
cd cumobs_vis
python fixtures\make_fixture.py --out _fixture_run
python launch.py --manifest _fixture_run\runs.json
```

A browser tab opens at `http://localhost:5006/server`. The fixture has two runs
(`train-probes` / `test-probes`) with **incompatible configs** (so the config
guard fires), a dead-window stretch, SE bands, per-group sub-series, and a
covariance-spectrum sidecar (so click-to-set-reference is enabled).

## Final testing — train a model and visualize it

```powershell
cd cumobs_vis
python scripts\train_and_visualize.py                 # train, then open the app
python scripts\train_and_visualize.py --no-serve      # just write artifacts
python scripts\train_and_visualize.py --mismatch-configs   # also demo the config guard
```

This trains a small MLP with a **curriculum transition** (the training target
switches `w1 -> w2` at `--transition`, default step 300), watched by **two probe
sets** via two `cumobs.CumulantMonitor`s. It writes two observables JSONL files,
a metrics file, per-run config JSON, a `covariance_spectrum` `.npz` sidecar, and a
`runs.json` manifest into `_train_run/`, then launches the visualizer. Watch the
SHC spike, the participation ratio drop (one mode condensing), and the distance
panels jump at the transition.

---

## Input contract

The visualizer consumes the flattened rows written by `cumobs.JSONLLogger`
(one JSON object per line, a `step` field, flat scalar keys). A **run** = one
probe set + config = one observables JSONL, described by a manifest:

```jsonc
// runs.json
{
  "runs": [
    { "name": "train-probes", "color": "#3b82f6",
      "observables": "train.jsonl",
      "config":      "train.config.json",     // optional
      "metrics":     "train.metrics.jsonl" }, // optional (reference panel)
    { "name": "test-probes",  "color": "#ef4444", "observables": "test.jsonl" }
  ]
}
```

Key handling (per `cumobs/docs/logging.md`): `health/*` → validity diagnostics;
`<scalar>/se` → uncertainty-band sidecar (not a trace); `group_*/<label>` →
per-group sub-series; a full `covariance_spectrum` in the `.npz` array sidecar →
enables click-to-set-reference. Missing keys are tolerated (NaN-filled). You can
also drag a `.jsonl` into the sidebar's "add run" control.

**`--manifest` also accepts a directory.** Point it at a run folder and the app
uses its `runs.json` if present, else auto-discovers the `*.jsonl` files inside
(pairing `metrics.jsonl` / `config.json` by convention):

```powershell
python launch.py --manifest .\logs\mnist_cumobs_cnn_20260701_162756
```

**`ProbeSetCollection` logs are handled.** When the monitor watches a collection,
every key is prefixed with the probe-set name (`train_mix/spectral_heat_capacity`)
and all probe sets share one JSONL. The loader **splits that file into one run per
probe set** (prefix stripped), so e.g. an MNIST run with mix + per-digit probe
sets becomes `train_mix`, `test_mix`, `train_digit_0…9`, … — each toggleable.
The "*mix*" sets are shown first; if there are many, only the first two are
visible by default (toggle the rest on in the sidebar). Nested configs
(`{"config": {"cumobs": {...}}}`) and metric keys like `train/loss` /
`train/accuracy` are recognised automatically.

---

## Features (plan → implementation)

- **Locked shared step axis** — one x-range object across the reference panel and
  every scalar panel; zoom/pan/brush on any panel moves all.
- **Linked crosshair + readout** — one shared vertical crosshair; a readout box
  shows *every* visible trace's value at the hovered step.
- **Non-redundant defaults + collapse groups** — one representative per redundancy
  axis (Concentration / Distance-shape / Distance-subspace); expand a panel to
  overlay members, and z-score to put them on one scale.
- **Run + per-group toggles** — color = run everywhere; toggle a run to hide its
  traces/bands/spans in every panel; toggle its `group_*` sub-series.
- **Uncertainty bands** — `value ± k·se` from `<scalar>/se` via a *pluggable*
  band provider (jackknife SE shipped; RMT/MP-edge provider stubbed).
- **Config-compatibility guard** — a dismissible banner when overlaid runs differ
  in `ddof` / `cumulant_estimator` / `detrend_order` / `shc_mode` / `window_size`.
- **Validity shading** — buffer-fill and dead-window (`constant_window == 1.0`)
  regions; PR / effective_rank carry the window-cap caveat in their tooltip.
- **Log-y + normalize toggles** — per-panel and global log-y; per-trace z-score.
- **Event markers** — create / update / delete, categorized (`curriculum` /
  `detected` / `manual`) and styled, **import/export as JSON**.
- **Click-to-set-reference (§4i)** — enabled iff `covariance_spectrum` is logged;
  click a panel to recompute the *classical* distances against that step
  (dashed overlays). Disabled with an explanation otherwise. Quantum/CKA
  distances are never recomputed client-side.
- **Per-mode → scalar reduction (§4j)** — available in `cumobs_vis.reductions`
  (advanced, off by default; carries the mode-crossing caveat).

---

## Architecture (`cumobs_vis/`)

| module | role |
|--------|------|
| `catalog.py` | scalar key catalog: families, tooltips, transforms |
| `collapse_registry.py` | redundancy/collapse groups + panel resolution |
| `data_loader.py` | manifest + JSONL + metrics + `.npz` ingestion |
| `run_store.py` | loaded runs, visibility, colors, per-group state |
| `band_provider.py` | pluggable uncertainty-band interface (+ RMT stub) |
| `config_guard.py` | pairwise config-compatibility warnings |
| `validity_spans.py` | buffer-fill / dead-window / imag-ratio regions |
| `marker_store.py` | marker CRUD + JSON import/export + category styles |
| `reductions.py` | optional per-mode → scalar reduction hook |
| `panels.py` | Bokeh figure/glyph builders + crosshair controller |
| `app.py` | `VisApp` — state + widgets + callbacks + rebuild |
| `server.py` | `bokeh serve` entry point |

`launch.py` wraps `bokeh serve`; `fixtures/make_fixture.py` generates the
synthetic run; `scripts/train_and_visualize.py` is the end-to-end training demo.

---

## Tests

```powershell
# from the repo root
.venv\Scripts\python.exe -m pytest cumobs_vis\tests -q
```

Covers `data_loader`, `collapse_registry`, `config_guard`, and `validity_spans`.
