# Build plan — scalar-observable visualizer for `cumobs`

A local, interactive dashboard for the **scalar** observables produced by the
`cumobs` monitor. Clean scalar line plots, stacked on a shared step axis, with a
loss/accuracy reference curve pinned at the top, toggle-able probe sets/runs,
event markers, and — the two features that change what the plots *mean* —
uncertainty bands and a config-compatibility guard.

This is a scoped tool, not a general plotting library. Vector/matrix observables
(spectra, per-mode quantities, heatmaps) are **explicitly out of scope**.

---

## 0. First step — read the package docs before writing any code

The `cumobs` package and its docs are in this repo. **Read them first** and let
them, not this plan, be authoritative for exact names and formats:

- `docs/observables.md` — the per-key catalog. Extract the exact **scalar** key
  names, their buckets (`scalars` / `vectors` / `matrices` / `health`), and which
  are per-group (suffixed `.../<label>`).
- `docs/concepts.md` — the meaning of each observable (needed for tooltips and
  the collapse groups below).
- `docs/configuration.md` — every `CumulantConfig` field (needed for the config
  guard) and the D1–D6 decisions.
- `docs/logging.md` + `README` — the **flattening policy** and `JSONLLogger`
  format. This defines the input contract; do not invent one.

The key lists in this plan are a **starting point to verify against the catalog**,
not a source of truth. If a name here disagrees with `observables.md`, the
catalog wins.

---

## 1. Data contract (input)

The `cumobs` monitor logs one measurement per computation step via
`JSONLLogger("run.jsonl")` as **flattened rows** (one JSON object per line, flat
keys, a `step` field). The visualizer consumes these files. Confirm the exact
flattening (key separators, how `scalars`/`health` are prefixed, how per-group
keys like `group_shc/A` appear) from `logging.md`.

**A "run"** = one probe set + config = one observables JSONL, plus optional
metadata:

```jsonc
// runs.json — the primary input manifest (also support drag-drop as convenience)
{
  "runs": [
    {
      "name": "train-probes",
      "color": "#3b82f6",
      "observables": "runs/train.jsonl",
      "config": "runs/train.config.json",   // optional; else parse from JSONL if embedded
      "metrics": "runs/train.metrics.jsonl"  // optional; for the top reference panel
    },
    { "name": "test-probes", "color": "#ef4444", "observables": "runs/test.jsonl" }
  ]
}
```

Ingestion requirements:
- **Tolerate missing keys.** Not every config emits every scalar (distances need
  a reference; `quantum_*` need `materialize_matrices`; `excess_loss` needs
  `reference_loss`; etc.). Missing → that trace is simply absent, no error.
- Parse `<scalar>/se` sidecar keys → drive the uncertainty bands (§4c). They are
  **not** separate traces.
- Parse `health/*` keys (`constant_window`, `contracted_*_imag_ratio`, `rank`,
  `use_gram`, …) → drive validity shading (§4e). Not plotted as traces by default.
- **Per-group keys** (`group_shc/<label>`, `group_pr/<label>`,
  `group_total_variance/<label>`) → surface each group as a toggle-able sub-series
  *nested under its parent run*. This is a second form of "probe-set subset."
- The top reference panel reads an **optional separate metrics file**
  (`step`, `loss`, `acc`, …). If none is supplied, fall back to
  `total_first_cumulant` (mean probe loss) as the reference. Keep this source
  decoupled from the observables — training loss/acc does not live in the
  observables log.

---

## 2. Scope — scalar-only

**In scope (scalar keys — verify against catalog):**
`total_first_cumulant`, `total_second_cumulant`, `spectral_heat_capacity`,
`participation_ratio`, `spectral_gap`, `top_eigenvalue_fraction`,
`von_neumann_entropy`, `effective_rank`, `total_third_cumulant`,
`total_fourth_cumulant`, `total_cumulant_5/6/…` (diagnostic),
`classical_fidelity`, `trace_distance`, `spectral_drift`, `quantum_fidelity`,
`quantum_trace_distance`, `cka_similarity`, `cka_distance`, `excess_loss`, the
per-group `group_*` scalars, and `<scalar>/se`.

**Out of scope (do not build views for):** `covariance_spectrum`,
`principal_third_cumulant`, `principal_fourth_cumulant`, `wick_breakdown_ratio`,
`inverse_participation_ratio`, `block_energy`, `contracted_*_spectrum`, and the
`n×n` matrices (`covariance`, `contracted_*_matrix`). No heatmaps, no mode
ribbons, no mode-matching. (An optional hook to *reduce* a per-mode vector to a
labeled scalar is described in §4j but is off by default.)

---

## 3. Layout

- **Top, pinned:** the reference panel — training `loss` and `acc` (twin y-axes),
  from the metrics source. Always visible; sets the x-domain.
- **Below:** a vertical stack of scalar panels, **all sharing one x-axis (step)**.
  Group panels by family with light section separators:
  1. *Detector & scale* — `spectral_heat_capacity`, `total_second_cumulant`,
     `spectral_gap`.
  2. *Concentration* — `participation_ratio` (see collapse group).
  3. *Non-Gaussianity* — `total_fourth_cumulant`, `total_third_cumulant`.
  4. *Distances* — `trace_distance`, `spectral_drift`, `cka_distance` (when present).
  5. *Misc* — `excess_loss` (when present).
- **Color = run** (consistent across every panel); a run keeps one color
  everywhere, so overlaying two probe sets on one observable is two colored lines.
  Panel = observable. Per-group series use the parent run's color with a distinct
  dash.
- **X-axes are locked:** zoom/pan/brush on any panel applies to all panels and
  the reference curve simultaneously.

---

## 4. Features

**a. Non-redundant default set + principled collapse.**
Show one representative per redundancy axis by default; let the user expand a
group to **overlay its members (to verify they agree) or collapse to the
representative**. Default groups (verify keys/semantics against `concepts.md`):

| Group | Members | Representative | Note |
|---|---|---|---|
| Concentration | `participation_ratio`, `effective_rank`, `von_neumann_entropy`, `top_eigenvalue_fraction` | `participation_ratio` | `effective_rank == exp(von_neumann_entropy)` **exactly** |
| Distance — shape | `trace_distance`, `classical_fidelity` | `trace_distance` | plot fidelity as `1 − fidelity` for comparability |
| Distance — subspace | `cka_distance`, `quantum_trace_distance`, `cka_similarity`, `quantum_fidelity` | `cka_distance` | plot similarities/fidelities as `1 − x` |

Singletons (no collapse): `spectral_heat_capacity`, `total_second_cumulant`,
`spectral_gap`, `total_third_cumulant`, `total_fourth_cumulant`,
`spectral_drift`, `excess_loss`. Make the group registry a small config object so
it's easy to edit.

**b. Probe-set / run toggles.** A sidebar (or legend) to show/hide each loaded
run globally, and to expand a run to toggle its per-group sub-series. Toggling a
run hides its traces, bands, and validity spans in *every* panel at once.

**c. Uncertainty band behind each trace — highest-value feature.** Shade
`value ± k·se` from the `<scalar>/se` keys (default k=1, adjustable). Architect
the band source as a **pluggable "band provider"** interface
(`(runId, key, step) → {lo, hi} | null`) so a future RMT / Marchenko–Pastur–edge
band can replace the jackknife SE without touching the plotting code. A trace
whose movement stays inside its band is not an event — make the band visually
clearly behind the line.

**d. Config provenance + compatibility guard.** Parse each run's config
(`CumulantConfig`) and display it (a per-run info popover). When runs are
overlaid, **warn on incompatible configs**, because the lines are silently
non-comparable:
- different `ddof` → SHC scales as `D⁻²`, not comparable;
- `cumulant_estimator` `biased` vs `unbiased` → different quantities;
- different `detrend_order` → different baseline (and unbiased-estimator
  guarantee differs);
- different `shc_mode` (`rank` vs `full`) → different SHC;
- different `window_size` → different buffer-fill region and SHC magnitude.
Surface these as a dismissible banner on the affected panels, not a hard block.

**e. Validity shading.** Behind each run's traces, shade unreliable regions:
- **buffer-fill:** `step < window_size` (from config) or before the first
  non-NaN row;
- **dead window:** `constant_window == 1.0`;
- **(optional)** high `contracted_*_imag_ratio` (> `imag_ratio_warn`, default
  1e-3) — only relevant if a contracted-derived scalar is shown.
Also add a static tooltip note on `participation_ratio` / `effective_rank` that
for the dynamical source they are **capped by the window** (`rank ≤ m−1`), so
their level partly reflects window length, not just the model.

**f. Log-y toggle + normalize/z-score toggle.** Per-panel (and a global) log-y,
because SHC and eigenvalue-derived scalars span decades. A normalize/z-score
toggle so members of a collapse group can be overlaid on one scale to *visually
confirm* they move together before collapsing.

**g. Linked crosshair + synchronized zoom.** A single vertical crosshair tracking
mouse-x across the whole stack (shared step). A hover readout shows **every
visible trace's value at that step** at once. Brush-to-zoom on any panel zooms all.

**h. Event markers.** Vertical lines with full CRUD. Each marker has a `label`, a
`step`, and a `category` (`curriculum` / `detected` / `manual`) rendered with a
**distinct style per category**, kept in a layer separate from any auto-detected
ones. **Import/export markers as JSON** so they persist across sessions and can be
reused across runs. (Auto-detection of events from null-band crossings is a
future extension — leave a seam, don't build it now.)

**i. Click-to-set-reference for distances — gated.** The distance scalars are
computed by the monitor against a fixed reference (`previous`/`baseline`/`lag`).
Recomputing them against a user-clicked step requires the **per-measurement
spectra**, which are vectors. So:
- If `covariance_spectrum` is present in the log, enable "click a step → recompute
  the *classical* distances (`trace_distance`, `spectral_drift`,
  `classical_fidelity`) client-side against that step."
- If it is absent, **disable** this feature with a tooltip explaining why.
Do **not** attempt to recompute quantum/CKA distances client-side. State this
contingency clearly in the UI.

**j. Per-mode → scalar reduction hook (optional, off by default).** Provide a
mechanism to add a *labeled scalar* derived from a per-mode vector (e.g.
`wick_breakdown_ratio` top mode or mean-over-modes). If used: the reduction
(`index k` or `mean`) **must appear in the trace name**, and the tooltip **must
carry the mode-crossing caveat** — a "top mode" reduction silently switches modes
at transitions, so a discontinuity in `wick_breakdown_ratio[0]` can be a mode swap
rather than a real change. Keep this behind a clearly-labeled advanced toggle so
the default tool stays strictly scalar.

---

## 5. Tech stack

**Default recommendation:** a local single-page app — **Vite + React +
TypeScript**, plots rendered with **uPlot**. Rationale: many synchronized scalar
time series with a linked crosshair and shaded bands is exactly uPlot's strength
(fast, small, native cursor-sync), while React cleanly handles the stateful UI
(run toggles, marker CRUD + persistence, collapse groups, config guard, reference
selection). Runs entirely locally against local files; no server, no deployment.

**Fallback (state it and use it if the user prefers Python / the repo already has
a Python-frontend toolchain):** a **Bokeh** app — it has native linked crosshairs,
`Span` for markers/validity shading, and `varea` for bands, and stays in the
`cumobs` ecosystem.

Follow the repo's **`frontend-design` skill** for visual styling (design tokens,
typography, spacing) if building the React version. If the repo dictates a
different stack, prefer it and note the deviation; otherwise default to the above
without asking.

---

## 6. Architecture

Keep it modular so the deliberately-excluded pieces (RMT null band, vector views)
can slot in later without a rewrite:

- **`dataLoader`** — parse manifest + JSONL (+ metrics), normalize to a per-run
  `{ steps, series: {key → values}, se: {key → values}, health, config }`.
  Tolerant of missing keys.
- **`runStore`** — loaded runs, visibility, colors, per-group sub-series.
- **`collapseRegistry`** — the redundancy groups (§4a), editable config.
- **`bandProvider`** — pluggable interface (§4c); ship the jackknife-SE provider,
  leave the RMT provider as a stub.
- **`configGuard`** — pairwise compatibility check (§4d) → warnings.
- **`validitySpans`** — derive shaded regions from health + config (§4e).
- **`markerStore`** — CRUD + JSON import/export + category styles (§4h).
- **`panelGrid`** — the stacked uPlot panels with a shared x-scale + a single
  **`crosshairController`** syncing cursor and zoom across all panels + reference.
- **`referencePanel`** — the pinned loss/acc chart.

Ship a **synthetic JSONL generator / fixture** (a fake run with an injected
transition — rising SHC, a `total_fourth_cumulant` spike, a `constant_window`
stretch, some `/se` keys, two probe-set runs) so the app can be developed and
tested with no training run. Add unit tests for `dataLoader`, `collapseRegistry`,
`configGuard`, and `validitySpans`. Write a short `README` for running it.

---

## 7. Definition of done

- Loads N runs from a manifest (and via drag-drop); tolerates missing keys.
- Reference loss/acc panel pinned on top; falls back to `total_first_cumulant`.
- Stacked scalar panels with a **locked shared x-axis**; zoom/pan/brush
  synchronized across all panels and the reference.
- Toggling a run hides its traces, bands, and spans everywhere; per-group
  sub-series toggle under their run.
- Collapse groups render one representative by default; expand → overlay members
  or collapse to representative; z-score toggle overlays them on one scale.
- `<scalar>/se` bands render when present, via the pluggable band provider.
- Validity spans render for buffer-fill and dead windows; PR/effective-rank carry
  the window-cap note.
- Config guard warns on incompatible overlays (ddof / estimator / detrend_order /
  shc_mode / window_size).
- Linked crosshair shows every visible trace's value at the hovered step; log-y
  and normalize toggles work.
- Markers: create/edit/delete, categorized and styled, **persist via JSON
  export/import**.
- Click-to-set-reference for classical distances is enabled iff
  `covariance_spectrum` is logged, and clearly disabled otherwise.

---

## 8. Non-goals

- No vector/matrix views of any kind (spectra, per-mode traces, heatmaps,
  ribbons, mode matching).
- No server, no deployment, no auth — local analysis tool only.
- No writing back to or interfering with the training run.
- No recomputation of observables that require the raw loss buffers (the only
  client-side recomputation permitted is classical distances *when the spectra
  are logged*, §4i).
- No auto event-detection in v1 (leave the seam only).

---

## 9. Suggested build order (runnable early)

1. **Phase 1 — minimum usable tool:** `dataLoader` + `runStore` + reference panel
   + stacked scalar line panels + locked shared x-axis + linked crosshair. At the
   end of this phase you can already load runs and read the scalars.
2. **Phase 2 — comparison:** run/probe-set toggles, per-group sub-series, collapse
   groups, log-y + normalize toggles, event markers with persistence.
3. **Phase 3 — significance & trust:** SE bands (pluggable provider), validity
   shading, config-compatibility guard.
4. **Phase 4 — optional:** gated click-to-set-reference; the per-mode reduction
   hook (advanced toggle, off by default).
