# API reference

Module-by-module reference for the public surface (everything re-exported from
`cumobs/__init__.py`). Signatures are abbreviated; see the source for full type
hints and docstrings. Section numbers point to
[`cumulant_observables_spec.md`](../cumulant_observables_spec.md).

```python
from cumobs import (
    CumulantConfig, LoggingConfig,                 # config
    Measurement, SharedIntermediates,              # types
    build_intermediates, compute_all,              # observable core
    scalar_cumulants, spectral, contracted, per_group,
    derived_cumulants, autocorrelation, alignment, null_model, bbp_bitmask,
    total_scalar_cumulants, moments_from_cumulants, factorization_residual,
    leading_fisher_zero, integrated_autocorr_time,
    classical_fidelity, trace_distance, spectral_drift,   # distances
    fisher_rao_distance, quantum_fidelity, quantum_trace_distance,
    bures_distance, bures_angle,
    denoised_spectral_drift, spectral_drift_snr, signal_energy_drift,
    cka_similarity, cka_distance,
    SlidingWindow, BufferSource,                   # window
    DynamicalSource, GaussianSource,
    ProbeSet, ProbeSetCollection,                  # probes
    Logger, InMemoryLogger, JSONLLogger,           # logging
    StdoutLogger, WandbLogger, TensorBoardLogger,
    CumulantMonitor,                               # monitor
)
```

---

## `config.py`

### `CumulantConfig`
Dataclass of every knob. Full table + guidance in
[configuration.md](configuration.md). Validates in `__post_init__`. Helpers:
`divisor` (property → `window_size − ddof`), `lag()` (parse `"lag:k"`).

### `LoggingConfig`
`vector_policy` / `vector_overrides` / `log_health`. `policy_for(key)` returns
the effective policy for a key.

---

## `types.py`

### `SharedIntermediates`
The §4 reuse bundle, built once per measurement by `build_intermediates`. Holds
`m, n, D, use_gram, rank, top_k_eff`, the shared blocks `E_L, B_c, S, S2, S3,
S4, s2`, the spectral objects `eigvals, lam_top, P, V`, the kurtosis operator
`Q`, the optional `G0/C/M3/K`, and a `health` dict. You rarely build this
directly — `compute_all` does. Useful when you want an intermediate (e.g. `si.C`,
`si.V`) without re-deriving it.

### `Measurement`
```python
@dataclass
class Measurement:
    step: int
    scalars:  dict[str, float]
    vectors:  dict[str, Tensor]
    matrices: dict[str, Tensor]
    health:   dict[str, float]
    factors:  dict[str, Tensor]    # {"V", "lam_top"} for CKA; never auto-logged
    aux:      dict[str, Any]       # non-float observables (BBP mask hex strings)
    def to_flat_dict(vector_policy="summary",
                     vector_overrides=None,
                     log_health=True,
                     drop_keys=(), drop_prefixes=()) -> dict[str, Any]: ...
```
The structured result. `to_flat_dict` applies the logging policy (§7): scalars
pass through; vectors become `key/max`, `key/mean`, `key/top1`, `key/l2` under
`"summary"`, the raw array under `"full"`, or are dropped under `"off"`; `aux`
values (strings/ints/lists, e.g. the BBP mask) pass through verbatim; health goes
to `health/*`; matrices are never auto-flattened. `drop_keys` / `drop_prefixes`
suppress keys (the §8 consolidation, via `LoggingConfig.trim_consolidated`). See
[observables.md](observables.md) for every key.

---

## `window.py`

### `SlidingWindow(n_features, window_size, device="cpu", dtype=float64)`
Fixed-capacity ring buffer of loss vectors.
- `append(loss_vector: Tensor)` — add a `(n,)` row, overwriting the oldest when
  full (raises if the length mismatches `n_features`).
- `is_full` (property), `count` (property: rows available, capped at
  `window_size`).
- `as_tensor() -> Tensor` — the buffered rows `(m, n)` ordered **oldest →
  newest** (a clone).
- `reset()` — clear.

### `BufferSource` (Protocol)
`collect(model, probeset, step) -> Tensor | None` — return the current `(m, n)`
buffer or `None` if not ready. The monitor polls this each eval step.

### `DynamicalSource(window_size, dtype=float64, device=None)`
The dynamical 2RDM. Lazily sizes its window from the first probe evaluation
(learns `n`), appends the per-step loss, and returns `(m, n)` once full. Exposes
`window` (property) and `reset()`.

### `GaussianSource(sigma, n_samples)`
The Gaussian 2RDM **seam** (decision D5). `collect` raises
`NotImplementedError`. The intended behavior is documented in the docstring:
draw `k` i.i.d. `N(0, σ²)` parameter perturbations, evaluate the probe loss
under each, restore parameters, return the `(k, n)` buffer immediately (no
windowing). See [extending.md](extending.md).

---

## `probes.py`

### `ProbeSet(inputs, loss_fn, group_labels=None, batch_size=64, name="probe")`
A named, optionally grouped collection of loss probes.
- `loss_fn(model, batch) -> Tensor[(b,)]` — **one scalar loss per input
  sample**, user-supplied.
- `evaluate(model) -> Tensor[(n,)]` — mini-batches `inputs` through `loss_fn`
  under `torch.no_grad()` and concatenates (raises if the total length ≠ `n`).
- `groups() -> dict[label, LongTensor]` — column indices per group label (empty
  if no labels). `group_labels` length must equal `len(inputs)`.
- `__len__` → `n`.

### `ProbeSetCollection(probesets)`
Several independently-windowed named `ProbeSet`s (a list — names must be unique —
or a `{name: ProbeSet}` dict). Supports `len`, iteration, `items()`, `names()`,
`coll[name]`.

---

## `observables.py`

The pure-tensor math core (no model, no IO). Section functions read a
`SharedIntermediates`; `compute_all` is the top-level entry point.

### `build_intermediates(B, cfg, contraction=None) -> SharedIntermediates`
Runs the shared linear algebra once in §4 order. `B` is `(m, n)`; `contraction`
is an optional `(n,)` direction for the aggregate `S = B_c @ contraction`
(default all-ones, §9.1 seam). Raises if `D = m − ddof ≤ 0` ("Is the window
full?").

### `compute_all(B, cfg, groups=None, step=0, contraction=None, reference_loss=None) -> Measurement`
Computes the full catalog for buffer `B`. `groups` maps a label to a
`LongTensor` of column indices (from `ProbeSet.groups()`). `reference_loss`
(optional `(n,)`) enables the `excess_loss` byproduct (§9.4). With
`cfg.error_bars="jackknife"` every scalar also gets a `<scalar>/se`. **Distances
are not computed here** (they need a reference measurement); the monitor adds
them.

### `total_scalar_cumulants(series, D, max_order) -> dict[int, float]`
General scalar cumulants `κ₁…κ_max_order` of **any** 1-D `series` via the
(plug-in) moment–cumulant recurrence (§3.8), using the single divisor `D` for
every raw moment. Returns `{p: κ_p}`. Reproduces the plug-in
`total_third_cumulant`/`total_fourth_cumulant` when fed the aggregate `S`;
`compute_all` calls it for orders ≥ 5 when `cfg.max_scalar_cumulant_order > 4`.
High orders are numerically delicate — use `float64`.

### Section functions (advanced / partial computation)
| Function | Returns |
|----------|---------|
| `scalar_cumulants(si, cfg)` | `(scalars, vectors)` — first/total/diagonal/principal cumulants up to `cfg.order` (+ `total_cumulant_{5..}` if `max_scalar_cumulant_order > 4`) |
| `spectral(si, cfg)` | `(scalars, vectors, matrices)` — covariance spectrum, SHC, PR, gap, entropy, effective rank, `normalized_shc`, `log_det`, IPR, `C` |
| `derived_cumulants(si, cfg)` | `(scalars, vectors)` — `standardized_cumulant_spectrum`, `principal_standardized_cumulant`, `principal_cumulant_{p}`, the Fisher-zero scalars |
| `contracted(si, cfg)` | `(vectors, health, matrices)` — contracted 3rd/4th cumulant spectra (+ imag-ratio health) |
| `autocorrelation(si, cfg)` | `(scalars, health, τ_int)` — `autocorr_lag1`/`autocorr_time` + `effective_window` |
| `alignment(si, cfg)` | `(scalars, vectors)` — `align_cov_skew`/`align_cov_kurtosis`, `subspace_align_*` |
| `null_model(si, cfg, τ_int, observed, contraction=None)` | `(scalars, vectors, health)` — surrogate `<scalar>/null_*` and/or MP/TW edge |
| `bbp_bitmask(si, cfg, mp_edge_upper)` | `(scalars, vectors, aux)` — `active_probe_mask*`, `active_probe_count` |
| `per_group(si, cfg, groups)` | `(scalars, vectors)` — group variance, group SHC/PR, block energy |

Use these only if you want to compute a subset directly; otherwise call
`compute_all`.

### Helper functions
| Function | Computes |
|----------|----------|
| `moments_from_cumulants(kappa, max_order)` | raw moments `m_0..m_max_order` of a centered variable from its cumulants (inverse recurrence) |
| `factorization_residual(cumulants, M, N)` | §1b residual `R(M,N)` of the M-point function after reconstruction into blocks `≤ N` |
| `leading_fisher_zero(kappa, root_order, eps)` | `(complex t, gap)` of the leading Fisher/Lee–Yang zero, or `None` for a dead window |
| `integrated_autocorr_time(x, c=5.0)` | `(τ_int, ρ₁)` of a 1-D series via Sokal automatic windowing |

---

## `distances.py`

All return Python floats; degenerate inputs (zero total mass) return `NaN`.

| Function | Computes | Input |
|----------|----------|-------|
| `classical_fidelity(p, q, eps=1e-8)` | `(Σ√(p_i q_i))²` ∈ `[0,1]` | raw spectra (clipped + normalized internally; warns on signed input) |
| `trace_distance(p, q, eps=1e-8)` | `½ Σ|p_i − q_i|` ∈ `[0,1]` | raw spectra |
| `spectral_drift(p, q, eps=1e-8)` | `‖p−q‖₂/max(‖p‖₂,‖q‖₂)` — scale-sensitive | raw spectra (un-normalized) |
| `fisher_rao_distance(p, q, eps=1e-8)` | `2·arccos(√classical_fidelity)` ∈ `[0,π]` | raw spectra (inherits the signed-source warning) |
| `quantum_fidelity(rho, sigma, eps=1e-8)` | Uhlmann `(tr√(√ρ σ √ρ))²` | raw covariance matrices (symmetrized + trace-normalized) |
| `quantum_trace_distance(rho, sigma, eps=1e-8)` | `½‖ρ − σ‖₁` | raw covariance matrices |
| `bures_distance(rho, sigma, eps=1e-8)` | `√(2(1 − √quantum_fidelity))` ∈ `[0,√2]` | raw covariance matrices |
| `bures_angle(rho, sigma, eps=1e-8)` | `arccos(√quantum_fidelity)` ∈ `[0,π/2]` | raw covariance matrices |
| `denoised_spectral_drift(drift, μ₀, eps=1e-8)` | `max(0,(drift−μ₀)/(1−μ₀))` | scalars (drift + surrogate null mean) |
| `spectral_drift_snr(drift, μ₀, s₀, eps=1e-8)` | `(drift−μ₀)/s₀` | scalars |
| `signal_energy_drift(p, q, edge, eps=1e-8)` | `spectral_drift` on `max(0, λ−edge)` | raw spectra + MP edge |
| `cka_similarity(V1, lam1, V2, lam2, eps=1e-8)` | linear CKA ∈ `[0,1]` | top-`k` eigen-factors (`V` `(n,k)`, `lam` `(k,)`) |
| `cka_distance(V1, lam1, V2, lam2, eps=1e-8)` | `1 − cka_similarity` | same |

Classical functions zero-pad the shorter spectrum to align lengths and are
scale-blind; `spectral_drift` is the scale-sensitive companion. Quantum and CKA
functions are eigenvector-sensitive (`O(n³)` and `O(n·k²)` respectively). See
[concepts.md §6](concepts.md#6-distances-between-measurements-decision-d2-d4).

---

## `logging.py`

`Logger` protocol: `log(metrics: Mapping[str, float | Tensor], step: int)`.
Backends: `InMemoryLogger`, `JSONLLogger`, `StdoutLogger`, `WandbLogger`,
`TensorBoardLogger`. Full reference in [logging.md](logging.md).

---

## `monitor.py`

### `CumulantMonitor(model, probes, cfg, logger=None, source=None, logging_config=None, contraction=None, reference_loss=None)`
The drop-in orchestrator.
- `probes` — a `ProbeSet` or `ProbeSetCollection`. Each probe gets its own
  `DynamicalSource` (independently windowed) unless you pass a custom single
  `source` for a single `ProbeSet`.
- `contraction` / `reference_loss` are forwarded to `compute_all`.
- **`step(global_step) -> Measurement | dict[str, Measurement] | None`** — the
  one method you call each training step. On eval steps it polls each source; on
  update steps with a full window it runs `compute_all`, computes distances
  against the retained reference (decision D4), records the new reference, and
  logs the flattened result. Returns the `Measurement` (single probe), a
  `{name: Measurement}` dict (collection), or `None` (nothing to compute this
  step). Logging happens internally.

The monitor retains a per-probe reference **history** of spectra (cheap
`n`-vectors) and, if quantum distances are on, matrices — used by
`_reference_for` to resolve `"previous"` / `"baseline"` / `"lag:k"`. With
`track_modes=True` it additionally keeps the last canonical eigenvector matrix
and a ring buffer of tracked per-mode eigenvalues, used to emit
`mode_permutation` / `crossing_detected` and the per-mode CSD series
`principal_autocorr_lag1` / `principal_autocorr_time`. The geodesic distances
(`fisher_rao_distance`, `bures_*`) and the noise-floored distances are wired
automatically alongside their parent fidelities / when `null_model` is on.
