# cumobs — cumulant observables for deep-learning phase transitions

A lean, framework-free training monitor that turns the cumulant-expansion picture
from *"From Density Matrices to Phase Transitions in Deep Learning"* (Hennick &
Corlouer, 2026) into a drop-in observable stack. It tracks scalar cumulants up to
4th order plus spectral diagnostics of the per-sample loss covariance (the
"2RDM") over a sliding window, needing only a forward pass on a probe set.

The whole scalar + spectral observable set is computed in **window space**
(`m × m`) via the Gram trick — no `n × n` matrix is ever allocated unless you
explicitly ask for it (`materialize_matrices=True`). Only dependency is `torch`.

See [`cumulant_observables_spec.md`](cumulant_observables_spec.md) for the full
mathematical specification and design decisions.

## Documentation

Human- and agent-readable docs live in [`docs/`](docs/) — start at
[`docs/index.md`](docs/index.md):

- [getting-started.md](docs/getting-started.md) — run it in a training loop (or
  offline) in a few lines.
- [concepts.md](docs/concepts.md) — the math: cumulants, the 2RDM, the Gram
  trick, window space.
- [observables.md](docs/observables.md) — per-key reference of every output a
  `Measurement` can hold.
- [configuration.md](docs/configuration.md) — every `CumulantConfig` field, the
  D1–D6 decisions, the cost model.
- [api-reference.md](docs/api-reference.md) — module-by-module API.
- [logging.md](docs/logging.md) — the flattening policy and logger backends.
- [extending.md](docs/extending.md) — the §9 extension seams.

Runnable examples are in [`examples/`](examples/) (see
[examples/README.md](examples/README.md)).

## Install

The package lives in `cumobs/` and needs only `torch` (already in this project's
environment). Run the tests with:

```bash
python -m pytest tests -q
```

## Quick start (Algorithm 1 of the paper)

```python
from cumobs import CumulantConfig, CumulantMonitor, ProbeSet, JSONLLogger

probes = ProbeSet(inputs=probe_inputs, loss_fn=my_loss,
                  group_labels=labels, name="mix")
cfg = CumulantConfig(window_size=20, update_interval=10, top_k=3,
                     compute_eigenvectors=True, per_group=True)
monitor = CumulantMonitor(model, probes, cfg, logger=JSONLLogger("run.jsonl"))

for step in range(num_steps):
    train_step(...)
    monitor.step(step)        # evaluates probes / appends / computes / logs on schedule
```

`monitor.step` returns the `Measurement` on a computation step and `None`
otherwise; it logs internally. `loss_fn(model, batch) -> Tensor[(b,)]` is
user-supplied — the package makes no assumption about your loss API.

## What it computes (§3 of the spec)

| Order | Observables |
|-------|-------------|
| 1 | `first_cumulant_vector`, `total_first_cumulant` |
| 2 | `covariance_spectrum`, `spectral_heat_capacity`, `participation_ratio`, `spectral_gap`, `top_eigenvalue_fraction`, `von_neumann_entropy`, `effective_rank`, `normalized_shc`, `log_det`, `inverse_participation_ratio` (needs `V`) |
| 3 | `total_third_cumulant`, `diagonal_third_cumulant`, `principal_third_cumulant`, `contracted_third_cumulant_spectrum` |
| 4 | `total_fourth_cumulant`, `diagonal_fourth_cumulant`, `principal_fourth_cumulant`, `wick_breakdown_ratio`, `contracted_fourth_cumulant_spectrum` |
| 5+ (testing) | `total_cumulant_5`, `total_cumulant_6`, … — general scalar cumulants of the aggregate (`max_scalar_cumulant_order`) |
| standardized (v0.3 §1) | `standardized_cumulant_spectrum`, `principal_standardized_cumulant`, `leading_fisher_zero_*`, `fisher_zero_gap` |
| CSD / alignment (v0.3 §5,§7) | `autocorr_lag1`, `autocorr_time`, `align_cov_skew`, `align_cov_kurtosis`, `subspace_align_*` |
| null model / BBP (v0.3 §3,§6) | `<scalar>/null_z`, `mp_edge_upper`, `n_spikes`, `active_probe_mask` (opt-in `null_model`) |
| distances | `classical_fidelity`, `trace_distance`, `spectral_drift`, `fisher_rao_distance`, `quantum_fidelity`, `quantum_trace_distance`, `bures_distance`, `bures_angle`, `cka_similarity`, `cka_distance` |
| per-group | `group_total_variance`, `group_shc`, `group_pr`, `block_energy` |

The **v0.3 extension batch** (see [`cumobs_extension_spec.md`](../cumobs_extension_spec.md))
adds the standardized cumulant spectrum + Fisher/Lee–Yang zero, geodesic +
noise-floored distances, a surrogate + Marchenko–Pastur null model, CSD
autocorrelation, a BBP participation bitmask, and cross-operator alignment, plus
eigenvector tracking. All additive; the one behaviour change is the divisor
`D = m − (ddof + detrend_order)` (R1).

The named 3rd/4th cumulants are raw cumulants `κ₃`/`κ₄` (not standardized
coefficients) and default to the **unbiased** k-statistic estimator
(`cumulant_estimator`). The opt-in `n × n` matrices `covariance` (`C`),
`contracted_third_cumulant_matrix` (`M3`), and `contracted_fourth_cumulant_matrix`
(`K`) are built only when `materialize_matrices=True`.

## Key configuration (`CumulantConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `window_size` | 20 | rows `m` of the sliding buffer |
| `ddof` | 1 | the *single* normalization divisor `D = m − (ddof + detrend_order)` (decision D1 / R1) |
| `cumulant_estimator` | `"unbiased"` | `"unbiased"` (k-statistics) or `"biased"` (plug-in) for the 3rd/4th cumulants |
| `order` | 4 | max cumulant order (2, 3, or 4) |
| `detrend_order` | 0 | within-window polynomial detrend (`1` strips a linear trajectory drift) |
| `top_k` | 3 | principal directions for projection cumulants / IPR / CKA |
| `shc_mode` | `"rank"` | SHC over nonzero eigenvalues (`"rank"`) or the padded spectrum (`"full"`) |
| `materialize_matrices` | False | build the opt-in `n × n` matrices |
| `compute_eigenvectors` | False | build sample-space `V` (IPR, block energy, quantum + CKA distances) |
| `per_group` | False | per-group decomposition (§3.7) |
| `distance_flavor` | `"classical"` | `"classical"`, `"quantum"`, or `"both"` (decision D2) |
| `distance_reference` | `"previous"` | `"previous"`, `"baseline"`, or `"lag:k"` (decision D4) |
| `distance_source` | `"covariance"` | spectrum feeding the distances (decision D3) |
| `compute_cka_distance` | False | emit `cka_*` representation-drift distance (needs `compute_eigenvectors`) |
| `error_bars` | `"off"` | `"jackknife"` emits `<scalar>/se` (costs `m`× a measurement) |
| `dtype` | `float64` | recommended for the asymmetric contracted eigs |

### Decisions baked in (§0 of the spec)

- **D1** one divisor `D = m − (ddof + detrend_order)` everywhere (R1); SHC
  scales as `D⁻²`, so fix `ddof` *and* `detrend_order` across runs.
- **D2** both classical (spectrum-only, eigenvector-blind) and quantum
  (density-matrix, eigenvector-sensitive) distances ship. Quantum needs
  `materialize_matrices=True`.
- **D3** distances default to the covariance spectrum; the contracted 3rd/4th
  cumulant spectra are selectable (signed — the classical distances warn there).
- **D4** reference defaults to `"previous"`; `"baseline"` and `"lag:k"` are
  available. (A transition-aware baseline-swap is a documented TODO.)
- **D5** the Gaussian 2RDM source is a seam only — `GaussianSource.collect`
  raises `NotImplementedError`.
- **D6** eigenvector signs are gauged so the max-magnitude entry is positive.

## Example: the Fig. 1 synthetic transition

```bash
python examples/fig1_transition.py
```

Injects `ℓ(t) = a(t)·v₁ + ε` with `v₁` planted on a single probe group. The
spectral heat capacity spikes when the mode turns on and the top eigenvector
localizes on the planted group (high block energy) — the behavioral acceptance
check (§10 test 12).

## Extension seams (defined, not built — §9)

Activation/representation cumulants (injectable `contraction` direction),
blended Gaussian↔dynamical sampling, quantum distances on stored matrices, and a
trajectory-LLC byproduct (`reference_loss` hook) are all wired as seams.
