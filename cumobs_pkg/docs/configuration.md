# Configuration

Everything is driven by one dataclass, [`CumulantConfig`](../cumobs/config.py),
plus a small [`LoggingConfig`](../cumobs/config.py) for the flattening policy.

```python
from cumobs import CumulantConfig
cfg = CumulantConfig(window_size=20, top_k=3, compute_eigenvectors=True)
```

---

## `CumulantConfig` fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `window_size` | int | `20` | rows `m` of the sliding buffer `B` |
| `eval_interval` | int | `10` | steps between probe evaluations / window appends |
| `update_interval` | int | `10` | steps between observable computations; keep `< window_size` |
| `ddof` | int | `1` | delta-DOF for the single divisor `D = window_size − (ddof + detrend_order)` (decision D1 / R1) |
| `cumulant_estimator` | str | `"unbiased"` | `"unbiased"` (Fisher k-statistics) \| `"biased"` (plug-in) for the named 3rd/4th cumulants |
| `order` | int | `4` | max cumulant order to compute: `2`, `3`, or `4` |
| `max_scalar_cumulant_order` | int | `4` | if `> 4` (and `order = 4`), also emit `total_cumulant_{5..max}` — general scalar cumulants of the aggregate `S` (§3.8) |
| `detrend_order` | int | `0` | degree of the within-window polynomial (in row/time index) removed before `B_c`; `0` = mean-centering, `1` = strip linear drift |
| `top_k` | int | `3` | principal directions for projection cumulants / IPR / block energy / CKA |
| `eig_threshold` | float | `1e-7` | relative cutoff for a "nonzero" eigenvalue (× `μ_max`); also defines the `rank` for `shc_mode="rank"` |
| `eps` | float | `1e-13` | denominator / sqrt guard |
| `shc_mode` | str | `"rank"` | `"rank"` (variance over nonzero eigenvalues, `n`-invariant) \| `"full"` (legacy variance over the padded spectrum) |
| `materialize_matrices` | bool | `False` | build the opt-in `n × n` matrices `C`, `M3`, `K` |
| `compute_eigenvectors` | bool | `False` | build sample-space `V` (IPR, block energy, quantum + CKA distances) |
| `per_group` | bool | `False` | compute the per-group observables (needs `group_labels`) |
| `distance_reference` | str | `"previous"` | `"previous"` \| `"baseline"` \| `"lag:k"` (decision D4) |
| `distance_source` | str | `"covariance"` | `"covariance"` \| `"contracted_third_cumulant"` \| `"contracted_fourth_cumulant"` (D3) |
| `distance_flavor` | str | `"classical"` | `"classical"` \| `"quantum"` \| `"both"` (decision D2) |
| `compute_cka_distance` | bool | `False` | also emit `cka_similarity` / `cka_distance` (needs `compute_eigenvectors`) |
| `error_bars` | str | `"off"` | `"off"` \| `"jackknife"` (emits `<scalar>/se`; `m`× cost) |
| `dtype` | torch.dtype | `torch.float64` | working dtype; float64 recommended for the asymmetric eigs |
| `sign_gauge` | bool | `True` | deterministic eigenvector sign (decision D6) |
| `imag_ratio_warn` | float | `1e-3` | warn threshold on `max|Im|/(max|Re|+eps)` for contracted spectra |
| `root_order` | int \| None | `None` | truncation order of the Fisher/Lee–Yang partition function (§1c); `None` → `min(4, P_tot)` |
| `principal_cumulant_order` | int | `4` | highest per-mode cumulant order for `principal_standardized_cumulant` / `principal_cumulant_{p}` (§1); `>4` adds the one new primitive |
| `null_model` | str | `"off"` | `"off"` \| `"surrogate"` \| `"mp"` \| `"both"` — §3 null background |
| `null_samples` | int | `50` | surrogates per measurement when `null_model` ∈ {surrogate, both} |
| `rho_keep` | float | `0.9` | energy target for the §6 BBP support set |
| `track_modes` | bool | `False` | track eigenvector identity across steps (§7 step 2) → `mode_permutation` / `crossing_detected` / `principal_autocorr_*` (needs `compute_eigenvectors`) |

### Derived helpers

- `cfg.divisor` → `window_size − ddof`, the single `D`.
- `cfg.lag()` → parses `k` from a `"lag:k"` reference (default 1).

### Validation (`__post_init__`)

The config rejects invalid combinations at construction:

- `order ∉ {2, 3, 4}` → `ValueError`.
- `cumulant_estimator ∉ {biased, unbiased}` → `ValueError`.
- `window_size − ddof ≤ 0` (divisor non-positive) → `ValueError`.
- `detrend_order < 0` or `detrend_order + 1 ≥ window_size` (too few rows) → `ValueError`.
- `top_k < 1` → `ValueError`.
- `shc_mode ∉ {rank, full}`, `error_bars ∉ {off, jackknife}` → `ValueError`.
- `compute_cka_distance=True` without `compute_eigenvectors=True` → `ValueError`.
- `distance_flavor`, `distance_source`, `distance_reference` outside their
  allowed sets → `ValueError`.
- `null_model ∉ {off, surrogate, mp, both}`, `null_samples < 1`,
  `rho_keep ∉ (0, 1]`, `principal_cumulant_order < 4`, `root_order` not `None`
  and `< 2` → `ValueError`.
- `track_modes=True` without `compute_eigenvectors=True` → `ValueError`.

---

## How to think about the key knobs

### Window length and schedule (`window_size`, `eval_interval`, `update_interval`)

- `window_size` is your statistical sample size `m`. Larger ⇒ smoother
  estimates but slower to react to a transition. The covariance has rank ≤
  `min(m−1, n)`, so `m` also caps how many modes you can resolve.
- A measurement requires the window to be **full** *and* `step %
  update_interval == 0`. Keep `update_interval < window_size` so consecutive
  measurements don't share the entire window.
- Following the paper's cost guidance, you can gate the expensive eigenvector /
  matrix materialization behind an SHC threshold by toggling config between
  cheap and full measurements (you own the loop, so you can swap the monitor's
  `cfg` or run two monitors).

### Order (`order`)

`2` → covariance + spectral diagnostics only (cheapest, and you may then drop to
`float32` for the symmetric-only path). `3` adds the 3rd cumulant. `4` (default)
adds the connected 4th cumulant and `wick_breakdown_ratio`.

### Estimator: biased vs unbiased

`cumulant_estimator` selects how the named 3rd/4th cumulants
(`total_/diagonal_/principal_*_cumulant`) are estimated:

- **`"unbiased"` (default)** — Fisher **k-statistics**, the unique unbiased
  estimators of `κ₃`/`κ₄`. A genuinely Gaussian window then reads ≈ 0 (up to
  sampling noise), so `total_fourth_cumulant` / `wick_breakdown_ratio` are honest
  non-Gaussianity signals.
- **`"biased"`** — the plug-in central-moment formulas with the single divisor
  `D` (`Ê[S³]`, `Ê[S⁴]−3Ê[S²]²`, …). **Lower variance** but a finite-window
  baseline: for a Gaussian, `E[total_fourth_cumulant] ≈ −c·κ₂²/m` (≈ −260 for
  `κ₂≈24`, `m=20`), which *co-varies with the variance* and so confounds a pure
  variance change with a non-Gaussianity change.

Orders ≥ 5 (`total_cumulant_p`) are plug-in in **both** modes (no closed-form
k-statistic ships beyond 4, matching `scipy.stats.kstat`). The second cumulant is
tied to the divisor `D` in both modes (set `ddof=1` for the unbiased variance,
the default). For small windows the unbiased estimator is noisier per step — pair
it with `error_bars="jackknife"` or prefer `"biased"` if you only track relative
changes.

### SHC normalization (`shc_mode`)

`spectral_heat_capacity` is the variance of the covariance spectrum. `"rank"`
(default) takes it over the `rank` **nonzero** eigenvalues, so it is invariant to
the number of probes and to dead/structural-zero modes — comparable across probe
sets. `"full"` is the legacy variance over the whole zero-padded length-`n`
spectrum, which shrinks as `1/n` as you add probes. Both scale as `D⁻²`. The new
scalars `spectral_gap`, `top_eigenvalue_fraction`, `von_neumann_entropy`, and
`effective_rank` are `n`-independent complements that need no mode choice.

### Detrending (`detrend_order`)

The dynamical 2RDM's rows are *consecutive training steps*, not i.i.d. samples,
so a smooth within-window drift in the loss shows up as a spurious dominant
covariance mode (high SHC, PR ≈ 1) even with no real transition. `detrend_order`
removes the per-probe least-squares polynomial of that degree in the row/time
index before forming `B_c`:

- `0` (default) — plain mean-centering (`B − mean`); the original behaviour and
  the documented "2RDM captures movement along the trajectory" semantics.
- `1` — strip a **linear** drift (recommended when you want genuine fluctuation
  co-movement rather than slow drift). A sharp transition (sigmoid/step) survives
  linear detrending; a smooth ramp does not.

`E_L` (the reported first cumulant) is always the plain mean. Detrending of
degree `d` removes `d+1` degrees of freedom, so set `ddof = d + 1` if you want
the second-order quantities unbiased.

### Error bars (`error_bars`)

`"jackknife"` re-runs the scalar pipeline leave-one-row-out and emits a
`<scalar>/se` standard error for every scalar — the cheapest way to tell whether
a small-window cumulant move is signal or sampling noise. It costs `m`× a normal
measurement, so gate it (e.g. only when SHC crosses a threshold) or run it
offline.

### CKA distance (`compute_cka_distance`)

Emits `cka_similarity` / `cka_distance` between successive measurements — a
linear-CKA representation-similarity on the top-`k` eigen-factors. Unlike the
classical spectrum distances it is **eigenvector-sensitive** (it sees a subspace
rotation), and unlike the quantum distances it is `O(n·k²)` (no `n×n` matrix, no
`O(n³)` eig). Needs `compute_eigenvectors=True`.

### Higher-order scalar cumulants (`max_scalar_cumulant_order`)

`order` caps at 4 because the *spectral / per-feature* cumulants (diagonal,
principal, contracted) get expensive and unwieldy beyond that. But the **scalar
"total" cumulants** of the aggregate `S` cost almost nothing (`O(m · max)` on a
length-`m` series) and generalize to any order via the moment–cumulant
recurrence. Set `max_scalar_cumulant_order = 6` (with the default `order = 4`)
and the measurement gains `total_cumulant_5` and `total_cumulant_6`. Orders 2–4
are already the named totals, so only `≥ 5` add keys. This is meant as a
mathematical check / heavy-tail probe — keep `float64` (high orders cancel
badly). See [observables.md → General higher-order scalar cumulants](observables.md#general-higher-order-scalar-cumulants-testing).

### `top_k`

How many leading modes get principal cumulants, IPR, and block energy. It is
**clamped to the numerical rank** — if the window is rank-deficient you get
fewer entries (recorded in `health["n_principal"]`), never garbage padding.

### v0.3 knobs (null model, tracking, Fisher zero)

- **`null_model`** is off by default. `"surrogate"` re-runs the (stripped) scalar
  pipeline `null_samples`× per measurement to score every scalar against a
  phase-scrambled null (`<scalar>/null_z`) — the principled "could this be
  generic training noise?" detector, answering a *different* question than the
  jackknife SE (distance from the no-transition baseline vs precision of the
  estimate). `"mp"` is cheap (`O(n)` on the spectrum) and adds the
  Marchenko–Pastur bulk edge `mp_edge_upper`, which also gates the §6 BBP bitmask
  and the §4 `signal_energy_drift`. `"both"` runs both. Gate the surrogate cost
  (it is ~`null_samples`× a measurement).
- **`track_modes`** (needs `compute_eigenvectors`) keeps mode identity stable
  across measurements so the per-mode series (`principal_autocorr_*`, the BBP
  bitmask) refer to the same physical mode through crossings. It uses SciPy's
  Hungarian assignment when available, else a greedy fallback.
- **`root_order`** truncates the Fisher-zero partition function; leave it `None`
  (→ `min(4, P_tot)`) unless you have higher aggregate cumulants
  (`max_scalar_cumulant_order > 4`) and want a higher-degree `Z_P`.
- **`principal_cumulant_order`** is the only §1 knob that adds *state*: `> 4`
  computes per-mode cumulants past kurtosis. Default `4` is free.

### Opt-in cost flags

| Flag | Cost added | Unlocks |
|------|-----------|---------|
| `compute_eigenvectors` | `O(mn·top_k)`, allocates `(n, top_k)` | IPR, block energy, subspace alignment, quantum + CKA distances |
| `materialize_matrices` | `O(mn²)`, allocates three `(n, n)` | `covariance`/`M3`/`K`, quantum distances |
| `per_group` | re-runs the (cheap) spectral path per group | `group_*`, block energy decomposition |
| `compute_cka_distance` | `O(n·top_k²)` per pair (from the factors) | `cka_similarity`/`cka_distance` |
| `null_model="surrogate"`/`"both"` | **`null_samples`× the scalar pipeline** | `<scalar>/null_*`, the drift null floor |
| `null_model="mp"` | `O(n)` on the spectrum | `mp_edge_upper`, `n_spikes`, BBP edge |
| `track_modes` | `O(n·top_k²)` match + a per-mode ring buffer | `mode_permutation`, `crossing_detected`, `principal_autocorr_*` |
| `error_bars="jackknife"` | **`m`× the whole scalar pipeline** | `<scalar>/se` standard errors |

`materialize_matrices` is the only flag that introduces an `n²` term — leave it
off for large probe sets unless you need the matrices or quantum distances.
`error_bars` and `null_model="surrogate"` are the most expensive (an extra `m` /
`null_samples` measurements); gate them.

### `dtype`

`float64` (default) is recommended because the **asymmetric** `eig` on the
contracted Gram partners (`G3`, `G4`) is the numerically sensitive step. If you
compute order ≤ 2 only (symmetric covariance path), `float32` is fine.

---

## The design decisions (D1–D6)

These are documented forks from the spec §0, with the resolution baked into the
defaults.

| ID | Decision | Resolution in `cumobs` |
|----|----------|------------------------|
| **D1** | One normalization divisor everywhere | **Yes** — `D = window_size − (ddof + detrend_order)` (R1: detrending of degree `d` is a rank-`(d+1)` projection, so the unbiased divisor adds `detrend_order`; reduces to `window_size − ddof` at `detrend_order = 0`) for every moment and cumulant subtraction. PR / fidelity / TV are `D`-invariant; **SHC scales as `D⁻²`** so fix **both** `ddof` *and* `detrend_order` across compared runs. Under `detrend_order > 0` the k-statistic unbiasedness (orders ≥ 3) breaks — a `RuntimeWarning` suggests `cumulant_estimator="biased"`. |
| **D2** | Classical vs quantum distances | **Both ship.** `distance_flavor` ∈ {`classical`, `quantum`, `both`}. Classical is eigenvector-blind (spectrum only); quantum is eigenvector-sensitive and needs `materialize_matrices`. |
| **D3** | Which matrix feeds distances | **Covariance by default**; `contracted_third_cumulant` / `contracted_fourth_cumulant` spectra are selectable via `distance_source` (signed — the classical distances warn and are only strictly valid on the PSD covariance). |
| **D4** | Distance reference | **`"previous"` by default**; `"baseline"` (first measurement) and `"lag:k"` available. A transition-aware baseline-swap is a documented **TODO** (see `monitor._reference_for`). |
| **D5** | Gaussian 2RDM source in v1 | **Seam only.** `GaussianSource.collect` raises `NotImplementedError`. |
| **D6** | Eigenvector sign gauge | **Fixed.** `sign_gauge=True` flips each eigenvector column so its max-magnitude entry is positive (deterministic cross-step export). Squared/4th-power observables (IPR, block energy) are already sign-invariant; this matters only if you export `V` / odd-power principal cumulants. |

---

## `LoggingConfig`

Controls how a `Measurement` is flattened for a logger (full reference:
[logging.md](logging.md)).

| Field | Default | Meaning |
|-------|---------|---------|
| `vector_policy` | `"summary"` | global vector policy: `"summary"` (max/mean/top1/l2), `"full"` (raw array), `"off"` |
| `vector_overrides` | `{}` | per-key policy overrides, e.g. `{"covariance_spectrum": "full"}` |
| `log_health` | `True` | include the `health/*` scalars |
| `trim_consolidated` | `False` | opt into the §8 consolidation — suppress the deprecated-default keys (`von_neumann_entropy`, `classical_fidelity`, `quantum_fidelity`, `total_cumulant_{5+}`) from the flattened output, keeping their companions (`effective_rank`, the geodesic distances, the on-demand cumulant routine). Everything is still **computed** and present on the `Measurement`. |
| `drop_keys` | `{}` | extra user-specified keys to suppress at flatten time |

```python
from cumobs import LoggingConfig
lc = LoggingConfig(vector_policy="summary",
                   vector_overrides={"covariance_spectrum": "full"})
```

---

## Cost and performance

Per measurement (dynamical source; `m` = window, `n` = probes; common regime
`m ≪ n`):

| Stage | Cost | Allocates |
|-------|------|-----------|
| probe forward passes | `n` forwards | `(n,)` |
| `G₀ = B_c B_cᵀ` | `O(m²n)` | `(m,m)` |
| `eigh(G₀)` + spectrum | `O(m³)` | `(m,)`, `(m,m)` |
| principal projections `P` | `O(m·top_k)` | `(m, top_k)` |
| sample-space `V` *(opt-in)* | `O(mn·top_k)` | `(n, top_k)` |
| diagonal cumulants | `O(mn)` | `(m,n)` temporaries |
| contracted spectra (2× `eig` on `m×m`) | `O(m³)` | `(m,m)` |
| materialize `C`/`M3`/`K` *(opt-in)* | `O(mn²)` | `(n,n)` × 3 |

**Default path: `O(m²n + m³)`, peak allocation `O(mn)` — no `n²` term.** The
model's parameter count enters only through the `n` forward passes. With `m =
20`, a full measurement is `1 eigh + 2 eig` on `20 × 20` matrices — negligible.
