# Concepts — the math behind `cumobs`

This page explains the ideas the package implements, in plain terms, with just
enough notation to read the code and the [observable catalog](observables.md).
The authoritative version is [`cumulant_observables_spec.md`](../cumulant_observables_spec.md)
§3–§4; section numbers below point there.

---

## 1. The per-sample loss vector and the buffer `B`

Pick a fixed set of `n` **probe** inputs (held-out examples you re-evaluate
during training). At a measurement time you do one forward pass and record the
**per-sample loss** — a vector `ℓ ∈ ℝⁿ`, one number per probe. No gradients,
just `loss_fn(model, probe)` under `no_grad`.

Stack the last `m` such vectors into a **buffer**

```
B ∈ ℝ^{m × n}      m = window length (rows, "time")
                   n = number of probes (columns, "features")
```

Each **row** is a snapshot of the loss across probes at one timestep; each
**column** is one probe's loss trajectory over the window. `cumobs` computes the
statistics of the *columns'* joint fluctuations.

> **Generalization (§9.1).** Nothing in the math knows that a "feature" is a
> loss. Swap in flattened activations and you get representation cumulants. The
> only loss-specific choice is the aggregate direction `S` below, which is made
> injectable. See [extending.md](extending.md).

There are two ways to fill `B`:

- **Dynamical 2RDM** — append the current loss every few steps; the rows are
  *consecutive timesteps*. This is what `DynamicalSource` does and the default
  use case. Captures how the loss moves *along the training trajectory*.
- **Gaussian 2RDM** — the rows are `k` i.i.d. parameter perturbations around the
  current `θ`. Useful very early in training when you don't yet have enough
  trajectory samples. This is a **seam only** in v1 (`GaussianSource` raises
  `NotImplementedError`, decision D5).

---

## 2. Centering, cumulants, and the aggregate variable `S`

**Centering.** Subtract each probe's window-mean: `E_L = mean(B, axis=0)` (length
`n`), `B_c = B − E_L`. Now every column has zero mean over the window.

> **Detrending (`detrend_order`).** Because the dynamical-2RDM rows are
> *consecutive* training steps (not i.i.d. samples), a smooth drift in the loss
> over the window shows up as a spurious dominant covariance mode. Setting
> `detrend_order = d` subtracts the per-probe least-squares polynomial of degree
> `d` in the row index instead of just the mean (`d = 0` is the plain centering
> above; `d = 1` strips a linear drift). `E_L` stays the plain mean.

**Cumulants** are the connected moments of the centered fluctuations
`δℓ = B_c`. Order by order:

| Order | Object | Intuition |
|-------|--------|-----------|
| 1 | `E_L` (mean loss per probe) | where the loss sits |
| 2 | `κ₂[i,j] = Ê[δℓ_i δℓ_j]` — the **covariance / 2RDM** | how probes' losses co-fluctuate |
| 3 | `κ₃[i,j,k] = Ê[δℓ_i δℓ_j δℓ_k]` — co-skewness | asymmetry of the joint fluctuations |
| 4 | `κ₄[i,j,k,l]` — **connected** co-kurtosis | heavy-tailedness *beyond* what the covariance predicts |

The 4th cumulant is "connected": it subtracts the Gaussian (Wick/Isserlis) part
built from `κ₂`, so it is **zero for a Gaussian window** and grows only when the
fluctuations are genuinely non-Gaussian (§3.5):

```
κ₄[i,j,k,l] = Ê[δℓ_i δℓ_j δℓ_k δℓ_l]
            − κ₂[i,j]κ₂[k,l] − κ₂[i,k]κ₂[j,l] − κ₂[i,l]κ₂[j,k]
```

**The aggregate variable `S`.** The full co-skewness `κ₃` is an `n × n × n`
tensor and `κ₄` is `n × n × n × n` — far too big to form. The trick is to never
materialize them. Define the scalar **aggregate**

```
S = B_c · 1_n  ∈ ℝ^m         (row sums of the centered buffer)
```

`S_t` is the total centered loss at timestep `t`. Then the **"total" cumulants**
are just ordinary moments of the 1-D series `S`:

```
total_second_cumulant = κ₂ of S  (= Ê[S²])              = Σ_{ij}  κ₂[i,j]
total_third_cumulant  = κ₃ of S  (= Ê[S³])              = Σ_{ijk} κ₃[i,j,k]
total_fourth_cumulant = κ₄ of S  (= Ê[S⁴] − 3·Ê[S²]²)   = Σ_{ijkl} κ₄[i,j,k,l]
```

i.e. *fully contracting* each cumulant tensor with the all-ones direction equals
a cumulant of `S`. No big tensor is ever allocated. (For activations you may want
a different contraction direction than all-ones — that is the injectable
`contraction` parameter, [extending.md](extending.md).)

> These are **cumulants** `κ₃`/`κ₄`, not the standardized skewness/kurtosis
> coefficients — hence the names `total_third_cumulant` / `total_fourth_cumulant`
> (renamed in v0.2). The parenthesized formulas are the **plug-in** estimators
> (`cumulant_estimator="biased"`); the default `"unbiased"` replaces them with
> the Fisher k-statistics so a Gaussian window reads ≈ 0 instead of carrying an
> `O(1/m)` plug-in baseline. See
> [configuration.md](configuration.md#estimator-biased-vs-unbiased).

This ladder generalizes to **any order**: the cumulants `κ_p` of the 1-D series
`S` follow from its raw moments `m_k = Ê[S^k]` by the moment–cumulant recurrence

```
κ_p = m_p − Σ_{k=1}^{p−1} C(p−1, k−1) · κ_k · m_{p−k}
```

which reproduces `total_third_cumulant` (`κ₃`) and `total_fourth_cumulant` (`κ₄`)
and extends to 5th, 6th, … order at negligible cost (`max_scalar_cumulant_order`,
[observables.md](observables.md#general-higher-order-scalar-cumulants-testing)).
Useful mainly as a non-Gaussianity check; high orders cancel badly, so use
`float64`.

### The single divisor `D` (decision D1, corrected in R1)

Every empirical expectation uses **one** divisor

```
D = window_size − (ddof + detrend_order),     Ê[f] = (1/D) Σ_{t=1..m} f_t
```

with `ddof = 1` by default. At `detrend_order = 0` this is `D = m − 1` (matching
the paper's `C`). **R1 correction:** detrending of degree `d` is a rank-`(d+1)`
projection, so for Gaussian residuals `E[RSS] = σ²(m − (d+1))`; dividing by only
`m − ddof` biases the variance (and the whole covariance) downward, so the
divisor adds `detrend_order`. Using a single `D` everywhere — for raw moments
*and* the cumulant subtractions — is a hard requirement: mixing `1/m` and
`1/(m−1)` silently breaks the full-contraction identities above. Consequences:

- **PR, fidelity, trace distance** are `D`-invariant (ratios / normalized).
- **Spectral heat capacity** scales as `D⁻²`, so cross-run comparisons must fix
  **both** `ddof` *and* `detrend_order`.
- This restores order-2 unbiasedness but **not** the k-statistic unbiasedness at
  orders ≥ 3 (the `m²/((m−1)(m−2))` prefactors are derived for the centering
  projector only). Under `detrend_order > 0` either use
  `cumulant_estimator="biased"` (the single-`D` plug-in stays consistent under
  any projector) or accept the emitted `RuntimeWarning`.
- **Rank cap (R3):** `rank(C) ≤ m − (detrend_order+1)` *regardless of `n`*, so
  `PR` / `effective_rank` / `normalized_shc` / rank-mode SHC are window-limited.

(These invariances are asserted in `tests/test_math_core.py::test_ddof_invariances`.)

---

## 3. The 2RDM and its spectrum

The second cumulant is the covariance matrix (the paper's **two-particle reduced
density matrix**, hence "2RDM"):

```
C = B_cᵀ B_c / D   ∈ ℝ^{n × n}
```

You almost never want to form it. What you want are its **eigenvalues** `λ` (and
sometimes its top **eigenvectors** `V`). The spectrum carries the physics:

- **`covariance_spectrum`** — `λ`, sorted descending, zero-padded to length `n`.
- **`spectral_heat_capacity` (SHC)** — the variance of the spectrum, which
  **spikes at a phase transition** when one or a few eigenvalues pull away from
  the bulk. By default (`shc_mode="rank"`) it is taken over the nonzero
  eigenvalues, so it is invariant to the number of probes; `"full"` is the legacy
  variance over the zero-padded spectrum. The headline transition diagnostic (see
  the Fig. 1 example).
- **`participation_ratio` (PR)** — `(Σλ)² / Σλ²`, in `[1, r]`. The *effective
  number of modes* the fluctuations occupy. PR ≈ 1 means a single dominant
  rank-1 mode (everything moves together); PR ≈ `r` means isotropic.
- **`spectral_gap`, `top_eigenvalue_fraction`, `von_neumann_entropy`,
  `effective_rank`** — `n`-independent companions: the leading-mode separation
  `λ₁−λ₂` and its variance fraction `λ₁/Σλ` (condensation order parameters), and
  the spectral (von Neumann) entropy `−Σ p log p` of `p = λ/Σλ` with its
  exponential `effective_rank` (a smooth mode count).
- **`inverse_participation_ratio` (IPR)** — for each top eigenvector `v_a`,
  `Σ_i v_a[i]⁴`, in `[1/n, 1]`. The dual of PR: it counts the *effective number
  of probes* the mode lives on. IPR = 1 means the mode is a single probe; IPR =
  1/n means it is spread uniformly. (Needs `compute_eigenvectors=True`.)

---

## 4. The Gram trick (why it stays cheap)

`C` is `n × n` (probes), but its **nonzero eigenvalues equal those of the much
smaller Gram matrix**

```
G₀ = B_c B_cᵀ   ∈ ℝ^{m × m}      (window × window)
```

When `m ≤ n` (the common regime — short window, many probes) the package
eigendecomposes `G₀` (`O(m³)`), divides by `D`, and zero-pads to length `n`. The
single expensive operation is forming `G₀` itself (`O(m²n)`). The eigenvalues of
`C` are recovered exactly; the `min(m−1, n)` nonzero ones (the `−1` is from
centering) and the rest are structural zeros.

Two derived objects also stay in window space:

- **Principal projections** `P[:,a] = √μ_a · u_a ∈ ℝ^{m}` (from `eigh(G₀) →
  (μ_a, u_a)`). These satisfy `Ê[P_a²] = λ_a` exactly, so *principal cumulants*
  (`principal_third_cumulant`, `principal_fourth_cumulant`) are moments of `P` —
  `m`-vectors, never `n`-sized.
- **Sample-space eigenvectors** `V[:,a] = B_cᵀ u_a / √μ_a ∈ ℝ^{n}` (unit norm).
  This is the only object indexed by `n`, and it is `n × top_k`, **not** `n ×
  n`. Built only when IPR / block energy / quantum + CKA distances are requested
  (`compute_eigenvectors=True`).

When `m > n` (rare: long window, few probes) forming `C` directly is cheaper, so
the code branches to the direct path. Both paths are tested to agree on the
overlapping eigenvalues (`test_path_agreement_m_ge_n`).

---

## 5. Contracted-cumulant spectra (orders 3 & 4)

You can't eigendecompose the full `κ₃` / `κ₄` tensors, but you *can*
eigendecompose them after **contracting one (skew) or two (kurtosis) indices**
with the all-ones direction. That gives `n × n` matrices:

```
M3[i,j] = Σ_k    κ₃[i,j,k]       = Ê[δℓ_i δℓ_j S]          (contracted skew)
K [i,j] = Σ_{kl} κ₄[i,j,k,l]                               (contracted kurtosis)
```

and — crucially — their **spectra are again computable in window space** via
"Gram partners":

```
contracted_third_cumulant_spectrum   ←  eig( G3 = diag(S/D) · G₀ )
contracted_fourth_cumulant_spectrum  ←  eig( G4 = Q · G₀ )
```

where `Q = diag(S²/D) − (‖S‖²/D²)·I − (2/D²)·S Sᵀ` (a symmetric `m × m`
operator, §3.5). Because `G3`/`G4` are *products* of symmetric matrices they are
**asymmetric**, so the code uses the general `eig` and takes `.real`. In exact
arithmetic the imaginary parts vanish (the underlying `M3`, `K` are symmetric);
the leftover `max|Im| / max|Re|` is logged as a **numerical-health** metric
(`contracted_*_imag_ratio`). A large value (default warn threshold `1e-3`) flags
precision loss — use `dtype=float64` (the default).

---

## 6. Distances between measurements (decision D2, D4)

To detect *when* the loss structure changes, compare a measurement against a
reference one. Two flavors:

- **Classical** (spectrum-only, eigenvector-*blind* **and scale-blind**).
  Normalize each spectrum to a probability vector `p = λ / Σλ` and compute
  - `classical_fidelity = (Σ_i √(p_i q_i))²` — Bhattacharyya overlap, `1` iff
    equal.
  - `trace_distance = ½ Σ_i |p_i − q_i|` — total variation, `0` iff equal.

  Two states with the *same spectrum but rotated eigenvectors* score identical,
  and so do two whose spectra differ only by an overall scale (the normalization
  divides it out). Because they treat the spectrum as a probability vector they
  are only valid for the **PSD covariance** source — on the signed contracted
  spectra they clip the negative half and emit a warning.

- **`spectral_drift`** (scale-*sensitive*). `‖λ − μ‖₂ / max(‖λ‖₂, ‖μ‖₂)` on the
  **raw** (un-normalized) spectra — the companion that *does* respond to a pure
  change in total variance. Emitted alongside the classical distances.

- **Quantum** (density-matrix, eigenvector-*sensitive*, `O(n³)`). On `ρ = C/tr C`,
  the Uhlmann fidelity and Schatten-1 trace distance *do* see eigenvector
  rotation. They need the materialized covariance (`materialize_matrices=True`).

- **CKA** (`cka_similarity` / `cka_distance`, eigenvector-*sensitive*). Linear
  CKA — the Frobenius cosine `⟨C₁,C₂⟩_F / (‖C₁‖_F‖C₂‖_F)` — computed from the
  top-`k` eigen-factors at `O(n·k²)`, so it sees a subspace rotation like the
  quantum distances but without forming any `n×n` matrix. The natural
  *representation-drift* signal between checkpoints (`compute_cka_distance=True`,
  `compute_eigenvectors=True`).

The **reference** is chosen by `distance_reference` (decision D4): `"previous"`
(default), `"baseline"` (first measurement), or `"lag:k"`. A transition-aware
baseline-swap is a documented TODO.

---

## 7. Per-group decomposition (decision §3.7)

If you tag each probe with a **group label** (e.g. "data domain A vs B",
"in-distribution vs OOD"), `cumobs` decomposes the observables per group:

- `group_total_variance[a]` — summed per-probe variance over group `a`.
- `group_shc[a]`, `group_pr[a]` — re-run the spectral path on that group's
  columns.
- `block_energy[a, k]` — `Σ_{i ∈ group a} V[i,k]²`, how much of eigenvector `k`
  lives on group `a`. This is the **localization** signal: when a transition's
  top mode concentrates on one group, its block energy jumps to ≈ 1 (the
  acceptance check in the Fig. 1 example).

---

## 8. Putting it together — one measurement

`build_intermediates(B, cfg)` runs the shared linear algebra **once** in the
dependency order of §4 and returns a `SharedIntermediates` bundle; the section
functions read from it so nothing large is recomputed:

```
center → S, S² , S³, S⁴ → G₀ = B_c B_cᵀ → eigh(G₀) → λ, P (, V)
       → B_c powers → diagonal / total cumulants
       → G3, G4 → contracted spectra
       (→ C, M3, K only if materialize_matrices)
```

Default-path cost is `O(m²n + m³)` with peak allocation `O(mn)` — **no `n²`
term**. The model's parameter count enters only through the `n` probe forward
passes. See [the cost model](configuration.md#cost-and-performance) for the full
table.

---

## 9. The v0.3 layer (extension batch)

Almost everything in the extension batch is a **near-free byproduct of linear
algebra the package already runs** — `eigh(G₀)` for `C`, the asymmetric `eig` of
`G3`/`G4` for the contracted spectra. Conceptually:

- **Standardized cumulant spectrum** `γ_p = κ_p(S)/κ₂^{p/2}` (§1) — the
  dimensionless order profile of non-Gaussianity (γ₃ = skewness, γ₄ = excess
  kurtosis), a pure transform of the tracked cumulants. The **Fisher/Lee–Yang
  zero** (§1c) reads criticality off the truncated free energy `Z_P(t)` rather
  than off the spectrum: its gap → 0 near a transition.
- **Critical slowing down** (§5) — the paper's early-warning story is *diverging
  variance **and** rising autocorrelation*. SHC is the variance half;
  `autocorr_lag1` / `autocorr_time` (of the aggregate `S`) are the
  autocorrelation half, and `τ_int` gives the honest sample size `m_eff` used by
  the RMT null.
- **Null background** (§3) — a surrogate (phase-scrambled) bootstrap and the
  analytic Marchenko–Pastur / Tracy–Widom edge answer "is this mode
  distinguishable from no co-fluctuation," which neither SHC nor the jackknife
  does. The MP edge is the BBP detachment threshold; the **BBP bitmask** (§6)
  encodes *which probes* a detached mode acts on as one decodable integer.
- **Cross-operator alignment** (§7) — does the skew/kurtosis structure live
  *where* the variance condenses? A window-space cosine `cos(C, M3)` /
  `cos(C, K)`, free given `G₀`/`G3`/`G4`.
- **Geodesic & noise-floored distances** (§2/§4) — the Riemannian metric forms
  of the fidelities (Fisher–Rao, Bures) and drift with the null floor subtracted.
- **Eigenvector tracking** keeps mode identity stable across measurements
  (Hungarian matching), the prerequisite for every per-mode time series.

See [observables.md](observables.md) for the per-key reference and
[configuration.md](configuration.md) for the knobs.

---

Next: [getting-started.md](getting-started.md) to run it, or
[observables.md](observables.md) for the per-key reference.
