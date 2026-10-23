# Observable catalog

Every key a `Measurement` can contain, with its shape, formula, interpretation,
and what config it requires. Keys land in `meas.scalars` (float), `meas.vectors`
(Tensor), `meas.matrices` (Tensor), or `meas.health` (float). The math is in
[`cumobs/observables.py`](../cumobs/observables.py); section numbers point to
[`cumulant_observables_spec.md`](../cumulant_observables_spec.md).

Conventions used below:

- `m` = window length, `n` = number of probes, `D = m − ddof` (the single
  divisor, decision D1), `B_c` = centered/detrended buffer, `S` = aggregate
  variable (`B_c · 1_n`), `δℓ_i` = column `i` of `B_c`.
- `Ê[·] = (1/D) Σ_t (·)` is the empirical expectation.
- `λ` = covariance eigenvalues, `P` = principal projections (`m × top_k`), `V` =
  sample-space eigenvectors (`n × top_k`).
- **Gating** columns: which `CumulantConfig` setting must be on. "order ≥ k"
  means `cfg.order` must be at least `k`.

> **v0.2 note.** The named 3rd/4th cumulants are renamed to
> `*_third_cumulant` / `*_fourth_cumulant` (they are raw cumulants `κ₃`/`κ₄`,
> *not* standardized skewness/kurtosis coefficients). By default they now use the
> **unbiased** Fisher k-statistics (`cumulant_estimator="unbiased"`); set
> `"biased"` for the plug-in central-moment estimators. See
> [configuration.md](configuration.md#estimator-biased-vs-unbiased).

> **v0.3 note.** A large additive batch (the *cumobs extension spec*) lands a
> standardized cumulant spectrum and Fisher/Lee–Yang zero (§1), geodesic
> distances (§2), a surrogate + Marchenko–Pastur null model (§3), noise-floored
> distances (§4), critical-slowing-down autocorrelation (§5), a BBP
> participation bitmask (§6), and cross-operator alignment (§7), plus
> eigenvector tracking. All keys are **additive**; the §8 consolidations are
> opt-in via `LoggingConfig(trim_consolidated=True)`. One **behaviour change**:
> the single divisor is now `D = m − (ddof + detrend_order)` (R1 — only differs
> from before when `detrend_order > 0`).

---

## Quick index

| Bucket | Keys |
|--------|------|
| 1st order | `first_cumulant_vector`, `total_first_cumulant` |
| 2nd order | `total_second_cumulant`, `covariance_spectrum`, `spectral_heat_capacity`, `participation_ratio`, `spectral_gap`, `top_eigenvalue_fraction`, `von_neumann_entropy`, `effective_rank`, `normalized_shc`, `log_det`, `inverse_participation_ratio`, `covariance` |
| 3rd order | `total_third_cumulant`, `diagonal_third_cumulant`, `principal_third_cumulant`, `contracted_third_cumulant_spectrum`, `contracted_third_cumulant_matrix` |
| 4th order | `total_fourth_cumulant`, `diagonal_fourth_cumulant`, `principal_fourth_cumulant`, `wick_breakdown_ratio`, `contracted_fourth_cumulant_spectrum`, `contracted_fourth_cumulant_matrix` |
| standardized (§1) | `standardized_cumulant_spectrum`, `principal_standardized_cumulant`, `principal_cumulant_{p}` (p≥5, opt-in) |
| Fisher zero (§1c) | `leading_fisher_zero_re`/`_im`/`_abs`/`_arg`, `fisher_zero_gap` |
| 5th+ order (testing) | `total_cumulant_5`, `total_cumulant_6`, … (general scalar cumulants of the aggregate, plug-in) |
| distances | `classical_fidelity`, `trace_distance`, `spectral_drift`, `fisher_rao_distance`, `quantum_fidelity`, `quantum_trace_distance`, `bures_distance`, `bures_angle`, `cka_similarity`, `cka_distance` |
| noise-floored (§4) | `denoised_spectral_drift`, `spectral_drift_snr`, `signal_energy_drift` |
| autocorrelation (§5) | `autocorr_lag1`, `autocorr_time`, `principal_autocorr_lag1`, `principal_autocorr_time` |
| alignment (§7) | `align_cov_skew`, `align_cov_kurtosis`, `subspace_align_cov_skew`, `subspace_align_cov_kurtosis` |
| null model (§3) | `<scalar>/null_mean`/`null_sd`/`null_z`, `mp_edge_upper`, `n_spikes`, `spike_energy_fraction` |
| BBP bitmask (§6) | `active_probe_mask`, `active_probe_mask_any` (aux), `active_probe_count` |
| tracking | `mode_permutation`, `crossing_detected` |
| per-group | `group_total_variance/<label>`, `group_shc/<label>`, `group_pr/<label>`, `block_energy/<label>` |
| byproduct | `excess_loss` |
| error bars | `<scalar>/se` (every scalar, when `error_bars="jackknife"`) |
| health | `rank`, `n_principal`, `n_nonzero`, `use_gram`, `constant_window`, `effective_window`, `mp_bulk_ks`, `mp_gamma`, `mp_sigma2`, `contracted_third_cumulant_imag_ratio`, `contracted_fourth_cumulant_imag_ratio` |

---

## First cumulant (mean)

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `first_cumulant_vector` | vectors | `(n,)` | `E_L = mean(B, axis=0)` (per-probe mean loss) | always |
| `total_first_cumulant` | scalars | scalar | `E_L.sum()` | always |

The baseline level of the loss. `E_L` is always the plain per-probe mean (the
degree-0 trend), even when `detrend_order > 0`. Mostly useful as a reference and
for the excess-loss byproduct (below).

---

## Second cumulant (the 2RDM) and spectral diagnostics

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `total_second_cumulant` | scalars | scalar | `Ê[S²] = ‖S‖²/D = Σ_{ij} κ₂[i,j]` | always |
| `covariance_spectrum` | vectors | `(n,)` | eigenvalues `λ` of `C`, clipped ≥ 0, sorted desc, zero-padded | always |
| `spectral_heat_capacity` | scalars | scalar | variance of `λ` (see `shc_mode`) | always |
| `participation_ratio` | scalars | scalar | `(Σλ)² / Σλ²` ∈ `[1, r]` | always |
| `spectral_gap` | scalars | scalar | `λ₁ − λ₂` | always |
| `top_eigenvalue_fraction` | scalars | scalar | `λ₁ / Σλ` ∈ `[0, 1]` | always |
| `von_neumann_entropy` | scalars | scalar | `−Σ_i p_i log p_i`, `p = λ/Σλ` | always |
| `effective_rank` | scalars | scalar | `exp(von_neumann_entropy)` ∈ `[1, r]` | always |
| `normalized_shc` | scalars | scalar | `Var(λ)/mean(λ)² = r/PR − 1` over nonzero λ | always |
| `log_det` | scalars | scalar | `Σ log(λ_i + eps)` over the `rank` nonzero λ | always |
| `inverse_participation_ratio` | vectors | `(top_k,)` | `IPR[a] = Σ_i V[i,a]⁴` ∈ `[1/n, 1]` | `compute_eigenvectors` |
| `covariance` | matrices | `(n,n)` | `C = B_cᵀ B_c / D` | `materialize_matrices` |

**Interpretation.**

- **`spectral_heat_capacity` (SHC)** — the headline transition diagnostic: the
  variance of the eigenvalue distribution, which **spikes** when one or a few
  modes detach from the bulk. By default (`shc_mode="rank"`) it is the variance
  over the `rank` **nonzero** eigenvalues, so it is *invariant to the number of
  probes / structural zeros*. `shc_mode="full"` is the legacy variance over the
  full zero-padded length-`n` spectrum (which shrinks as `1/n` when you add
  probes). Either way it scales as `D⁻²`, so fix `ddof` across compared runs.
  > **R2 (re-grounding).** SHC is a **condensation / anisotropy order
  > parameter**, *not* a specific heat. The literal heat-capacity analog in the
  > paper's cluster expansion is the *energy* fluctuation `∂²log Z/∂β² =
  > Var(Σℓ_i) = 1ᵀC1 = total_second_cumulant`, already in the scalars bucket.
  > The name is kept as analogy.
  > **R3 (rank cap).** `rank(C) ≤ m − (detrend_order+1)` *regardless of `n`* —
  > at `m=20` you resolve ≤ 19 modes even with 512 probes, so SHC (rank mode),
  > `PR`, `effective_rank`, and `normalized_shc` are **window-limited** and
  > partly track window length rather than the model.
- **`normalized_shc`** — `Var(λ)/mean(λ)²` over the nonzero spectrum, which
  equals `r/PR − 1` exactly (a scale-free condensation index that removes the
  `n`/structural-zero effect). It is a convenience transform of `PR` and `rank`,
  not independent state. `NaN` for a dead window.
- **`log_det`** — `Σ log λ_i` over the nonzero eigenvalues (with an `eps`
  jitter), the log-determinant of the covariance restricted to its range — a
  volume/uncertainty scalar logged in the paper's grokking set. `NaN` when the
  rank is zero.
- **`participation_ratio` (PR)** — the effective number of active modes. PR ≈ 1
  ⇒ a single rank-1 mode dominates (everything co-moves); PR ≈ `r` ⇒ isotropic.
  Returns `NaN` for a dead/constant window (Σλ² ≈ 0). Invariant to structural
  zeros.
- **`spectral_gap` / `top_eigenvalue_fraction`** — direct *condensation* order
  parameters: the separation of the leading mode from the rest (`λ₁ − λ₂`) and
  the fraction of total variance it carries (`λ₁/Σλ → 1` as one mode dominates).
  These are `n`-independent complements to SHC.
  > **§8.** `spectral_gap` is a *condensation / one-mode-detaching* parameter,
  > **not** a relaxation time (relaxation time lives in the Hessian / trajectory
  > autocorrelation — see `autocorr_time` / `principal_autocorr_*`). Once the
  > MP null is on, prefer the noise-edge gap `edge_gap = λ₁ − mp_edge_upper`.
- **`von_neumann_entropy` / `effective_rank`** — the Shannon entropy of the
  normalized spectrum `p = λ/Σλ` (the von Neumann entropy of `ρ = C/tr C`) and
  its exponential. `effective_rank` is a smooth "number of modes" in `[1, r]`
  (`= n` for an isotropic spectrum, `→ 1` for rank-1) — a softer companion to PR.
  `NaN` for a dead window.
- **`inverse_participation_ratio` (IPR)** — *per top eigenvector*, `Σ_i V[i,a]⁴`
  ∈ `[1/n, 1]`. It is **small** when the mode is spread out (`= 1/n` uniform) and
  `= 1` when the mode is a single probe; the participation *number* (effective
  probes the mode occupies) is its inverse `1/IPR`. Sign-invariant (4th power).
- **`covariance_spectrum`** — the raw eigenvalues; feeds the classical distances.
  Padded with `max(0, n − r)` zeros (`r` = number of computed eigenvalues).

---

## Third cumulant (co-skewness)

| Key | Bucket | Shape | Formula (estimator) | Gating |
|-----|--------|-------|---------------------|--------|
| `total_third_cumulant` | scalars | scalar | `κ₃` of `S` (k-statistic / `Ê[S³]`) = `Σ_{ijk} κ₃[i,j,k]` | order ≥ 3 |
| `diagonal_third_cumulant` | vectors | `(n,)` | `κ₃` of `δℓ_i` per probe | order ≥ 3 |
| `principal_third_cumulant` | vectors | `(top_k,)` | `κ₃` of `P_a` per principal mode | order ≥ 3 |
| `contracted_third_cumulant_spectrum` | vectors | `(n,)` | real eig of `M3` via `G3 = diag(S/D)·G₀`, sorted desc, padded | order ≥ 3 |
| `contracted_third_cumulant_matrix` | matrices | `(n,n)` | `M3 = B_cᵀ diag(S/D) B_c`, `M3[i,j] = Σ_k κ₃[i,j,k]` | order ≥ 3 **and** `materialize_matrices` |

**Interpretation.** The third cumulant measures asymmetry of the loss
fluctuations (these are *cumulants* `κ₃`, not the standardized skewness
coefficient `κ₃/κ₂^{3/2}`). The *total* is `κ₃` of the aggregate `S`; the
*diagonal* is per-probe; the *principal* projects onto the leading covariance
modes. The named `total_/diagonal_/principal_third_cumulant` use the
`cumulant_estimator` (unbiased k-statistic by default). The *contracted spectrum*
eigendecomposes the once-contracted `κ₃` (always the plug-in matrix `M3`) — its
spread complements SHC for asymmetric transitions. `M3`'s Gram partner `G3` is
**asymmetric**, so the eigenvalues use general `eig` + `.real`; see
`contracted_third_cumulant_imag_ratio` under [health](#numerical-health).

---

## Fourth cumulant (connected co-kurtosis)

| Key | Bucket | Shape | Formula (estimator) | Gating |
|-----|--------|-------|---------------------|--------|
| `total_fourth_cumulant` | scalars | scalar | connected `κ₄` of `S` (k-statistic / `Ê[S⁴]−3Ê[S²]²`) = `Σ_{ijkl} κ₄` | order = 4 |
| `diagonal_fourth_cumulant` | vectors | `(n,)` | connected `κ₄` of `δℓ_i` per probe | order = 4 |
| `principal_fourth_cumulant` | vectors | `(top_k,)` | connected `κ₄` of `P_a` per principal mode | order = 4 |
| `wick_breakdown_ratio` | vectors | `(top_k,)` | `principal_fourth_cumulant[a] / (3·λ_a² + eps)` | order = 4 |
| `contracted_fourth_cumulant_spectrum` | vectors | `(n,)` | real eig of `K` via `G4 = Q·G₀`, sorted desc, padded | order = 4 |
| `contracted_fourth_cumulant_matrix` | matrices | `(n,n)` | `K = B_cᵀ Q B_c`, `K[i,j] = Σ_{kl} κ₄[i,j,k,l]` | order = 4 **and** `materialize_matrices` |

**Interpretation.** The 4th cumulant is **connected** — it subtracts the
Gaussian (Wick/Isserlis) part, so it is **zero in expectation for a Gaussian
window** *when using the unbiased estimator* (the default). With
`cumulant_estimator="biased"` the plug-in has an `O(1/m)` negative baseline of
order `−c·κ₂²/m` that co-varies with the variance — informative for *changes* but
not "≈ 0 for Gaussian" at small windows.

- **`wick_breakdown_ratio`** is the most interpretable: per principal mode it is
  the ratio of the connected kurtosis to the Gaussian prediction `3λ_a²`. It is
  `0` when Isserlis holds and grows as non-Gaussianity develops — the direct
  numerical analog of "the cross terms dropped under the Gaussian assumption" in
  the paper's Appendix C.2. (`λ_a` is the eigenvalue / principal variance, tied
  to the divisor `D`.)
  > **R5.** Per mode `wick_breakdown_ratio[a] = κ₄(P_a)/(3λ_a²)` is exactly the
  > per-mode standardized excess kurtosis over 3 — i.e. `γ₄(P_a)/3`, the
  > `p=4` column of `principal_standardized_cumulant`. Not independent of it.
- `Q = diag(S²/D) − (‖S‖²/D²)·I − (2/D²)·S Sᵀ` is the symmetric kurtosis
  contraction operator (§3.5). Its Gram partner `G4 = Q·G₀` is asymmetric ⇒ `eig`
  + `.real`; watch `contracted_fourth_cumulant_imag_ratio`.

---

## General higher-order scalar cumulants (testing)

The "total" cumulants (`total_second_cumulant`, `total_third_cumulant`,
`total_fourth_cumulant`) are the cumulants `κ₂, κ₃, κ₄` of the **aggregate
variable** `S = B_c · 1_n`. That ladder generalizes to **any order** via the
moment–cumulant recurrence — no extra `n`-space work, since `S` is a length-`m`
series:

```
κ_p = m_p − Σ_{k=1}^{p−1} C(p−1, k−1) · κ_k · m_{p−k},   m_k = Ê[S^k]
```

Set `max_scalar_cumulant_order = P > 4` (with `order = 4`) and the measurement
gains one scalar per extra order:

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `total_cumulant_p` (`p` = 5…`P`) | scalars | scalar | `κ_p` of the aggregate `S` (plug-in moment–cumulant recurrence) | `order = 4` **and** `max_scalar_cumulant_order ≥ p` |

Orders 2–4 are already the named totals, so only `p ≥ 5` add new keys. These
higher orders are **plug-in in both estimator modes** — no closed-form
k-statistic ships beyond order 4 (mirroring `scipy.stats.kstat`).

**Interpretation.** Higher-order non-Gaussianity probes of the *total* loss
fluctuation: ≈ 0 for a Gaussian window, growing with heavy tails. Primarily a
**mathematical check** / diagnostic — rarely needed beyond 5th–6th order on small
probe sets, and numerically delicate (catastrophic cancellation + large sampling
variance), so keep `dtype=float64`. The underlying routine is exported as
[`total_scalar_cumulants(series, D, max_order)`](api-reference.md#observablespy)
and works on **any** 1-D series.

---

## Standardized cumulant spectrum (§1)

The *order profile of non-Gaussianity*: which orders of fluctuation are nonzero,
in dimensionless (scale-free) form. Pure transforms of the cumulants the package
already tracks — they carry no new information, but are the primary readout.

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `standardized_cumulant_spectrum` | vectors | `(P−2,)` | `γ_p = κ_p(S)/κ₂(S)^{p/2}`, `p=3…P` (`P = max_scalar_cumulant_order`) | order ≥ 3 |
| `principal_standardized_cumulant` | vectors | `(top_k, q−2)` | `κ_p(P_a)/λ_a^{p/2}`, `p=3…q` | order ≥ 3 |
| `principal_cumulant_{p}` | vectors | `(top_k,)` | per-mode `κ_p(P_a)`, `p ≥ 5` | `principal_cumulant_order ≥ p` |

`γ₃` is classical Fisher skewness, `γ₄` is excess kurtosis. The first nonzero
`γ_p` is the leading non-Gaussianity (odd `p` = asymmetry, even `p` =
tail/peakedness). `γ₃`/`γ₄` inherit `cumulant_estimator` (unbiased k-statistics
by default); `γ_{p≥5}` are plug-in. `q = min(order, 4)` by default, extended to
`principal_cumulant_order` (the *one* genuinely new primitive — per-mode cumulants
past kurtosis — gated behind its own order so the default cost is unchanged).
Keep `float64` (the recurrence cancels badly at high `p`).

> The reconstruct-on-demand helper `factorization_residual(cumulants, M, N)`
> gives the residual of the M-point function after the most generous
> reconstruction into blocks `≤ N` (`R(M,M−1)=κ_M`, `R(6,3)=κ₆+15κ₄κ₂`, …) — a
> fixed combinatorial function of the stored cumulants, never tracked.

---

## Partition-function (Fisher / Lee–Yang) zero (§1c)

The truncated free energy `Z_P(t) = Σ_p m_p t^p/p!` (moments reconstructed from
the cumulants) is a degree-`P` polynomial whose complex roots are the **Fisher /
Lee–Yang zeros**. The zero nearest the positive real axis controls the analytic
structure of the free energy — its gap shrinking toward zero is the
partition-function signature of an approaching transition.

| Key | Bucket | Formula | Gating |
|-----|--------|---------|--------|
| `leading_fisher_zero_re` / `_im` / `_abs` / `_arg` | scalars | the leading complex root `t` (split, since `scalars` is float) | order ≥ 2 |
| `fisher_zero_gap` | scalars | distance of `leading_fisher_zero` to the positive real axis (`→ 0` near a transition) | order ≥ 2 |

To leading order `gap ≈ √(2/κ₂)`, so it mostly tracks
`1/√total_second_cumulant`; the **genuinely new** content is the **phase**
(`_arg`), switched on by `κ₃` (asymmetry tilts the zero off the imaginary axis).
`root_order` truncates `Z_P`; it defaults to `min(4, P_tot)`. `NaN` for a dead
window.

---

## Critical slowing down — autocorrelation (§5)

The paper's early-warning story is "diverging variance **and** rising
autocorrelation." SHC is the variance half; this is the autocorrelation half — a
genuine *new leading indicator* — and it also produces `τ_int` for the honest
sample size.

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `autocorr_lag1` | scalars | scalar | `ρ₁(S)` — **rising = CSD early warning** | always |
| `autocorr_time` | scalars | scalar | `τ_int(S) = 1 + 2 Σ_{k≥1} ρ_k(S)` (Sokal auto-window) | always |
| `effective_window` | health | scalar | `(m − (detrend_order+1)) / max(τ_int, 1)` — honest sample size | always |
| `principal_autocorr_lag1` | vectors | `(top_k,)` | `ρ₁` of the tracked per-mode eigenvalue series across measurements | `track_modes` |
| `principal_autocorr_time` | vectors | `(top_k,)` | `τ_int` of the same per-mode series | `track_modes` |

`autocorr_*` on `S` are 1-D and tracking-free (always-on); `autocorr_time`
reports the raw Sokal estimate (can be `< 1` for anti-correlated rows), while the
`m_eff` used for `effective_window` and the MP/TW null floors `τ_int` at 1 (an
effectively-independent count can't exceed the row count). The per-mode versions
need the monitor's ring buffer of tracked eigenvalues (§ tracking). A large
`τ_int` warns (RMT calibration then uses `m_eff`, and the jackknife SE
underestimates).

---

## Cross-operator alignment (§7)

Does the skew (`M3`) or connected-kurtosis (`K`) structure live *where* the
variance (`C`) is condensing? A near-free byproduct of the existing
`G₀`/`G3`/`G4` work (the window-space trace identity makes the `1/D` cancel).

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `align_cov_skew` | scalars | scalar | `cos(C, M3) = tr(G3 G₀)/(√tr(G₀²)·√tr(G3²))` ∈ `[−1,1]` | order ≥ 3 |
| `align_cov_kurtosis` | scalars | scalar | `cos(C, K)` (same, with `G4`) ∈ `[−1,1]` | order = 4 |
| `subspace_align_cov_skew` | vectors | `(top_k,)` | cos of principal angles between `V_C` and `V_{M3}=orthonormalize(B_cᵀ W_{M3})` | order ≥ 3 + `compute_eigenvectors` |
| `subspace_align_cov_kurtosis` | vectors | `(top_k,)` | same, `V_C` vs `V_K` | order = 4 + `compute_eigenvectors` |

Signed: "is non-Gaussianity co-locating with variance condensation." This is
**not** the paper's Gaussian-part `⟨C⁽¹⁾, Δ⟩_F` (the Isserlis piece `K`
discards) — sell it as the co-location measure it is, the link to the
early-warning theorem is analogy. The subspace eigenvectors `W` of the
asymmetric `G3`/`G4` are recovered for free (`v = B_cᵀ w` is an `M3`/`K`
eigenvector) — the whole layer is online, no `materialize_matrices`.

---

## Null-background model (§3)

A parameter-light null answering "is this mode/jump distinguishable from no
co-fluctuation," which neither the SHC threshold nor the jackknife SE does. Off
by default; set `null_model`.

**Surrogate (block bootstrap)** — `null_model="surrogate"` (or `"both"`):
`null_samples` surrogates by independent circular column shifts of `B_c`
(destroys cross-probe phase = the 2RDM signal, preserves marginals +
within-column autocorrelation). For every scalar:

| Key | Bucket | Formula |
|-----|--------|---------|
| `<scalar>/null_mean` | scalars | mean over surrogates |
| `<scalar>/null_sd` | scalars | sd over surrogates |
| `<scalar>/null_z` | scalars | `(obs − null_mean)/null_sd` |

(plus `spectral_drift/null_mean`/`null_sd`, the floor for the §4 noise-floored
drift.)

**Marchenko–Pastur / Tracy–Widom** — `null_model="mp"` (or `"both"`), aspect
`γ = n/m_eff`, robust bulk noise `σ̂²` (median-matched):

| Key | Bucket | Formula | Interpretation |
|-----|--------|---------|----------------|
| `mp_edge_upper` | scalars | `σ̂²·(1 + √γ)²` | bulk edge; BBP detachment threshold |
| `edge_gap` | scalars | `λ₁ − mp_edge_upper` | the re-grounded condensation gap (§8): leading mode vs the *noise edge*, preferred over the bare `spectral_gap` once MP is on |
| `n_spikes` | scalars | `#{λ_a > edge + 2·tw_sd}` | detached modes (TW-gated) |
| `spike_energy_fraction` | scalars | `Σ_{λ>edge} λ / Σλ` | variance above the noise sea |
| `mp_bulk_ks` | health | KS distance of the sub-edge spectrum to the MP density | null goodness-of-fit |

MP/TW use `m_eff` (autocorrelated rows), so they are only *approximate* on the
dynamical source; the Gaussian source (§9 seam) would make them exact. A
`null_z` (or `n_spikes`) crossing threshold is the principled transition trigger.

---

## BBP spike-detachment bitmask (§6)

For each detached mode (`λ_a > mp_edge_upper`), an integer encoding *which probes
participate* — the support is the smallest index set capturing `rho_keep` of the
eigenvector's energy. Needs `compute_eigenvectors` **and** the MP edge
(`null_model` ∈ {`mp`, `both`}).

| Key | Bucket | Shape | Formula |
|-----|--------|-------|---------|
| `active_probe_mask` | aux | `(n_detached,)` hex strings | per mode: bit `i` set ⇔ probe `i` participates |
| `active_probe_mask_any` | aux | hex string | OR over all detached modes |
| `active_probe_count` | vectors | `(n_detached,)` | participating-probe count per mode |

Bit `i` ↔ probe `i` in `ProbeSet` order, LSB = probe 0 (the D6 sign gauge is
irrelevant — entries are squared). The mask is a Python big-int that doesn't fit
the float `scalars` bucket, so it lives in `meas.aux` serialized as a **hex
string** (passes through `to_flat_dict`; the JSONL logger writes it verbatim).
With `group_labels`, decode the mask → which *groups* light up (the Fig.1
localization result as one decodable integer).

---

## Eigenvector tracking (§7 step 2)

With `track_modes=True` (needs `compute_eigenvectors`) the monitor matches
sample-space eigenvectors across measurements (Hungarian on `|⟨v_a(t),
v_b(t+1)⟩|`, SciPy if present else greedy) to keep mode identity stable through
crossings — the prerequisite for the per-mode CSD series above.

| Key | Bucket | Shape | Formula | Gating |
|-----|--------|-------|---------|--------|
| `mode_permutation` | vectors | `(top_k,)` | current eigenvector column placed at each canonical mode slot | `track_modes` |
| `crossing_detected` | scalars | scalar | `1.0` if the optimal match permutes the descending-eigenvalue order | `track_modes` |

---

## Distances between measurements

Computed by the **monitor** (not `compute_all` — they need a reference
measurement). The source spectrum/matrix is selected by `distance_source`
(default `"covariance"`); the reference by `distance_reference` (default
`"previous"`). All return Python floats; degenerate inputs (zero total mass)
return `NaN` and are skipped in logging. On the *first* measurement there is no
reference, so distances are absent.

| Key | Bucket | Formula | Range | Gating |
|-----|--------|---------|-------|--------|
| `classical_fidelity` | scalars | `(Σ_i √(p_i q_i))²` on normalized spectra | `[0,1]`, `1` iff equal | `distance_flavor` ∈ {classical, both} |
| `trace_distance` | scalars | `½ Σ_i |p_i − q_i|` (total variation) | `[0,1]`, `0` iff equal | `distance_flavor` ∈ {classical, both} |
| `spectral_drift` | scalars | `‖λ − μ‖₂ / max(‖λ‖₂, ‖μ‖₂)` on **raw** spectra | `[0, ~√2]`, `0` iff equal | `distance_flavor` ∈ {classical, both} |
| `fisher_rao_distance` | scalars | `2·arccos(√classical_fidelity)` — geodesic on the simplex | `[0, π]`, proper metric | `distance_flavor` ∈ {classical, both} |
| `quantum_fidelity` | scalars | Uhlmann `(tr√(√ρ σ √ρ))²` on `ρ = C/tr C` | `[0,1]` | `distance_flavor` ∈ {quantum, both}, `materialize_matrices`, `distance_source="covariance"` |
| `quantum_trace_distance` | scalars | `½‖ρ − σ‖₁` | `[0,1]` | same as `quantum_fidelity` |
| `bures_distance` | scalars | `√(2(1 − √quantum_fidelity))` — Bures geodesic | `[0, √2]` | same as `quantum_fidelity` |
| `bures_angle` | scalars | `arccos(√quantum_fidelity)` — operator analog of Fisher–Rao | `[0, π/2]` | same as `quantum_fidelity` |
| `cka_similarity` | scalars | linear CKA `⟨C₁,C₂⟩_F / (‖C₁‖_F‖C₂‖_F)` on top-`k` factors | `[0,1]`, `1` iff proportional | `compute_cka_distance`, `compute_eigenvectors` |
| `cka_distance` | scalars | `1 − cka_similarity` | `[0,1]` | same as `cka_similarity` |
| `denoised_spectral_drift` | scalars | `max(0, (spectral_drift − μ₀)/(1 − μ₀))` | `[0,1]` | `null_model` ∈ {surrogate, both}, `distance_source="covariance"` |
| `spectral_drift_snr` | scalars | `(spectral_drift − μ₀)/s₀` | — | same as `denoised_spectral_drift` |
| `signal_energy_drift` | scalars | `spectral_drift` on `max(0, λ − mp_edge_upper)` only | `[0, ~√2]` | `null_model` ∈ {mp, both}, `distance_source="covariance"` |

**Geodesic forms (§2).** `fisher_rao_distance` (shape / simplex) and
`bures_distance` / `bures_angle` (operator) are the Riemannian metric forms of
the two fidelities — they respect the triangle inequality and the fidelity is
recoverable from them, so they are the preferred default. They ship *alongside*
the bare fidelities; set `LoggingConfig(trim_consolidated=True)` to suppress the
bare `classical_fidelity` / `quantum_fidelity` from the default log. When `ρ, σ`
commute, `bures_angle = ½·fisher_rao_distance`. `fisher_rao_distance` inherits
`classical_fidelity`'s negative-mass clip-and-warn on signed sources (R6).

**Noise-floored forms (§4).** `denoised_spectral_drift` / `spectral_drift_snr`
subtract the surrogate null floor `μ₀, s₀` (null mean/sd of the drift between
independent surrogates); `signal_energy_drift` zeroes the sub-edge bulk
analytically using the MP edge. Their zero point is "indistinguishable from
noise."

> **R7.** The quantum/Bures framing is an **analogy** to surface the
> basis-selection problem, *not* a literal reduced density matrix. The
> quantum/Bures distances are an expensive (`O(n³)`) way to measure subspace
> drift that CKA measures cheaply — reserve them for offline checkpoint analysis.

**Classical** distances are eigenvector-*blind* (spectrum only) **and
scale-blind** (`classical_fidelity` / `trace_distance` normalize the spectrum to
a probability vector, so a window whose spectrum keeps its shape but grows 10×
scores identical). **`spectral_drift`** is the scale-*sensitive* companion: it
compares the raw (un-normalized) spectra, so a pure variance change shows up.

> ⚠️ The contracted (3rd/4th) spectra are **signed**; the classical fidelity /
> trace distance treat a spectrum as a probability distribution and clip the
> negative half. Selecting `distance_source="contracted_*"` therefore emits a
> `RuntimeWarning` (significant negative mass). Use `"covariance"`, the quantum
> distances, or the CKA distance for signed/contracted sources.

**Quantum** distances are eigenvector-*sensitive* and cost `O(n³)` — they need
the materialized covariance and only make sense for the PSD covariance source.
**CKA** is also eigenvector-sensitive (it sees a subspace rotation) but is built
from the top-`k` eigen-factors at `O(n·k²)` — much cheaper than the quantum
distances, and the natural "representation drift" signal between checkpoints. See
[concepts.md §6](concepts.md#6-distances-between-measurements-decision-d2-d4).

---

## Per-group decomposition

Requires `per_group=True` and `group_labels` on the `ProbeSet`. Keys are
suffixed with the group label, e.g. `group_shc/A`.

| Key | Bucket | Formula | Gating |
|-----|--------|---------|--------|
| `group_total_variance/<label>` | scalars | `Σ_{i ∈ group} Ê[δℓ_i²]` | `per_group` |
| `group_shc/<label>` | scalars | SHC of the group's column sub-buffer (honours `shc_mode`) | `per_group` |
| `group_pr/<label>` | scalars | PR of the group's column sub-buffer | `per_group` |
| `block_energy/<label>` | vectors `(top_k,)` | `Σ_{i ∈ group} V[i,k]²` per eigenvector `k` | `per_group` **and** `compute_eigenvectors` |

**`block_energy`** is the localization signal: the fraction of each top
eigenvector that lives on the group. When a transition's leading mode
concentrates on one group, its block energy jumps toward 1 — the behavioral
acceptance check in the [Fig. 1 example](../examples/fig1_transition.py)
(`energy_a ≈ 1.0`, `energy_b ≈ 0`).

---

## Excess-loss byproduct (§9.4)

| Key | Bucket | Formula | Gating |
|-----|--------|---------|--------|
| `excess_loss` | scalars | `Σ_i (E_L[i] − ℓ*(θ*)[i])` | `reference_loss` passed to the monitor / `compute_all` |

The summed probe loss in excess of a converged-model reference `ℓ(θ*)`,
"effectively free if computing the 2RDM" (it reuses the mean `E_L`). The inverse
temperature `β` is absorbed by the caller. Only emitted when you supply
`reference_loss`. (This is a loss-gap diagnostic, **not** a Watanabe local
learning coefficient — it was renamed from `trajectory_llc` to avoid that
implication.)

---

## Error bars (optional)

With `error_bars="jackknife"`, every scalar observable gains a leave-one-row-out
jackknife standard error:

| Key | Bucket | Formula | Gating |
|-----|--------|---------|--------|
| `<scalar>/se` | scalars | `√((m−1)/m · Σ_i (θ_(i) − θ̄)²)` over leave-one-out estimates | `error_bars="jackknife"` |

This re-runs the scalar pipeline `m` times (one row dropped each), so it is
**`m`× more expensive** — gate it behind an SHC threshold or use it offline. It
is the cheapest way to tell whether a small-window cumulant move is signal or
sampling noise. (NaN leave-one-out values are dropped per key.)

---

## Numerical health

Always present in `meas.health` (logged under `health/*` by default). These are
diagnostics, not observables.

| Key | Meaning |
|-----|---------|
| `rank` | number of eigenvalues above `eig_threshold × μ_max` |
| `n_principal` | principal directions actually computed (`top_k` clamped to `rank`) |
| `n_nonzero` | same as `rank` (alias) |
| `use_gram` | `1.0` if the Gram path (`m ≤ n`) was used, `0.0` for the direct path |
| `constant_window` | `1.0` if the buffer has ~zero total variance (dead window) |
| `effective_window` | honest sample size `(m − (detrend_order+1)) / max(τ_int, 1)` (§5) |
| `mp_bulk_ks` | KS distance of the sub-edge spectrum to the MP density (`null_model` ∈ {mp, both}) |
| `mp_gamma` / `mp_sigma2` | the MP aspect `n/m_eff` and robust bulk noise `σ̂²` (same gating) |
| `contracted_third_cumulant_imag_ratio` | `max|Im λ| / (max|Re λ| + eps)` for `G3`; should be ≈ 0 |
| `contracted_fourth_cumulant_imag_ratio` | same for `G4`; should be ≈ 0 |

A large `*_imag_ratio` (default warn threshold `imag_ratio_warn = 1e-3`) signals
precision loss or near-degeneracy in the asymmetric contracted eig — emits a
rate-limited `RuntimeWarning` suggesting `dtype=float64` (already the default).

---

## What each config flag unlocks (at a glance)

| To get… | Set |
|---------|-----|
| skewness/kurtosis keys (3rd/4th cumulants) | `order ≥ 3` / `order = 4` (default) |
| `standardized_cumulant_spectrum`, `normalized_shc`, `log_det`, Fisher zero | always (order ≥ 3 / ≥ 2) |
| unbiased vs plug-in cumulants | `cumulant_estimator` ∈ {`unbiased` (default), `biased`} |
| `total_cumulant_5…P` (general scalar cumulants) | `order = 4` + `max_scalar_cumulant_order = P` |
| `principal_cumulant_{p}` (per-mode, p>4) | `principal_cumulant_order = P` |
| rank- vs full-padded SHC | `shc_mode` ∈ {`rank` (default), `full`} |
| linear/poly detrending of the window | `detrend_order = 1` (or higher) |
| `align_cov_skew` / `align_cov_kurtosis`, `autocorr_*` | always (order-gated) |
| `inverse_participation_ratio`, `block_energy`, `subspace_align_*` | `compute_eigenvectors=True` |
| `covariance` / `contracted_third_cumulant_matrix` / `contracted_fourth_cumulant_matrix` | `materialize_matrices=True` |
| `group_*`, `block_energy` | `per_group=True` (+ `group_labels`) |
| surrogate `<scalar>/null_*`, MP `mp_edge_upper`/`n_spikes`/… | `null_model` ∈ {`surrogate`, `mp`, `both`} |
| `active_probe_mask*`, `active_probe_count` (§6) | `compute_eigenvectors=True` + `null_model` ∈ {`mp`, `both`} |
| `mode_permutation`, `crossing_detected`, `principal_autocorr_*` | `track_modes=True` (+ `compute_eigenvectors`) |
| `spectral_drift`, `fisher_rao_distance` | any classical-flavor distance (always emitted with them) |
| `denoised_spectral_drift` / `spectral_drift_snr` / `signal_energy_drift` | + `null_model` on, `distance_source="covariance"` |
| `quantum_fidelity` / `quantum_trace_distance` / `bures_distance` / `bures_angle` | `distance_flavor="quantum"`/`"both"` + `materialize_matrices=True` |
| `cka_similarity` / `cka_distance` | `compute_cka_distance=True` + `compute_eigenvectors=True` |
| `<scalar>/se` error bars | `error_bars="jackknife"` |
| `excess_loss` | pass `reference_loss` |

Full field reference: [configuration.md](configuration.md).
