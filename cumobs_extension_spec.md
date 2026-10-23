# cumobs extension spec (v0.3 proposal)

Handoff spec for implementing a batch of new observables, plus a set of
re-groundings and correctness fixes, in `cumobs`. Written to be actioned
directly. It follows the `docs/observables.md` table format
(key / bucket / shape / formula / gating / **cost** / interpretation) and adds a
window-space cost column. Section numbers in *italics* point at the existing
`cumulant_observables_spec.md` / `docs/`.

The throughline: almost everything below is a near-free byproduct of linear
algebra the package already runs (`eigh(G₀)` for `C`; the asymmetric `eig` of
`G3`/`G4` for the contracted spectra). Where something is genuinely heavier
(quantum/Bures operator functions, the Gaussian source), it is tiered out.

---

## 0. Conventions and one corrected definition

Reuse the existing notation: `m` window rows, `n` probes, `B_c` the
centered/detrended buffer, `S = B_c · 1_n` the aggregate, `λ` the covariance
eigenvalues, `p = λ/Σλ`, `P` (`m × top_k`) principal projections, `V` (`n ×
top_k`) sample-space eigenvectors, `G₀ = B_c B_cᵀ`, `r = rank(C)`.

Contraction operators (already in the spec, *§3.5*):
- `G3 = diag(S/D) · G₀`,  partner of `M3 = B_cᵀ diag(S/D) B_c`.
- `G4 = Q · G₀`,  partner of `K = B_cᵀ Q B_c`, with
  `Q = diag(S²/D) − (‖S‖²/D²)·I − (2/D²)·S Sᵀ`.

**Corrected single divisor (decision D1, see §R1):**

```
D = m − (ddof + detrend_order)        # was m − ddof; reduces to m − ddof at detrend_order = 0
```

**Effective sample count** (used by the null model, the noise floor, and the BBP
edge — §3, §4, §6):

```
m_eff = (m − (detrend_order + 1)) / τ_int          # τ_int = integrated autocorr time of S (§5)
```

`m_eff`, not `m`, is the honest sample size for any RMT/Wishart calibration on
the dynamical source, because consecutive rows are autocorrelated.

---

## 1. Standardized cumulant spectrum (the order profile of non-Gaussianity)

`wick_breakdown_ratio` answers "does the 4-point function factorize to 2-point
(Gaussian)." The general version of *which orders of fluctuation are nonzero* is
the **standardized cumulant spectrum**

```
γ_p = κ_p / κ₂**(p/2)        for p = 3, 4, …, P
```

zero at every order for a Gaussian window; the first nonzero `γ_p` is the leading
non-Gaussianity (odd `p` = asymmetry, even `p` = tail/peakedness). This single
vector subsumes the ad-hoc skewness/kurtosis rows from the previous draft (`γ₃` is
classical Fisher skewness, `γ₄` is excess kurtosis, `γ₄/3 = wick_breakdown_ratio`).

**Almost all of this is reconstruction, not new state.** The `γ_p` are pure
transforms of cumulants the package already tracks (`total_*_cumulant` +
`total_cumulant_5…P` for the aggregate; `principal_*_cumulant` per mode), so they
carry no information the cumulants don't. The M→N "does order M reduce to order N"
residuals are likewise fixed functions of the stored cumulants (§1b). The **one
genuinely new primitive** is per-mode cumulants beyond order 4, which the package
does not currently compute — needed only if you want the per-mode spectrum past
kurtosis. Everything else here is a derived row, listed because it's the primary
readout, computed on demand (consistent with the eventual online/materialize-on-
condition design).

| Key | Bucket | Shape | Formula | Gating | Cost | Kind |
|-----|--------|-------|---------|--------|------|------|
| `standardized_cumulant_spectrum` | vectors | `(P−2,)` | `γ_p = κ_p(S)/κ₂(S)**(p/2)`, `p=3…P` | order ≥ 3 (`P = max_scalar_cumulant_order`) | O(P) | derived (transform of `total_*` cumulants) |
| `principal_standardized_cumulant` | vectors | `(top_k, q−2)` | `κ_p(P_a)/λ_a**(p/2)`, `p=3…q` | order ≥ 3 | O(top_k·q) | derived for `q ≤ 4`; **needs new primitive for `q > 4`** |
| `normalized_shc` | scalars | scalar | `Var(λ)/mean(λ)**2` over nonzero λ `= r/PR − 1` | always | O(1) | derived (function of `PR`, `r`) |


Notes for the implementer:
- Clip denominators with the same `eps` as `wick_breakdown_ratio`; `γ_p` blows up
  for tiny `κ₂`/`λ`. Keep `float64` (the recurrence cancels badly at high `p`).
- `normalized_shc = r/PR − 1` is exact (PR is Rényi-2 entropy, `r` the rank) — the
  thing the EM-section footnote ("normalize to remove this effect") asks for; log
  it as a convenience, don't treat it as independent of `PR`.
- A one-scalar reduction, `leading_nongaussian_order` = smallest `p ≥ 3` whose
  `|γ_p|` clears its null/SE band (§3/§5), is also reconstruction — compute from
  the spectrum on demand, don't track it.
- `principal_cumulant_p` is the only thing here that adds state: `P_a` is
  length-`m`, so it's the same cheap recurrence used for the aggregate, applied
  per mode. Gate it behind its own order so the default (≤4) cost is unchanged.

---

## 1b. Factorization residuals — reconstruct, never track a grid

You do **not** need an M→N grid. For a centered variable (`κ₁ = 0`), the residual
of the M-point function after the most generous reconstruction into blocks of size
`≤ N` is

```
R(M,N) = m_M − Σ_{partitions of M, all blocks ≤ N} Π_B κ_{|B|}
       = Σ_{partitions of M, some block > N}     Π_B κ_{|B|}
```

a fixed combinatorial function of the cumulants already stored. The purest case is
`R(M, M−1) = κ_M`. Concretely, the four cases that motivated this:

- **2→1:** `R(2,1) = κ₂` — the covariance / 2RDM itself. There is no Wick *ratio*
  here: 2nd order is the Gaussian baseline, and the disconnected piece (product of
  means) is stripped by centering and kept separately as `first_cumulant_vector`.
  The dimensionless "independent vs correlated probes" question is answered by
  `PR` / `top_eigenvalue_fraction` / off-diagonal Frobenius energy of `C`.
- **3→2 = 3→1:** `R = κ₃` → `γ₃` (`standardized_cumulant_spectrum[0]`). The
  `κ₂κ₁` partition vanishes, so 3→2 and 3→1 coincide.
- **4→3 ≡ 4→2:** `R = κ₄` → `wick_breakdown_ratio` / `γ₄`. **4→3 is not a separate
  quantity** — `m₄` has no `κ₃` term for a centered variable (the `{3,1}`
  partition carries `κ₁ = 0`).
- **4→2:** `R = κ₄`, i.e. `wick_breakdown_ratio = κ₄/(3λ²)`.

M→N becomes a genuine *combination* (rather than a single cumulant) only at
order ≥ 5 with `N < M−1`, e.g. `R(6,3) = κ₆ + 15κ₄κ₂`, `R(6,2) = κ₆ + 15κ₄κ₂ +
10κ₃²`, `R(5,2) = κ₅ + 10κ₃κ₂`. These are reconstructed from stored cumulants,
never cached.

**Helper to expose** (instead of any tracked residual):

```
factorization_residual(cumulants: dict[int, float], M: int, N: int) -> float
    # R(M,N) via partition enumeration of M (M small); singleton blocks contribute 0.
```

Moment ↔ cumulant both directions reuse the recurrence already in `concepts.md`
(`m_p = Σ_{k=1}^{p} C(p−1,k−1) κ_k m_{p−k}`, `m₀=1`, `κ₁=0`).

---

## 1c. Partition-function (Fisher / Lee–Yang) zero

A diagnostic built from the same aggregate cumulants. The paper's free energy *is*
the cumulant generating function of the total loss (`log Z = Σ_p κ_p t^p/p!`,
*App. cumulant expansion*). Truncating the partition function itself —

```
Z_P(t) = Σ_{p=0}^{P} m_p t^p / p!,      m_p reconstructed from {κ_2…κ_P}, m₀=1, m₁=0
```

— gives a degree-`P` polynomial whose complex zeros are the **Fisher / Lee–Yang
zeros**. The zero nearest the real axis controls the analytic structure of the
free energy: its imaginary part shrinking toward zero is the partition-function
signature of an approaching transition (critical slowing down read off `Z` rather
than off the spectrum). Because `Z` is built from `S`, whose variance
`Var(S) = total_second_cumulant = 1ᵀC1` is the *true* heat-capacity object (§R2),
this complements SHC: SHC is condensation in the eigenvalue spectrum, the Fisher
zero is criticality in the energy variable.

| Key | Bucket | Shape | Formula | Gating | Cost | Kind |
|-----|--------|-------|---------|--------|------|------|
| `leading_fisher_zero` | scalars (complex) | scalar | leading root of `Z_P(t)` (def. below) | order ≥ 2 | O(P³) | derived (transform of `total_*` cumulants) |
| `fisher_zero_gap` | scalars | scalar | distance of `leading_fisher_zero` to the positive real axis | order ≥ 2 | O(1) | derived; the early-warning scalar (`→ 0` near a transition) |

**`root_order` (new config field) resolution.** Let `P_tot = max_scalar_cumulant_
order if set else cfg.order` (the highest available "total"/aggregate cumulant
order).

```
root_order_default  = min(4, P_tot)                 # the "fourth order" default
root_order_eff      = min(root_order, P_tot)         # clamp on override …
if root_order > P_tot: warn(f"root_order {root_order} > max total cumulant order {P_tot}; using {P_tot}")
root_order_eff      = max(2, root_order_eff)          # need ≥2 for a nontrivial polynomial
```

**Construction and leading-root selection.**
1. Build `m_p` for `p = 2…root_order_eff` from the stored aggregate cumulants
   (recurrence above; `m₀=1`, `m₁=0` by centering).
2. For conditioning, root in the **standardized** variable `τ = t·√κ₂`: coefficients
   become `m_p/κ₂**(p/2)` (`= 1` at `p=2`, `γ₃` at `p=3`, `γ₄+3` at `p=4`, …), so
   the polynomial depends only on the shape `γ_p`. Recover `t = τ/√κ₂`.
   Use `numpy.roots` / `numpy.polynomial`, `float64`.
3. Roots come in complex-conjugate pairs (real coefficients), so keep the
   upper-half-plane representative of each pair. Among them pick the **leading**
   root: the one minimizing distance to the **positive** real axis,
   `d(t) = |Im t|` if `Re t ≥ 0` else `|t|`. Return that single complex `t`.
4. `fisher_zero_gap = d(leading_fisher_zero)`. Return `NaN`/`None` for a dead
   window (`κ₂ ≈ 0`).

Interpretation notes:
- To leading order `|Im(leading_fisher_zero)| ≈ √(2/κ₂)`, so the *gap* mostly
  tracks `1/√(total_second_cumulant)`; the **genuinely new** content is the
  **phase** `arg(leading_fisher_zero)`, which is switched on by `κ₃` (asymmetry
  tilts the zero off the imaginary axis) and dressed by `κ₄`. Report the complex
  number; read the phase for shape, the gap for criticality.
- Sign convention: "positive real axis" follows the source `t` (the paper uses
  `t = −β`, so flip the sign sector if you want the physical `β > 0` half). What is
  physical is proximity to the real axis; the half-axis choice is a convention,
  documented so it can be flipped.
- Serialization: `scalars` is float-typed, so a complex value needs handling like
  the `active_probe_mask` int — emit `leading_fisher_zero_re` / `_im` for flat
  loggers (and optionally `_abs` / `_arg`); `fisher_zero_gap` is a plain float.

---

## 2. Geodesic distances (free transforms of existing fidelities)

These are the Riemannian geodesic forms of the two fidelities already computed.
They are one-line transforms (no new linear algebra) and should be emitted
*instead of* the bare fidelities by default (the fidelity is recoverable and the
metric form respects the triangle inequality — see the distance consolidation in
§8).

| Key | Bucket | Formula | Range | Gating | Cost | Interpretation |
|-----|--------|---------|-------|--------|------|----------------|
| `fisher_rao_distance` | scalars | `2·arccos( clamp(√classical_fidelity, 0, 1) )` | `[0, π]` | classical/both flavor | O(n) | geodesic on the simplex of normalized spectra; proper metric |
| `bures_distance` | scalars | `√( max(0, 2(1 − √quantum_fidelity)) )` | `[0, √2]` | quantum flavor + `materialize_matrices` + `distance_source="covariance"` | O(n³)\* | Bures = geometry whose geodesic is Uhlmann fidelity |
| `bures_angle` | scalars | `arccos( clamp(√quantum_fidelity, 0, 1) )` | `[0, π/2]` | same as `bures_distance` | O(n³)\* | operator analog of `fisher_rao_distance` |

\* No extra cost over `quantum_fidelity`, which already does the `O(n³)` work.

Cross-check / consistency (worth a docstring line): when `ρ, σ` commute,
`quantum_fidelity → classical_fidelity` and `bures_angle = ½ · fisher_rao_distance`.
So Bures is the eigenvector-sensitive generalization of Fisher–Rao, giving a
matched spectrum-level / operator-level pair.

Caveat: `fisher_rao_distance` must inherit `classical_fidelity`'s
negative-mass clip-and-warn on signed (contracted) sources (§R6).

---

## 3. Null-background model for the spectrum (the biggest detector gap)

A parameter-light null answers "is this mode/jump distinguishable from no
co-fluctuation," which neither the SHC threshold nor the jackknife SE does. Two
implementations; ship the bootstrap as default, the analytic MP as the companion
that also yields the BBP edge for §6.

### 3a. Surrogate (block bootstrap) — default, assumption-light

Per measurement, generate `B_null` surrogates by **independent circular shifts of
each column of `B_c`** (destroys cross-probe phase alignment = the 2RDM signal,
preserves each probe's marginal *and* its within-column autocorrelation, so the
aspect ratio stays honest). Re-run the spectral path (reuse `build_intermediates`)
on each. For every scalar observable emit:

| Key | Bucket | Formula | Gating | Cost |
|-----|--------|---------|--------|------|
| `<scalar>/null_mean` | scalars | mean over surrogates | `null_model="surrogate"` | `B_null` × scalar cost |
| `<scalar>/null_sd` | scalars | sd over surrogates | same | — |
| `<scalar>/null_z` | scalars | `(obs − null_mean)/null_sd` | same | — |

This is the principled form of the EM-section "could this be generic training
noise?" worry, and it answers a *different* question than the existing
`<scalar>/se` jackknife (precision of the estimate vs distance from the
no-transition baseline). Pair them; the null subsumes the jackknife's intended
detection role.

### 3b. Marchenko–Pastur / Tracy–Widom — analytic companion

With aspect `γ = n / m_eff` and a **robust bulk noise level** `σ̂²` (match the MP
median to the empirical sub-edge median so spikes don't inflate it):

| Key | Bucket | Formula | Gating | Cost | Interpretation |
|-----|--------|---------|--------|------|----------------|
| `mp_edge_upper` | scalars | `σ̂²·(1 + √γ)²` | `null_model="mp"` | O(n) | bulk edge; BBP detachment threshold |
| `n_spikes` | scalars | `#{ λ_a > mp_edge_upper + c·tw_sd }` | same | O(n) | detached modes; `tw_sd ∝ σ̂²·m_eff^(−2/3)` (Tracy–Widom scale) |
| `spike_energy_fraction` | scalars | `Σ_{λ_a > edge} λ_a / Σλ` | same | O(n) | fraction of variance above the noise sea |
| `mp_bulk_ks` | health | KS distance of sub-edge spectrum to MP density | same | O(n) | null goodness-of-fit (large ⇒ MP inadequate) |

The Tracy–Widom band sets the threshold the earlier "count above the edge" left
unspecified: flag `λ₁` only when it clears the edge by a few `tw_sd`. MP/TW are
only approximate on the dynamical source (autocorrelated rows → use `m_eff`); the
Gaussian source (§9) makes them exact (iid rows ⇒ exact Wishart).

Synergy: `null_z` (or `n_spikes`) crossing threshold is the principled trigger
for the documented D4 transition-aware baseline swap.

---

## 4. Noise-floored distances

A distance whose zero point is "indistinguishable from noise."

| Key | Bucket | Formula | Gating | Cost | Interpretation |
|-----|--------|---------|--------|------|----------------|
| `denoised_spectral_drift` | scalars | `max(0, (spectral_drift − μ₀)/(1 − μ₀))` | needs §3a on the drift | `B_null` × O(m³) | drift with the null floor subtracted, renormalized so 1→1 |
| `spectral_drift_snr` | scalars | `(spectral_drift − μ₀)/s₀` | same | — | drift in units of its null sd |
| `signal_energy_drift` | scalars | drift computed on `E_sig = Σ_{λ>λ₊}(λ − λ₊)` only | needs §3b (`mp_edge_upper`) | O(n) | analytic floor: sub-edge bulk contributes exactly zero |

`μ₀, s₀` are the null mean/sd of `spectral_drift` between independent surrogates.
Ship `denoised_spectral_drift` as the default (no Gaussianity assumption);
`signal_energy_drift` when MP is on.

---

## 5. Critical slowing down (autocorrelation) — nearly free, regime-correct

The paper's whole early-warning story is "diverging variance **and** rising
autocorrelation" (*def:csd*, critical slowing down). The package implements only
the variance half (SHC). This adds the autocorrelation half, which is a genuine
*new leading indicator*, and also produces `τ_int` for `m_eff`.

| Key | Bucket | Shape | Formula | Gating | Cost | Interpretation |
|-----|--------|-------|---------|--------|------|----------------|
| `autocorr_lag1` | scalars | scalar | `ρ₁(S)` | always | O(m) | lag-1 autocorr of aggregate; **rising = CSD early warning** |
| `autocorr_time` | scalars | scalar | `τ_int(S) = 1 + 2 Σ_{k≥1} ρ_k(S)` (Sokal auto-window) | always | O(m) | integrated autocorr time of `S` |
| `effective_window` | health | scalar | `(m − (detrend_order+1)) / τ_int` | always | O(1) | honest sample size; warn when `τ_int ≫ 1` |
| `principal_autocorr_lag1` | vectors | `(top_k,)` | `ρ₁(P_a(t))` across measurements | `compute_eigenvectors` + tracking (§7) + history buffer | O(m·top_k) | per-mode CSD; sharper target than `S` |
| `principal_autocorr_time` | vectors | `(top_k,)` | `τ_int(P_a(t))` | same | O(m·top_k) | per-mode relaxation proxy; pairs with `spectral_gap` |

Implementation: `autocorr_*` on `S` is purely 1-D and tracking-free — make it
always-on. The per-mode versions need a short ring buffer of `P_a` across
measurements and depend on eigenvector tracking (§7) to keep mode identity stable
across crossings. Also: inflate the jackknife `<scalar>/se` to a block jackknife
by `√τ_int` for honest bars on autocorrelated rows.

---

## 6. BBP spike-detachment bitmask (which probes are participating, as an integer)

Formalizes the *§app:sensitivity* recipe (`S_k = {i : |v_k(i)|² > τ}`) with a
detachment gate so only genuine spikes are encoded. To leading order "eigenvalue
above the MP edge" *is* the BBP criterion. Requires `compute_eigenvectors` and
the §3b edge.

For each `λ_a > mp_edge_upper`, build the support as the **smallest set capturing
a target energy** (more robust than a flat threshold; the effective support is
already `≈ 1/IPR[a]`):

```
S_a = smallest index set with  Σ_{i ∈ S_a} v_a[i]² ≥ ρ_keep        # e.g. ρ_keep = 0.9
```

| Key | Bucket | Shape | Formula | Gating | Cost | Interpretation |
|-----|--------|-------|---------|--------|------|----------------|
| `active_probe_mask` | scalars (int) | `(n_spikes,)` | `Σ_{i ∈ S_a} 2**i` | `compute_eigenvectors` + §3b | O(n log n) | per detached mode: bit `i` set ⇔ probe `i` participates |
| `active_probe_count` | scalars | `(n_spikes,)` | `|S_a|` | same | O(n) | participating probe count per mode |
| `active_probe_mask_any` | scalars (int) | scalar | OR over detached `a` | same | O(n) | union support across all detached modes |

Implementer notes:
- Bit `i` ↔ probe `i` in `ProbeSet` order, LSB = probe 0. D6 sign gauge is
  irrelevant (we square). The mask is a Python big-int; **serialize as a hex
  string for JSONL logging** (the `scalars` bucket is float — this needs either a
  new int/string field or hex in `meas.health`-style storage; flag the type
  mismatch when wiring the logger).
- With `group_labels`, decode the mask → which *groups* light up; this is the
  Fig.1 localization result as one decodable integer.

---

## 7. Cross-operator alignment among {C, M3, K} (the headline new layer)

Asks: does the skew (`M3`) or connected-kurtosis (`K`) structure live *where* the
variance (`C`) is condensing? Both the scalar and the subspace forms are
**byproducts of computations the package already runs** (see costs). This is the
single highest-value addition.

Window-space identity (the reason it's free). For any `O = B_cᵀ W B_c` with Gram
partner `G_O = W G₀`:

```
tr(C·O) = (1/D) tr(G_O G₀) = (1/D) tr(W G₀²)          # cyclic invariance
```

so the normalized cosine has the `1/D` cancel:

```
cos(C, O) = tr(G_O G₀) / ( √tr(G₀²) · √tr(G_O²) )
```

### 7a. Scalar alignment — always-on

| Key | Bucket | Formula | Gating | Cost | Interpretation |
|-----|--------|---------|--------|------|----------------|
| `align_cov_kurtosis` | scalars | `tr(G4 G₀)/(√tr(G₀²)·√tr(G4²))` | order = 4 | O(m²) marginal\* | signed; is non-Gaussianity co-locating with variance condensation |
| `align_cov_skew` | scalars | `tr(G3 G₀)/(√tr(G₀²)·√tr(G3²))` | order ≥ 3 | O(m²) marginal\* | signed; does skew structure sit on the condensing subspace |

\* `Q` is `diag + I + rank-1`, so `G4 = Q G₀` costs O(m²) given `G₀`; the traces
are O(m²). If the contracted spectra are already being emitted, `G3`/`G4` exist
and this is essentially free.

Grounding caveat (do **not** overclaim): this is *not* the paper's
`⟨C⁽¹⁾, Δ⟩_F` from `thm:spectral_hc`. `Δ` is the Isserlis/Gaussian part of the
quartic covariance; `K` is the connected residual the Isserlis step discards. Sell
`align_cov_kurtosis` as "non-Gaussianity co-locating with variance condensation,"
which is what it measures — the link to the early-warning theorem is analogy.

### 7b. Subspace alignment — online (recovered eigenvectors, no dense matrix)

The eigenvectors `w` of the asymmetric partners are already computed when the
contracted spectra are emitted; the package currently discards them. Recover the
sample-space eigenvectors for free via the same pushforward used for `C`:

```
if  G3 w = μ w   then   M3 (B_cᵀ w) = μ (B_cᵀ w)      # so v = B_cᵀ w is an M3 eigenvector
if  G4 w = μ w   then   K  (B_cᵀ w) = μ (B_cᵀ w)      # likewise for K
```

(`M3`, `K` are symmetric ⇒ real eigenvalues; the asymmetric `eig` gives small
imaginary parts in finite precision ⇒ take `.real`, watch the existing
`contracted_*_imag_ratio` health metric near degeneracy.)

| Key | Bucket | Shape | Formula | Gating | Cost | Interpretation |
|-----|--------|-------|---------|--------|------|----------------|
| `subspace_align_cov_kurtosis` | vectors | `(top_k,)` | principal angles between top-`k` `V_C` and `V_K = orthonormalize(B_cᵀ W_K)` | order = 4 + `compute_eigenvectors` | O(mnk + nk²) marginal | per-mode overlap of variance vs kurtosis subspaces |
| `subspace_align_cov_skew` | vectors | `(top_k,)` | principal angles, `V_C` vs `V_{M3}` | order ≥ 3 + `compute_eigenvectors` | O(mnk + nk²) marginal | per-mode overlap of variance vs skew subspaces |

Because the `eig` of `G3`/`G4` is already paid by the contracted-spectrum path,
the only new work is `V = B_cᵀ W` (O(mnk)), a QR (O(nk²)), and a `k×k` principal-angle
SVD. So the entire cross-operator layer — scalar and subspace — is online; nothing
here needs `materialize_matrices`. (This resolves the open "recover M3/K top-k
eigenvectors in window space" question and collapses the middle materialization
tier in §8.)

---

## 8. Consolidations (default-logging changes, not deletions)

Keep everything computable; change the *default emitted set* for parsimony.

**Concentration scalars** (all Schur-monotone functionals of `p`):
- Drop `von_neumann_entropy` from the default set — `effective_rank =
  exp(von_neumann_entropy)` exactly; log one.
- **Keep both `PR` and `effective_rank`** — they are different Rényi orders
  (`PR = exp H₂`, `effective_rank = exp H₁`), `effective_rank ≥ PR` with equality
  only at flat/rank-1 spectra, and their ratio is a spectrum-tail descriptor. Not
  redundant.
- **Keep and mildly elevate `spectral_gap`**, but grounded as a *condensation /
  one-mode-detaching* parameter, **not** a relaxation time (relaxation time lives
  in the Hessian / trajectory autocorrelation, not in a covariance). Once §3b is
  in, prefer the edge gap `λ₁ − mp_edge_upper`.

**Distances** — three axes; default to one representative per axis, in metric form:
- Shape (normalized spectrum): emit `fisher_rao_distance` (Riemannian geodesic;
  subsumes `classical_fidelity`). Optionally keep `trace_distance` (L1/TV — a
  different, cheaper geometry, not redundant with Fisher–Rao).
- Scale (raw spectrum): `spectral_drift` (+ the noise-floored variants, §4).
- Subspace rotation: `cka_distance` (O(nk²)).
- Operator (opt-in, eigenvector-sensitive, dense): `bures_distance` /
  `bures_angle` (subsumes `quantum_fidelity`) and `quantum_trace_distance`. Do
  not also emit `quantum_fidelity` standalone.
- Stop emitting bare `classical_fidelity` / `quantum_fidelity` by default — each
  is recoverable from its geodesic metric, which is the more useful form. (They
  are related to TV by inequalities, not deterministic transforms, so this is
  parsimony, not dedup of identical bits.)

**Higher scalar cumulants** `total_cumulant_5/6…` — drop from the monitored set;
keep the exported `total_scalar_cumulants` routine as an on-demand diagnostic.

---

## 9. Heavier build-outs (tiered out; separate work items)

- **GaussianSource (decision D5).** Highest-priority larger build. It is
  **forward-only** (`k·n` forward passes per measurement via isotropic
  perturbations realizing `C = σ²GGᵀ` — costly but no autodiff, so it stays inside
  the package's design philosophy). It fixes *validity* of the existing detector:
  estimable higher cumulants, **exact-Wishart RMT calibration** (§3b becomes
  exact), **uncapped rank** (see §R3), and coverage toward initialization. Wire
  the documented blend `C(t) = (1 − w(t))·C_gauss + w(t)·C_traj`,
  `w(t) = min(1, (t − W)/W)`. Do **not** claim it recovers the Lyapunov
  stationary cloud `Σ_param` — that is what the *dynamical* source approximates
  near stationarity; the Gaussian source with `Σ_w = σ²I` is deliberately
  optimizer-free landscape geometry.
- **Timing / cross-correlation tooling.** Offline analyzer over the logged scalar
  series: CCF between two streams, peak-lag with a bootstrap CI, peak/changepoint
  detection. This is what reproduces the paper's *headline* lead-time figures
  (the CCF overlays, the `−765 ± 489` lag), which the per-step scalar API cannot
  currently do. High value, low math risk.
- **Curvature / LLC probes — OUT OF SCOPE for this package.** A Hessian spectrum
  (Hutchinson/Lanczos HVP) and an SGLD local learning coefficient are genuine
  framework order parameters, but they need **backprop**, so they belong in a
  sibling *gradient-based* tool, not in the forward-only stack. Note the ambition;
  keep it de-scoped the way `excess_loss` correctly is (it is a loss-gap, not a
  Watanabe LLC).

---

## R. Re-groundings and correctness fixes

**R1 — Detrending degrees of freedom (correctness).** Detrending degree `d` is a
rank-`(d+1)` projection, so for Gaussian residuals `E[RSS] = σ²(m − (d+1))`.
Dividing by `m − ddof` biases the variance (and the whole covariance, since the
detrend design is shared across columns) **downward**. Fix:
`D = m − (ddof + detrend_order)` everywhere (one global `D`, decision D1
preserved; the full-contraction identities are algebraic in `(B_c, D)` and
unaffected). This restores **order-2** unbiasedness, where SHC/PR live. It does
**not** restore the **k-statistic** unbiasedness at orders ≥ 3: the
`m²/((m−1)(m−2))`-type prefactors are derived for the centering projector only, and
the `(I − H)` hat matrix breaks them. So under `detrend_order > 0` either default
to `cumulant_estimator="biased"` (the single-`D` plug-in stays consistent under
any projector) or keep `"unbiased"` and emit a `RuntimeWarning` that orders ≥ 3
are no longer unbiased. Propagation to fix in the same change:
`rank ≤ min(m − (d+1), n)` (update `n_nonzero`/`rank`/`eig_threshold`), and
`shc_mode="rank"` shifts with `detrend_order` on identical data — document that
**both** `ddof` and `detrend_order` must be fixed across compared runs.

**R2 — SHC is not a specific heat (relabel, keep).** Real heat capacity is the
*energy* fluctuation `β²Var(E)`. In the paper's own cluster expansion,
`∂²log Z/∂β² = Var(Σℓ_i) = 1ᵀC1 = total_second_cumulant` — so the literal
heat-capacity analog is `total_second_cumulant`, already in the scalars bucket.
`spectral_heat_capacity = Var(λ)` is a **condensation / anisotropy order
parameter** (it spikes when a mode detaches — which is real and useful). Keep the
name as analogy; drop any fluctuation-dissipation wording from the prose.

**R3 — Rank cap (prominent caveat).** `rank(C) ≤ m − (d+1)` *regardless of `n`*.
At `m = 20` you resolve ≤ 19 modes even with `n = 512` probes, so `PR`,
`effective_rank`, `shc_mode="rank"`, and `normalized_shc` are window-limited and
partly track window length rather than the model. State this wherever a "number
of modes" reading appears. The Gaussian source (`k > n` perturbations) uncaps it
to `min(n, p)`.

**R4 — NTK / kurtosis-isolation notes (grounding for the docs).**
`(JJᵀ)_ij = ℓ'_i ℓ'_j Θ_ij`, so the Gaussian-source `C` is a
loss-derivative-reweighted empirical NTK Gram — the legitimizing fact that the
object is a principled shadow of the parameter cloud. Under the Gaussian source
(weights Gaussian by fiat), `κ₄` isolates the **loss-curvature** (`q_i`)
contribution, separating it from finite-width function non-Gaussianity — the clean
reading for the kurtosis observables.

**R5 — `wick_breakdown_ratio` docstring.** Note it equals `γ₂/3` so it is not read
as independent of the new `principal_excess_kurtosis`.

**R6 — `fisher_rao_distance` on signed sources.** Must inherit
`classical_fidelity`'s clip-negative-mass-and-warn behavior; do not let it run
silently on the contracted (signed) spectra.

**R7 — Quantum framing is analogy, not a literal RDM.** The paper's appendix
states the 2RDM / quantum-chemistry choice is an analogy to surface the
basis-selection problem, not a literal reduced density matrix. Consequence already
baked into §8: the quantum/Bures distances are an expensive way to measure
subspace drift that CKA measures cheaply — reserve them for offline
checkpoint analysis.

---

## Suggested implementation order

1. **Cross-operator eigenvector recovery + alignment (§7).** Smallest
   load-bearing piece; decides always-on vs spike-triggered for the whole layer.
   Verified free given the existing `G₀`/`G3`/`G4` work.
2. **Eigenvector tracking (Hungarian on `|⟨v_a(t), v_b(t+1)⟩|`, subspace-angle
   tracking in near-degenerate blocks; emit `mode_permutation`,
   `crossing_detected`).** Underpins every per-mode series (principal cumulants,
   `block_energy`, per-mode autocorrelation, the BBP bitmask). Crossings become a
   detectable event instead of a corruption.
3. **R1 + propagation (correctness).** Divisor, estimator guard, rank/`shc_mode`.
4. **Cheap transforms + `log_det` (§1, §1c, §2).** Standardized cumulant
   spectrum, `normalized_shc`, the geodesic distances, the Fisher-zero
   (`leading_fisher_zero`/`fisher_zero_gap`), and `log_det = Σ log λ_i` over
   nonzero (with the paper's `ε` jitter, scalars bucket, O(1) — in the paper's
   logged grokking set and currently missing). Add the `factorization_residual`
   helper (§1b) here too. The only new *state* in this batch is the opt-in
   `principal_cumulant_p` (per-mode cumulants > 4); everything else is a transform.
5. **CSD autocorrelation (§5):** `S`-level always-on first, `P_a`-level after (2).
6. **Null model (§3):** surrogate z-scores, then MP/TW edge.
7. **BBP bitmask (§6)** (needs the edge + eigenvectors).
8. **Noise-floored distances (§4)** (need the null).
9. **Consolidation (§8)** default-logging changes.
10. **Heavier (§9):** GaussianSource, then the timing/CCF analyzer.

---

## New / changed keys at a glance

Added: `standardized_cumulant_spectrum`, `principal_standardized_cumulant`,
`normalized_shc`, `principal_cumulant_p` (opt-in new primitive, `p>4`),
`leading_fisher_zero` (+ `_re`/`_im`/`_abs`/`_arg`), `fisher_zero_gap`,
`fisher_rao_distance`, `bures_distance`, `bures_angle`, `denoised_spectral_drift`,
`spectral_drift_snr`, `signal_energy_drift`, `<scalar>/null_mean|null_sd|null_z`,
`mp_edge_upper`, `n_spikes`, `spike_energy_fraction`, `mp_bulk_ks`,
`autocorr_lag1`, `autocorr_time`, `effective_window`, `principal_autocorr_lag1`,
`principal_autocorr_time`, `active_probe_mask`, `active_probe_count`,
`active_probe_mask_any`, `align_cov_kurtosis`, `align_cov_skew`,
`subspace_align_cov_kurtosis`, `subspace_align_cov_skew`, `log_det`,
`mode_permutation`, `crossing_detected`.

New config: `root_order` (Fisher zero, §1c).

Reconstruct-on-demand (helpers, not tracked): `factorization_residual(κ, M, N)`
(§1b), `leading_nongaussian_order` (§1). The standardized cumulant spectrum and
the Fisher zero are themselves transforms of the tracked cumulants — listed as
rows only because they are primary readouts.

Changed: `D = m − (ddof + detrend_order)`; `rank`/`n_nonzero`/`eig_threshold`
account for `(d+1)`; default-logging set trims `von_neumann_entropy`,
`classical_fidelity`, `quantum_fidelity`, `total_cumulant_5+`; the previous
draft's `total_skewness`/`principal_skewness`/`diagonal_skewness`/
`principal_excess_kurtosis` rows are folded into
`standardized_cumulant_spectrum` / `principal_standardized_cumulant`;
`spectral_gap` re-grounded; `spectral_heat_capacity` and `wick_breakdown_ratio`
docstrings re-grounded.
