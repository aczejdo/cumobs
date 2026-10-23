# Extending `cumobs`

The package ships v1 with several **seams** — places designed for extension that
are documented (and sometimes partially wired) but not built. They come from §9
of the [spec](../cumulant_observables_spec.md). Two hooks are already live and
usable today; the rest are clearly marked.

> **v0.3 scope note.** The extension batch implemented §1–§8 of
> `cumobs_extension_spec.md` (standardized cumulants, geodesic distances, the
> null model, CSD autocorrelation, the BBP bitmask, cross-operator alignment,
> eigenvector tracking). The **§9 heavier build-outs remain seams** — the
> `GaussianSource` (forward-only Gaussian 2RDM, which would make the §3b
> Marchenko–Pastur null *exact*), an offline CCF / lead-time analyzer, and the
> backprop-based curvature / LLC probes — and are deliberately de-scoped here.

---

## Live hooks (usable now)

### Injectable contraction direction (`contraction`) — §9.1

The "total" cumulants contract each cumulant tensor with the all-ones direction
(`S = B_c · 1_n`). For non-loss features (activations) you may want a different
aggregate — sum over a probe's units, a learned direction, or none. Pass a
`(n,)` `contraction` tensor to `compute_all`, `build_intermediates`, or the
`CumulantMonitor` constructor:

```python
direction = torch.ones(n) / n                 # e.g. a normalized mean
meas = compute_all(B, cfg, contraction=direction)
# or, for the whole monitor:
monitor = CumulantMonitor(model, probes, cfg, contraction=direction)
```

Default (`None`) is all-ones — the loss-semantics "total" direction. This is the
*one* place the math assumes loss semantics, so making it injectable is what lets
the same stack consume activation features (next).

### Reference-loss / excess-loss hook (`reference_loss`) — §9.4

Supply the converged-model loss `ℓ(θ*)` (a `(n,)` tensor) and `compute_all`
emits `excess_loss = Σ_i (E_L[i] − ℓ*(θ*)[i])`, the summed probe loss in excess
of the reference, which reuses the mean `E_L` (so it is "effectively free if
computing the 2RDM"). The inverse temperature `β` is absorbed by you. (This is a
loss-gap diagnostic — renamed from `trajectory_llc`, since it is *not* a Watanabe
local learning coefficient.)

```python
monitor = CumulantMonitor(model, probes, cfg, reference_loss=loss_at_theta_star)
```

---

## Activation / representation cumulants — §9.1 (not built)

The observable core is agnostic to what a "feature" is: it consumes any `B ∈
ℝ^{m×F}`. To monitor **activations** instead of losses, supply a feature
extractor that returns a flat `(F,)` vector per probe (e.g. flattened `(sample ×
unit)` activations or a per-layer pooled vector), feed that into a window, and
call the same observable stack — choosing a `contraction` direction (above) that
matches your aggregate semantics.

The extraction itself is **not** implemented in v1. The recommended shape (from
the spec) is a `FeatureExtractor(model, probe) -> ℝ^F` you write, plugged in
where `ProbeSet.evaluate` currently returns a loss vector.

---

## Gaussian / blended sampling — §9.2 / decision D5 (seam)

`GaussianSource(sigma, n_samples)` exists but `collect` raises
`NotImplementedError`. The intended v1 behavior, documented in its docstring:

> draw `k` i.i.d. `N(0, σ²)` perturbations of the model parameters, evaluate the
> probe loss under each, restore the parameters, and return the resulting `(k,
> n)` buffer immediately (no windowing).

This is the **Gaussian 2RDM**, useful very early in training when you don't yet
have enough trajectory samples for the dynamical 2RDM. Because the observable
math is **source-agnostic**, implementing it is purely a matter of filling the
buffer — `compute_all` is unchanged. A future `BlendedSource` would implement
the eq. 11 schedule `C(t) = (1−w)·C_gauss + w·C_traj` with `w(t) = min(1,
(t−W)/W)`.

> **Why it matters (R4 grounding).** `(JJᵀ)_ij = ℓ'_i ℓ'_j Θ_ij`, so the
> Gaussian-source `C` is a loss-derivative-reweighted empirical **NTK Gram** — a
> principled shadow of the parameter cloud, which legitimizes the whole 2RDM
> object. Under the Gaussian source (weights Gaussian by fiat) the connected
> `κ₄` isolates the **loss-curvature** contribution, separating it from
> finite-width function non-Gaussianity — the clean reading for the kurtosis
> observables. It also makes the §3b Marchenko–Pastur null **exact** (iid rows ⇒
> exact Wishart) and uncaps the rank to `min(n, k)` (R3).

To build it: subclass / replace `GaussianSource.collect` to perturb
`model.parameters()`, call `probeset.evaluate(model)` `n_samples` times, restore
the original parameters, and return the stacked `(k, n)` tensor. Pass the
instance as `source=` to a single-`ProbeSet` `CumulantMonitor`.

---

## Quantum distances on stored matrices — §9.3 (built)

`quantum_fidelity` and `quantum_trace_distance` are **implemented** (decision D2)
— they are eigenvector-sensitive distances on the normalized density matrix `ρ =
C/tr C`. They are listed here because they require the materialized covariance:
set `materialize_matrices=True` and `distance_flavor="quantum"` (or `"both"`),
with `distance_source="covariance"` (only the PSD covariance is a valid density
matrix). The monitor wires them automatically when those flags are set.

---

## Transition-aware distance baseline — decision D4 (TODO)

`distance_reference="baseline"` currently uses the **first** measurement as the
fixed reference. The spec calls for a baseline that **re-bases when a phase
transition is detected** (e.g. an SHC spike), so distances measure drift *since
the last regime change*. The hook is `CumulantMonitor._reference_for`; the
detection-and-swap logic is the documented TODO. You can approximate it today by
constructing a fresh monitor (or resetting reference history) after you detect a
transition in your loop.

---

## Where to plug in

| Extension | Touch point |
|-----------|-------------|
| Activation features | replace `ProbeSet.evaluate`'s loss with a feature vector; set `contraction` |
| Custom aggregate | `contraction=` on `compute_all` / monitor |
| Gaussian sampling | implement `GaussianSource.collect`; pass `source=` |
| Custom logger | implement the `Logger.log` protocol; pass `logger=` |
| Custom reference policy | extend `CumulantMonitor._reference_for` |
| Trajectory-LLC | `reference_loss=` on the monitor / `compute_all` |
