# `cumobs` examples

Runnable, self-contained scripts. Each adds the project root to `sys.path`, so
run them from the repo root with no install step:

```bash
python examples/01_minimal_observables.py
python examples/02_monitor_training_loop.py
python examples/03_distances.py
python examples/04_quantum_isospectral_rotation.py   # add --plot for the training figure
python examples/05_nongaussianity_and_fisher.py      # v0.3: §1 / §1b / §1c
python examples/06_null_model_and_bbp.py             # v0.3: §3 / §4 / §6
python examples/07_critical_slowing_down.py          # v0.3: §5 / §7 / §2
python examples/fig1_transition.py            # add --plot to save a figure
```

All are deterministic (seeded) and need only `torch` (plus `numpy` for `04` and
`matplotlib` for the optional `04`/`fig1` plots). Examples `05`–`07` cover the
v0.3 extension batch.

| Script | What it shows | Key APIs |
|--------|---------------|----------|
| [`01_minimal_observables.py`](01_minimal_observables.py) | The pure math core on a hand-built buffer — the whole observable catalog printed and interpreted. No model, no monitor. | `compute_all`, `CumulantConfig` |
| [`02_monitor_training_loop.py`](02_monitor_training_loop.py) | The monitor watching a real `nn.Linear` SGD training loop; SHC collapses as the model converges, block energy localizes on the harder probe group. | `CumulantMonitor`, `ProbeSet`, `InMemoryLogger` |
| [`03_distances.py`](03_distances.py) | Classical vs quantum distances, the eigenvector-blindness of the classical ones, and a monitor whose trace distance spikes at a synthetic transition. | `classical_fidelity`, `trace_distance`, `quantum_*`, `CumulantMonitor` |
| [`04_quantum_isospectral_rotation.py`](04_quantum_isospectral_rotation.py) | The **sensitivity ladder**: drive a 2RDM through a scale change, an *isospectral* basis rotation (mode hands off between probe groups at fixed spectrum), and a spectral reshape — and see which observable catches which. The rotation is invisible to SHC/PR/entropy/`classical_fidelity`/`spectral_drift` but lights up `quantum_fidelity`, `cka_distance`, and `block_energy`. `--plot` reruns the changes *continuously* through a sliding-window monitor and saves a trackable training figure ([04_quantum_isospectral_rotation.png](04_quantum_isospectral_rotation.png)) that decomposes any change into amplitude (`spectral_drift`) / spectrum-shape (`trace_distance`) / basis-reorganization (`quantum_TD − classical_TD`) signals. | `quantum_fidelity`, `cka_distance`, `spectral_drift`, `block_energy` |
| [`05_nongaussianity_and_fisher.py`](05_nongaussianity_and_fisher.py) | **(v0.3 §1)** The *order profile of non-Gaussianity*: the `standardized_cumulant_spectrum` (γ₃ skew, γ₄ kurtosis) reads ≈ 0 for a Gaussian window and turns on for a skewed one; `leading_nongaussian_order` names the leading order. The **Fisher/Lee–Yang zero** gap shrinks ∝ 1/√variance (criticality), and `factorization_residual` reconstructs the M→N residuals exactly from stored cumulants. | `standardized_cumulant_spectrum`, `leading_fisher_zero_*`, `fisher_zero_gap`, `factorization_residual`, `leading_nongaussian_order` |
| [`06_null_model_and_bbp.py`](06_null_model_and_bbp.py) | **(v0.3 §3/§4/§6)** The null-background detector: a planted rank-1 mode is highly significant vs the phase-scrambled surrogate null (`SHC/null_z ≫ 0`, `PR/null_z < 0`) and detaches above `mp_edge_upper` (`n_spikes=1`), while pure noise is not. The **BBP bitmask** decodes back to the planted probe group, and `denoised_spectral_drift` sits at the noise floor until a real mode switches on. | `null_model`, `mp_edge_upper`, `n_spikes`, `active_probe_mask`, `denoised_spectral_drift` |
| [`07_critical_slowing_down.py`](07_critical_slowing_down.py) | **(v0.3 §5/§7/§2)** The early-warning half the paper pairs with the variance spike: `autocorr_lag1` *rises* as an AR(1) mode's persistence ramps toward a transition (mean −0.16 → +0.31). Eigenvector **tracking** flags `crossing_detected` when a second mode takes over, alongside the geodesic `fisher_rao_distance` and the `align_cov_skew` cross-operator alignment. | `autocorr_lag1`, `autocorr_time`, `track_modes`, `crossing_detected`, `fisher_rao_distance`, `align_cov_skew` |
| [`fig1_transition.py`](fig1_transition.py) | The paper's Fig. 1 synthetic transition and the §10 test-12 acceptance check: SHC spikes ~10⁹× and the top eigenvector localizes (block energy → 1) on the planted group. | `CumulantMonitor`, `per_group`, `compute_eigenvectors` |

## Suggested reading order

1. **`01_minimal_observables.py`** — see what a `Measurement` contains and how to
   read each bucket (scalars / vectors / matrices / health).
2. **`02_monitor_training_loop.py`** — wire the monitor into a loop with a real
   `loss_fn` and group labels.
3. **`03_distances.py`** — add transition detection between measurements.
4. **`04_quantum_isospectral_rotation.py`** — *why* the catalog has more than one
   distance: what the eigenvector-sensitive (quantum / CKA) observables see that
   the spectral ones cannot.
5. **`05`–`07`** — the v0.3 extension batch: the non-Gaussianity profile + Fisher
   zero (`05`), the null-background detector + BBP bitmask (`06`), and the
   critical-slowing-down early warning + mode tracking (`07`).
6. **`fig1_transition.py`** — the full behavioral acceptance check end-to-end.

## Expected output (sanity check)

- `01` — leading eigenvalue ≈ 7.7, participation ratio ≈ 1.03 (one dominant
  mode), block energy of the top eigenvector on group A ≈ 1.000.
- `02` — `SHC` falls from ~10⁸ to ~10⁻¹⁴ across ~70 steps; `blockE_far` stays
  ≈ 0.98; `fidelity` ≈ 1.
- `03` — part 1 matches the printed closed forms; part 2 shows quantum fidelity
  0.949 < 1 while classical is exactly 1; part 3 shows fidelity dropping to ≈ 0.30
  and trace distance jumping to ≈ 0.50 at the transition step.
- `04` — in the rotation regime `classical_fidelity ≈ 1.000` and `spectral_drift`
  stays at its noise floor while `quantum_fidelity` collapses to ≈ 0.07,
  `cka ≈ 1.0`, and block energy hands off `A: 1→0`, `B: 0→1`. With `--plot`, the
  saved figure shows three cleanly separated bands: `spectral_drift` spikes only
  in the amplitude band, the basis-reorganization signal only in the rotation
  band, and `trace_distance`/`effective_rank` only in the new-mode band.
- `05` — the Gaussian window reads `g3 ≈ -0.04, g4 ≈ 0.15` →
  `leading_nongaussian_order = None`; the skewed window reads `g3 ≈ 1.34` → order
  3 (asymmetry). The Fisher gap halves each time the amplitude doubles (variance
  ×4), and all four factorization residuals print `[ok]`.
- `06` — `signal` row: `SHC null_z ≈ +8.5`, `PR null_z ≈ -4.6`, `n_spikes = 1`,
  `spike_energy ≈ 0.99`; `noise` row: `n_spikes = 0`. The BBP mask `0x35` decodes
  to group `A`. In the noise-floored part `denoised_spectral_drift ≈ 0` for noise
  and jumps to ≈ 0.78 at the transition.
- `07` — mean `autocorr_lag1` rises from ≈ -0.16 (noise) to ≈ +0.31 (approaching
  the transition); eigenvector tracking flags ~6 `crossing_detected` events once
  the two modes alternate dominance.
- `fig1` — "spike factor: ~10⁹×", "final block energy on planted group A: 1.000".

## Where to go next

See [`docs/`](../docs/) — start with [docs/getting-started.md](../docs/getting-started.md)
for the API walkthrough and [docs/observables.md](../docs/observables.md) for the
per-key reference of everything these examples print.
