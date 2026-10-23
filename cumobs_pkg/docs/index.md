# `cumobs` documentation

`cumobs` is a lean, framework-free training monitor that turns the
cumulant-expansion picture from *"From Density Matrices to Phase Transitions in
Deep Learning"* (Hennick & Corlouer, 2026) into a drop-in observable stack. It
tracks scalar cumulants up to 4th order plus spectral diagnostics of the
per-sample loss covariance (the **2RDM**) over a sliding window, needing only a
forward pass on a probe set. The only hard dependency is `torch`.

This folder is the human- and agent-readable documentation. The top-level
[`README.md`](../README.md) is the 60-second overview; the
[`cumulant_observables_spec.md`](../cumulant_observables_spec.md) is the
authoritative mathematical specification (every section reference like "§3.4" in
these docs points there).

## Reading order

| If you want to… | Read |
|-----------------|------|
| Run it in a training loop in 5 lines | [getting-started.md](getting-started.md) |
| Understand the math (what is a 2RDM, why the Gram trick) | [concepts.md](concepts.md) |
| Look up what a specific output key means | [observables.md](observables.md) |
| Tune the behavior / understand the design decisions | [configuration.md](configuration.md) |
| Know which class/function does what | [api-reference.md](api-reference.md) |
| Send observables to wandb / TensorBoard / disk | [logging.md](logging.md) |
| Add activation cumulants, Gaussian sampling, quantum distances | [extending.md](extending.md) |
| See it run end-to-end | [`examples/`](../examples/) (start with [examples/README.md](../examples/README.md)) |

## What the package is for

When a network undergoes a *phase transition* during training (a circuit forms,
a capability emerges, a representation reorganizes), the statistics of the
per-sample loss change in a structured way. `cumobs` measures those statistics
cheaply and continuously:

- **Second cumulant (covariance / 2RDM).** Its spectrum tells you how many
  effective modes the loss fluctuations live on (participation ratio) and how
  "spread" the eigenvalues are (spectral heat capacity — a transition
  diagnostic). Its top eigenvector tells you *which probes* a forming mode acts
  on (block energy / IPR).
- **Third & fourth cumulants (co-skewness, co-kurtosis).** Non-Gaussianity of
  the loss fluctuations — the `wick_breakdown_ratio` is a direct numerical
  analog of "the cross terms dropped under the Gaussian assumption."
- **Distances between successive measurements.** Fidelity and trace distance
  flag *when* the loss-covariance structure changes — a transition detector.

## The one idea to take away

Every scalar and spectral observable is computed in **window space** (`m × m`,
where `m` is the window length) via the *Gram trick*, so no `n × n` matrix (over
the `n` probes) is ever allocated unless you explicitly ask for it
(`materialize_matrices=True`). With a window of `m = 20` and thousands of probes,
a full measurement is one `m × m` symmetric eigendecomposition plus two small
asymmetric ones — cheap enough to run every few steps. See
[concepts.md → The Gram trick](concepts.md#4-the-gram-trick-why-it-stays-cheap).

## Package map

| Module | Responsibility | Docs |
|--------|----------------|------|
| `cumobs/config.py` | `CumulantConfig`, `LoggingConfig` | [configuration.md](configuration.md) |
| `cumobs/types.py` | `SharedIntermediates`, `Measurement`, `to_flat_dict` | [api-reference.md](api-reference.md#typespy) |
| `cumobs/window.py` | `SlidingWindow`, `BufferSource`, `DynamicalSource`, `GaussianSource` | [api-reference.md](api-reference.md#windowpy) |
| `cumobs/probes.py` | `ProbeSet`, `ProbeSetCollection` | [api-reference.md](api-reference.md#probespy) |
| `cumobs/observables.py` | the math core: `build_intermediates`, `compute_all`, … | [observables.md](observables.md), [api-reference.md](api-reference.md#observablespy) |
| `cumobs/distances.py` | `classical_fidelity`, `trace_distance`, quantum variants | [api-reference.md](api-reference.md#distancespy) |
| `cumobs/logging.py` | `Logger` protocol + backends | [logging.md](logging.md) |
| `cumobs/monitor.py` | `CumulantMonitor.step()` orchestrator | [getting-started.md](getting-started.md), [api-reference.md](api-reference.md#monitorpy) |
