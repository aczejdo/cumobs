# Getting started

## Install

The package lives in `cumobs/` and needs only **`torch`** (already in this
project's environment). There is nothing to build — import it from the repo
root. Optional logging backends (`wandb`, `tensorboard`, `pandas`) are
lazy-imported only when you use them.

Run the test suite (27 tests, ~1.5 s) to confirm the environment:

```bash
python -m pytest tests -q
```

Run the end-to-end example:

```bash
python examples/fig1_transition.py            # prints the SHC spike + localization
python examples/fig1_transition.py --plot     # also writes fig1_transition.png
```

---

## Two ways to use `cumobs`

### A. The monitor (online, in a training loop)

This is the intended use: drop `CumulantMonitor` into your loop and it
evaluates probes, slides the window, computes observables, and logs — on a
schedule you set. This mirrors **Algorithm 1** of the paper.

```python
from cumobs import CumulantConfig, CumulantMonitor, ProbeSet, JSONLLogger

# 1. Probes: fixed inputs + a loss_fn that returns one loss per input sample.
probes = ProbeSet(
    inputs=probe_inputs,                 # anything with __len__ and slicing
    loss_fn=my_loss,                     # (model, batch) -> Tensor[(b,)]
    group_labels=labels,                 # optional: one hashable per input
    name="mix",
)

# 2. Config: window length, schedule, which observables to compute.
cfg = CumulantConfig(window_size=20, update_interval=10, top_k=3,
                     compute_eigenvectors=True, per_group=True)

# 3. Monitor: model + probes + config + a logger.
monitor = CumulantMonitor(model, probes, cfg, logger=JSONLLogger("run.jsonl"))

# 4. Call step() every training step.
for step in range(num_steps):
    train_step(...)
    measurement = monitor.step(step)     # Measurement on compute steps, else None
```

`monitor.step` returns the `Measurement` on a computation step and `None`
otherwise; it logs internally regardless.

### B. The pure functions (offline, notebook analysis)

If you already have a buffer `B ∈ ℝ^{m×n}` (e.g. logged losses), skip the
monitor entirely and call the math core directly. No model, no IO:

```python
import torch
from cumobs import CumulantConfig, compute_all

B = torch.randn(20, 500, dtype=torch.float64)        # (window, probes)
cfg = CumulantConfig(window_size=20, top_k=3, compute_eigenvectors=True)
meas = compute_all(B, cfg)

print(meas.scalars["spectral_heat_capacity"])
print(meas.scalars["participation_ratio"])
print(meas.vectors["covariance_spectrum"][:5])
```

`compute_all` is the same function the monitor calls per measurement.
`examples/01_minimal_observables.py` is a runnable tour of this path.

---

## The two core abstractions

### `ProbeSet` — what to measure

A `ProbeSet` is `inputs` + a `loss_fn` (+ optional `group_labels`). The contract
for `loss_fn` is **one scalar loss per input sample**:

```python
loss_fn(model, batch) -> Tensor of shape (batch_size,)
```

`ProbeSet.evaluate(model)` mini-batches `inputs` (size `batch_size`, default 64)
through `loss_fn` under `torch.no_grad()` and concatenates to a length-`n`
vector — one column of the buffer `B`.

The package makes **no** assumption about your loss API — `loss_fn` is your
encapsulation. A common, clean pattern (used by the examples) is to let
`inputs` be plain indices and close over the real probe data:

```python
probe_x, probe_y = ...                       # fixed held-out probe data
def loss_fn(model, idx):                      # idx is a LongTensor slice
    pred = model(probe_x[idx])
    return ((pred - probe_y[idx]) ** 2).mean(dim=-1)   # (b,)
probes = ProbeSet(torch.arange(len(probe_x)), loss_fn, name="probe")
```

> For the emergent-misalignment use case the paper recommends `loss_fn` compare
> against the *uncorrupted* model's temperature-0 answers — that is the caller's
> responsibility, encapsulated inside `loss_fn`.

Use a **`ProbeSetCollection`** to monitor several independently-windowed probe
sets at once; the monitor then returns and logs a `{probe_name: Measurement}`
dict (keys prefixed by probe name).

### `CumulantMonitor` — when to measure

The monitor's schedule is governed by two intervals:

| Field | Meaning |
|-------|---------|
| `eval_interval` | steps between **probe evaluations** (each appends one row to the window) |
| `update_interval` | steps between **observable computations** (eigendecompose, log) |

A measurement is produced on a step only when *both* `step % update_interval ==
0` **and** the window is full. With `window_size=20, eval_interval=1`, the window
fills at step 20 and the first measurement lands on the next update step.

Keep `update_interval < window_size` so the window meaningfully refreshes between
measurements.

```
step:        0    1    2   ...        eval_interval=2 → append on even steps
window:     [ filling..............][full ─────────────────────────────────►]
update:               ▲ (skipped: not full)        ▲ measure   ▲ measure ...
                                              update_interval=10
```

---

## Reading a `Measurement`

`compute_all` / `monitor.step` return a `Measurement` with four dicts:

```python
meas.scalars    # {str: float}    total_*, SHC, PR, distances, group_* ...
meas.vectors    # {str: Tensor}   diagonal_*, principal_*, IPR, spectra, block_energy
meas.matrices   # {str: Tensor}   C, M3, K  (only if materialize_matrices=True)
meas.health     # {str: float}    rank, n_principal, *_imag_ratio, constant_window ...
meas.step       # int             the global step it was computed at
```

Every key is catalogued in [observables.md](observables.md). To flatten for a
scalar logger:

```python
flat = meas.to_flat_dict(vector_policy="summary")   # vectors → max/mean/top1/l2
```

See [logging.md](logging.md) for the policy and backends.

---

## A complete minimal monitor loop

A self-contained, runnable version (no real model needed):

```python
import torch
from cumobs import CumulantConfig, CumulantMonitor, ProbeSet, StdoutLogger

n = 32
state = {"loss": torch.zeros(n, dtype=torch.float64)}       # a "model" stand-in
probes = ProbeSet(torch.arange(n), lambda m, idx: state["loss"][idx], name="p")

cfg = CumulantConfig(window_size=10, eval_interval=1, update_interval=5, top_k=2)
monitor = CumulantMonitor(None, probes, cfg, logger=StdoutLogger(max_keys=6))

for step in range(40):
    # pretend training changes the loss; a rank-1 mode switches on at step 20
    if step < 20:
        state["loss"] = 0.01 * torch.randn(n, dtype=torch.float64)
    else:
        v = torch.zeros(n, dtype=torch.float64); v[:8] = 1.0
        state["loss"] = 3.0 * torch.randn(()) * v + 0.01 * torch.randn(n)
    monitor.step(step)
```

You'll see the `spectral_heat_capacity` jump once the rank-1 mode turns on —
the same signal the [Fig. 1 example](../examples/fig1_transition.py) measures.

Next: the full per-key reference in [observables.md](observables.md), or tuning
in [configuration.md](configuration.md).
