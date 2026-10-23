# Logging

A `Logger` receives a **flat** `{key: float | Tensor}` mapping plus a step index.
The monitor produces that mapping from a `Measurement` via `to_flat_dict`, then
calls `logger.log(flat, step)`. You can also flatten + log yourself when using
the pure-function path. Source: [`cumobs/logging.py`](../cumobs/logging.py),
[`cumobs/types.py`](../cumobs/types.py).

---

## The flattening policy (`to_flat_dict` / `LoggingConfig`)

`Measurement.to_flat_dict(vector_policy, vector_overrides, log_health)` turns the
four structured dicts into one flat mapping:

- **Scalars** (`total_*`, `spectral_heat_capacity`, `participation_ratio`,
  `von_neumann_entropy`, distances, `group_*/…`, `excess_loss`, and any
  `<scalar>/se` error bars) → passed through unchanged.
- **Vectors** (`diagonal_*`, `principal_*`, `inverse_participation_ratio`,
  `*_spectrum`, `block_energy/…`) → governed by `vector_policy`:
  - `"summary"` *(default)* → four scalars per vector: `key/max`, `key/mean`,
    `key/top1` (first element), `key/l2`. Keeps scalar-only backends legible.
  - `"full"` → the raw array (for array-capable backends / histograms).
  - `"off"` → skipped.
  - **Per-key overrides** via `vector_overrides`, e.g. always-full
    `covariance_spectrum`.
- **Aux** (`active_probe_mask`, `active_probe_mask_any` — the §6 BBP masks as hex
  strings / lists) → passed through **verbatim** (JSON-native primitives). The
  `JSONLLogger` writes them straight into the scalar line; `InMemoryLogger`
  retains them.
- **Health** (`rank`, `n_principal`, `effective_window`, `*_imag_ratio`, …) →
  emitted as `health/<key>` when `log_health=True`.
- **Matrices** (`covariance`, `M3`, `K`) and **factors** (`V`, `lam_top`, used
  internally for the CKA distance) → **never** auto-flattened. Persist matrices
  yourself (e.g. an `.npz` sidecar) if needed.

> **Consolidation trim (§8).** `LoggingConfig(trim_consolidated=True)` suppresses
> the deprecated-default keys (`von_neumann_entropy`, bare `classical_fidelity` /
> `quantum_fidelity`, `total_cumulant_{5+}`) from the flattened output in favour
> of their kept companions (`effective_rank`, the geodesic distances). Everything
> is still computed and present on the `Measurement`; only the flatten step drops
> them. Add more with `LoggingConfig(drop_keys={...})`.

```python
from cumobs import LoggingConfig
lc = LoggingConfig(
    vector_policy="summary",
    vector_overrides={"covariance_spectrum": "full"},   # keep the raw spectrum
    log_health=True,
)
monitor = CumulantMonitor(model, probes, cfg, logger=my_logger, logging_config=lc)
```

For a `ProbeSetCollection`, the monitor additionally prefixes every key with the
probe name (`"<probe>/<key>"`) so multiple probe sets don't collide.

---

## Backends

| Backend | Best for | Captures vectors? | Extra dep |
|---------|----------|-------------------|-----------|
| `InMemoryLogger` | notebooks, tests, ad-hoc analysis | yes (as tensors) | — (`pandas` optional) |
| `JSONLLogger` | persistent runs, scalars to disk | scalars to JSONL, arrays to `.npz` | `numpy` (for the sidecar) |
| `StdoutLogger` | quick console sanity checks | no (scalars only) | — |
| `WandbLogger` | dashboards | vectors → histograms | `wandb` |
| `TensorBoardLogger` | dashboards | 1-D vectors → histograms | `tensorboard` |

Optional deps (`wandb`, `tensorboard`, `pandas`, `numpy`) are imported **inside**
their backend so the core installs with just `torch`.

### `InMemoryLogger`
Retains every logged row. Inspect with:
- `as_dict()` → columnar `{key: [values…]}` (missing entries → `None`).
- `as_dataframe()` → a pandas `DataFrame` if pandas is installed, else the dict.

```python
from cumobs import InMemoryLogger
logger = InMemoryLogger()
# ... run the monitor ...
df = logger.as_dataframe()
df[["step", "spectral_heat_capacity", "participation_ratio"]]
```

### `JSONLLogger(path, array_sidecar=True)`
Writes one JSON line of **scalars** per step (truncates `path` on construction).
If `array_sidecar`, full-array values are accumulated keyed by `"{step}/{key}"`
and flushed on demand:

```python
from cumobs import JSONLLogger
logger = JSONLLogger("run.jsonl")
# ... run with vector_policy="full" to capture arrays ...
npz_path = logger.flush_arrays()        # writes run.jsonl.npz, returns the path
```

### `StdoutLogger(fmt="{:.4g}", max_keys=None)`
Prints `[step N] key=val …` for scalar values only. `max_keys` caps the line
width.

### `WandbLogger(**init_kwargs)` / `TensorBoardLogger(log_dir=None)`
`WandbLogger` calls `wandb.init(**init_kwargs)` if no run is active; logs scalars
directly and 1-D vectors as `wandb.Histogram`. `TensorBoardLogger` writes scalars
via `add_scalar` and non-empty 1-D vectors via `add_histogram`; call `close()`
when done. Both need `vector_policy="full"` to receive arrays for histograms.

---

## Writing your own backend

A logger is anything satisfying the `Logger` protocol — a single method:

```python
from typing import Mapping
from torch import Tensor

class MyLogger:
    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        for key, value in metrics.items():
            ...   # send `value` at `step` wherever you like
```

Pass an instance as `logger=` to `CumulantMonitor`. The helpers `_is_scalarish`
and `_to_py` in `logging.py` show how the shipped backends distinguish scalars
from arrays.
