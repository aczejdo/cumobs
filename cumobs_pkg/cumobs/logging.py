"""Pluggable logging backends (§7).

A :class:`Logger` receives a flat ``{key: float | Tensor}`` mapping (produced by
:meth:`cumobs.types.Measurement.to_flat_dict`) plus a step index.  Backends:

* :class:`InMemoryLogger` — retains everything; returns a history dict (or a
  pandas ``DataFrame`` if pandas is present) for notebook analysis.
* :class:`JSONLLogger` — one JSON line of scalars per step, with an optional
  ``.npz`` sidecar for arrays / matrices.
* :class:`StdoutLogger` — a compact scalar line.
* :class:`WandbLogger` / :class:`TensorBoardLogger` — lazy-import their deps and
  map vectors to histograms.

Optional deps (``wandb``, ``tensorboard``) are imported only inside their
backend classes so the core installs with just ``torch``.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Logger(Protocol):
    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None: ...


def _is_scalarish(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, Tensor):
        return v.numel() == 1
    return False


def _to_py(v: Any) -> Any:
    if isinstance(v, Tensor):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


# ---------------------------------------------------------------------------


class InMemoryLogger:
    """Retain every logged metric for later inspection."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        row: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            row[k] = v.detach().cpu() if isinstance(v, Tensor) else v
        self.history.append(row)

    def as_dict(self) -> dict[str, list[Any]]:
        """Columnar history ``{key: [values...]}`` (missing entries -> ``None``)."""
        keys: list[str] = []
        seen: set[str] = set()
        for row in self.history:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return {k: [row.get(k) for row in self.history] for k in keys}

    def as_dataframe(self):
        """Return a pandas ``DataFrame`` if pandas is installed, else the dict."""
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError:
            return self.as_dict()
        return pd.DataFrame(self.history)


class JSONLLogger:
    """Write one JSON line of scalars per step (+ optional ``.npz`` for arrays)."""

    def __init__(self, path: str, array_sidecar: bool = True) -> None:
        self.path = path
        self.array_sidecar = array_sidecar
        self._arrays: dict[str, Any] = {}
        # truncate any existing file
        with open(self.path, "w", encoding="utf-8"):
            pass

    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        scalars: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if _is_scalarish(v):
                scalars[k] = _to_py(v)
            elif isinstance(v, (str, list, dict)):
                # aux primitives (e.g. the BBP mask hex string) — JSON-native.
                scalars[k] = v
            elif self.array_sidecar and isinstance(v, Tensor):
                self._arrays[f"{step}/{k}"] = v.detach().cpu().numpy()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(scalars) + "\n")

    def flush_arrays(self, npz_path: str | None = None) -> str | None:
        """Persist accumulated arrays to an ``.npz`` sidecar; return its path."""
        if not self._arrays:
            return None
        import numpy as np  # noqa: PLC0415

        out = npz_path or (self.path + ".npz")
        np.savez(out, **self._arrays)
        return out


class StdoutLogger:
    """Print a compact, scalar-only line per step."""

    def __init__(self, fmt: str = "{:.4g}", max_keys: int | None = None) -> None:
        self.fmt = fmt
        self.max_keys = max_keys

    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        parts: list[str] = []
        for k, v in metrics.items():
            if not _is_scalarish(v):
                continue
            val = _to_py(v)
            try:
                parts.append(f"{k}={self.fmt.format(val)}")
            except (ValueError, TypeError):
                parts.append(f"{k}={val}")
        if self.max_keys is not None:
            parts = parts[: self.max_keys]
        print(f"[step {step}] " + " ".join(parts))


class WandbLogger:
    """Log scalars directly and vectors as histograms via Weights & Biases."""

    def __init__(self, **init_kwargs: Any) -> None:
        import wandb  # noqa: PLC0415

        self._wandb = wandb
        if wandb.run is None:
            wandb.init(**init_kwargs)

    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        payload: dict[str, Any] = {}
        for k, v in metrics.items():
            if _is_scalarish(v):
                payload[k] = _to_py(v)
            elif isinstance(v, Tensor) and v.dim() == 1:
                payload[k] = self._wandb.Histogram(v.detach().cpu().numpy())
        self._wandb.log(payload, step=step)


class TensorBoardLogger:
    """Log scalars and 1-D vectors (as histograms) via TensorBoard."""

    def __init__(self, log_dir: str | None = None) -> None:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        self._writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: Mapping[str, float | Tensor], step: int) -> None:
        for k, v in metrics.items():
            if _is_scalarish(v):
                self._writer.add_scalar(k, _to_py(v), global_step=step)
            elif isinstance(v, Tensor) and v.dim() == 1 and v.numel() > 0:
                self._writer.add_histogram(k, v.detach().cpu(), global_step=step)

    def close(self) -> None:
        self._writer.close()
