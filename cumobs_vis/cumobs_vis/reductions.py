"""Per-mode -> scalar reduction hook (plan §4j) — optional, off by default.

The tool is strictly scalar. This hook lets an *advanced* user derive a single
**labeled scalar** from a per-mode vector logged in the ``.npz`` sidecar (e.g.
``wick_breakdown_ratio`` top mode, or its mean over modes) without turning the
tool into a vector viewer.

Two hard rules from the plan, enforced here:

1. The reduction (``index k`` or ``mean``) **must appear in the trace name** —
   see :attr:`DerivedSeries.name`.
2. An *index* reduction **must carry the mode-crossing caveat**: a "top mode"
   reduction silently switches modes at a crossing, so a discontinuity in
   ``vector[0]`` can be a mode swap rather than a real change. The caveat is
   baked into :attr:`DerivedSeries.tooltip`.

This is gated behind a clearly-labeled advanced toggle in the app so the default
tool stays scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data_loader import RunData

MODE_CROSSING_CAVEAT = (
    " CAVEAT: an index reduction follows a fixed slot, not a fixed physical mode "
    "— at a mode crossing the slot silently switches modes, so a discontinuity "
    "here can be a mode swap rather than a real change."
)

_NAN = float("nan")


@dataclass
class DerivedSeries:
    name: str
    steps: list[int]
    values: list[float]
    tooltip: str


def _load_vector(run: RunData, vector_key: str) -> dict[int, list[float]]:
    """Load ``{step: vector}`` for ``vector_key`` from the run's ``.npz`` sidecar."""
    if not run.sidecar_path:
        return {}
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return {}
    target = (f"{run.key_prefix}/{vector_key}" if run.key_prefix else vector_key)
    out: dict[int, list[float]] = {}
    try:
        with np.load(run.sidecar_path) as npz:
            for name in npz.files:
                if "/" not in name:
                    continue
                step_str, obs = name.split("/", 1)
                if obs != target:
                    continue
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                out[step] = [float(x) for x in np.asarray(npz[name]).ravel()]
    except Exception:
        return {}
    return out


def available_vector_keys(run: RunData) -> list[str]:
    """Vector observables present in the run's ``.npz`` sidecar (reducible)."""
    if not run.sidecar_path:
        return []
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return []
    keys: set[str] = set()
    pfx = (run.key_prefix + "/") if run.key_prefix else ""
    try:
        with np.load(run.sidecar_path) as npz:
            for name in npz.files:
                if "/" not in name:
                    continue
                obs = name.split("/", 1)[1]     # strip the "{step}/"
                if pfx and obs.startswith(pfx):
                    obs = obs[len(pfx):]
                keys.add(obs)
    except Exception:
        return []
    return sorted(keys)


def reduce_series(run: RunData, vector_key: str, mode) -> DerivedSeries | None:
    """Reduce a per-mode vector to a labeled scalar series.

    ``mode`` is an int (take element ``mode`` of each step's vector) or the
    string ``"mean"`` (mean over each step's modes). Returns ``None`` if the
    vector is not present in the sidecar."""
    per_step = _load_vector(run, vector_key)
    if not per_step:
        return None

    is_mean = (mode == "mean")
    label = "mean" if is_mean else str(int(mode))
    name = f"{vector_key}[{label}]"

    values: list[float] = []
    for step in run.steps:
        vec = per_step.get(step)
        if not vec:
            values.append(_NAN)
            continue
        if is_mean:
            finite = [v for v in vec if not math.isnan(v)]
            values.append(sum(finite) / len(finite) if finite else _NAN)
        else:
            k = int(mode)
            values.append(vec[k] if 0 <= k < len(vec) else _NAN)

    tooltip = f"Reduction of the per-mode vector '{vector_key}' -> {name}."
    if not is_mean:
        tooltip += MODE_CROSSING_CAVEAT
    return DerivedSeries(name=name, steps=list(run.steps), values=values, tooltip=tooltip)
