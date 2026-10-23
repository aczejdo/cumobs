"""validitySpans — shaded regions where a run's traces are unreliable (plan §4e).

Derived from the ``health/*`` keys and the config, **not** invented:

* **buffer-fill** — before the sliding window is full (no measurement yet). The
  monitor only computes once the window is full, so this is the region before the
  first non-NaN measurement (equivalently ``step < window_size`` in append time).
* **dead window** — ``constant_window == 1.0``: the buffer has ~zero total
  variance, so PR / effective_rank / SHC are degenerate (NaN).
* **high imag-ratio** *(optional, off by default)* — ``contracted_*_imag_ratio``
  above ``imag_ratio_warn`` (default 1e-3); only relevant when a
  contracted-derived scalar is on screen.

A span with ``start is None`` extends to the left edge of the shared x-domain
(used for buffer-fill). Spans are in **step** units and shade every panel of the
owning run; they disappear with the run when it is toggled off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from .data_loader import RunData

# health keys we react to.
_CONSTANT_WINDOW = "constant_window"
_IMAG_RATIO_KEYS = (
    "contracted_third_cumulant_imag_ratio",
    "contracted_fourth_cumulant_imag_ratio",
)

KIND_STYLES = {
    # kind -> (fill color, alpha, hatch) for the BoxAnnotation
    "buffer_fill": ("#94a3b8", 0.14, None),   # slate
    "dead_window": ("#f43f5e", 0.12, None),   # rose
    "imag_ratio": ("#f59e0b", 0.10, "/"),     # amber, hatched
}


@dataclass(frozen=True)
class ValiditySpan:
    kind: str                 # "buffer_fill" | "dead_window" | "imag_ratio"
    start: float | None       # None => left edge of the domain
    end: float | None         # None => right edge of the domain
    reason: str


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _spacing(steps: list[int]) -> float:
    if len(steps) < 2:
        return 1.0
    diffs = [b - a for a, b in zip(steps, steps[1:]) if b > a]
    return float(median(diffs)) if diffs else 1.0


def _contiguous_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Index ranges (inclusive) where ``flags`` is True."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _first_valid_step(run: RunData) -> int | None:
    """First step where any primary scalar has a finite value."""
    best: int | None = None
    keys = run.primary_scalar_keys()
    for key in keys:
        vals = run.series.get(key, [])
        for step, v in zip(run.steps, vals):
            if _finite(v):
                if best is None or step < best:
                    best = step
                break
    return best


def compute_spans(run: RunData, include_imag_ratio: bool = False,
                  imag_ratio_warn: float | None = None) -> list[ValiditySpan]:
    spans: list[ValiditySpan] = []
    steps = run.steps
    if not steps:
        return spans
    pad = _spacing(steps) / 2.0

    # --- buffer-fill --------------------------------------------------------
    first_valid = _first_valid_step(run)
    if first_valid is not None:
        window = None
        if run.config:
            window = run.config.get("window_size")
        reason = "buffer-fill: window not yet full, no measurement"
        if window is not None:
            reason += f" (window_size = {window})"
        spans.append(ValiditySpan("buffer_fill", None, float(first_valid), reason))

    # --- dead windows -------------------------------------------------------
    cw = run.health.get(_CONSTANT_WINDOW)
    if cw is not None:
        flags = [(_finite(v) and float(v) >= 0.5) for v in cw]
        for i0, i1 in _contiguous_runs(flags):
            spans.append(
                ValiditySpan(
                    "dead_window",
                    float(steps[i0]) - pad,
                    float(steps[i1]) + pad,
                    "dead window: constant_window == 1.0 (near-zero total variance)",
                )
            )

    # --- optional high imag-ratio ------------------------------------------
    if include_imag_ratio:
        warn = imag_ratio_warn
        if warn is None:
            warn = float((run.config or {}).get("imag_ratio_warn", 1e-3))
        for hkey in _IMAG_RATIO_KEYS:
            arr = run.health.get(hkey)
            if arr is None:
                continue
            flags = [(_finite(v) and float(v) > warn) for v in arr]
            for i0, i1 in _contiguous_runs(flags):
                spans.append(
                    ValiditySpan(
                        "imag_ratio",
                        float(steps[i0]) - pad,
                        float(steps[i1]) + pad,
                        f"high {hkey} (> {warn:g}): precision loss in the "
                        "contracted eig",
                    )
                )

    return spans
