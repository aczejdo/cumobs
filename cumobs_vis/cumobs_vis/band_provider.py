"""bandProvider — a pluggable uncertainty-band interface (plan §4c).

The band behind each trace is the *highest-value* feature: a trace whose movement
stays inside its band is not an event. The **source** of the band is abstracted
behind :class:`BandProvider` so a future RMT / Marchenko-Pastur-edge band can
replace the jackknife SE without touching the plotting code:

    provider.band(run, key, k) -> (lo, hi) aligned to run.steps, in RAW units,
                                  or None if the provider has nothing for that key.

The panel applies any invert (``1 - value``) transform to *both* the trace and
its band, so providers always return raw-value bounds.

Shipped: :class:`JackknifeSEBandProvider` (uses the ``<scalar>/se`` sidecar keys).
Stubbed: :class:`RMTBandProvider` (returns ``None`` everywhere; the seam for the
Marchenko-Pastur null band).
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .data_loader import RunData

_NAN = float("nan")


def _finite(x: float | None) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _band_from(values: list[float], se: list[float] | None, k: float):
    if se is None:
        return None
    lo: list[float] = []
    hi: list[float] = []
    any_finite = False
    for v, e in zip(values, se):
        if _finite(v) and _finite(e):
            lo.append(float(v) - k * float(e))
            hi.append(float(v) + k * float(e))
            any_finite = True
        else:
            lo.append(_NAN)
            hi.append(_NAN)
    return (lo, hi) if any_finite else None


@runtime_checkable
class BandProvider(Protocol):
    name: str

    def band(self, run: RunData, key: str, k: float): ...

    def band_group(self, run: RunData, base_key: str, label: str, k: float): ...


class JackknifeSEBandProvider:
    """``value +/- k * se`` from the leave-one-row-out jackknife SE keys."""

    name = "jackknife-se"

    def band(self, run: RunData, key: str, k: float):
        values = run.series.get(key)
        if values is None:
            return None
        return _band_from(values, run.se.get(key), k)

    def band_group(self, run: RunData, base_key: str, label: str, k: float):
        series = run.groups.get(base_key, {}).get(label)
        if series is None:
            return None
        se = run.groups_se.get(base_key, {}).get(label)
        return _band_from(series, se, k)


class RMTBandProvider:
    """Stub for a future Random-Matrix-Theory / Marchenko-Pastur-edge band.

    This is the seam the plan calls for (§4c, §6): a provider that derives the
    band from the null model rather than the estimator SE. It returns ``None``
    for now so the app falls back cleanly to no band."""

    name = "rmt-edge (stub)"

    def band(self, run: RunData, key: str, k: float):
        return None  # TODO: derive from mp_edge_upper / null_sd once wired.

    def band_group(self, run: RunData, base_key: str, label: str, k: float):
        return None


# The default provider used by the app; swap this to change the band source.
DEFAULT_PROVIDER: BandProvider = JackknifeSEBandProvider()
