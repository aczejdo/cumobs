"""configGuard — pairwise config-compatibility warnings (plan §4d).

When runs are overlaid, some ``CumulantConfig`` differences make the lines
*silently non-comparable*. This module flags those so the app can raise a
dismissible banner (not a hard block) naming the field, the differing values, and
the affected observables.

Verified against ``docs/configuration.md`` (the D1-D6 decisions):

* ``ddof``               — SHC scales as D^-2, so a different ddof rescales it.
* ``detrend_order``      — changes the divisor D *and* the within-window baseline
                           (and the unbiased-estimator guarantee at orders >= 3).
* ``cumulant_estimator`` — biased vs unbiased are different quantities for the
                           3rd/4th cumulants.
* ``shc_mode``           — rank vs full is a different SHC definition.
* ``window_size``        — different buffer-fill region and SHC magnitude.

Runs without a parsed config are skipped (tolerant). With fewer than two
configured runs there is nothing to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_loader import RunData

# field -> (human message, affected scalar keys)
_CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ddof": (
        "ddof differs — SHC scales as D^-2 (D = window_size - (ddof + "
        "detrend_order)), so these SHC / cumulant lines are on different scales "
        "and are not directly comparable.",
        ("spectral_heat_capacity", "total_second_cumulant", "total_third_cumulant",
         "total_fourth_cumulant"),
    ),
    "detrend_order": (
        "detrend_order differs — different within-window baseline and divisor D; "
        "the k-statistic unbiasedness (orders >= 3) also differs.",
        ("spectral_heat_capacity", "total_second_cumulant", "total_third_cumulant",
         "total_fourth_cumulant", "participation_ratio"),
    ),
    "cumulant_estimator": (
        "cumulant_estimator differs (biased vs unbiased) — the 3rd/4th cumulants "
        "estimate different quantities; a Gaussian window reads ~0 only for "
        "'unbiased'.",
        ("total_third_cumulant", "total_fourth_cumulant"),
    ),
    "shc_mode": (
        "shc_mode differs (rank vs full) — a different SHC definition (full "
        "shrinks as 1/n as probes are added).",
        ("spectral_heat_capacity",),
    ),
    "window_size": (
        "window_size differs — different buffer-fill region, SHC magnitude, and "
        "resolvable-mode cap (rank <= m-1).",
        ("spectral_heat_capacity", "participation_ratio", "effective_rank"),
    ),
}


@dataclass(frozen=True)
class ConfigWarning:
    field: str
    message: str
    values: dict[str, object]              # run name -> that run's value
    affected_keys: tuple[str, ...] = ()

    def render(self) -> str:
        vals = ", ".join(f"{name}={val!r}" for name, val in self.values.items())
        return f"{self.message}  [{vals}]"


def check_configs(runs: list[RunData]) -> list[ConfigWarning]:
    """Return one :class:`ConfigWarning` per config field that varies across the
    supplied configured runs. Only runs with a parsed config participate."""
    configured = [r for r in runs if r.config]
    if len(configured) < 2:
        return []

    warnings: list[ConfigWarning] = []
    for field_name, (message, affected) in _CHECKS.items():
        values: dict[str, object] = {}
        for r in configured:
            if field_name in r.config:  # tolerate a field missing from a config
                values[r.name] = r.config[field_name]
        distinct = {_hashable(v) for v in values.values()}
        if len(values) >= 2 and len(distinct) >= 2:
            warnings.append(
                ConfigWarning(
                    field=field_name,
                    message=message,
                    values=values,
                    affected_keys=affected,
                )
            )
    return warnings


def _hashable(v: object) -> object:
    """Make list/dict config values comparable in a set."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(val)) for k, val in v.items()))
    return v
