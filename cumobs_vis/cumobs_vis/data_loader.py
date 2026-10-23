"""Ingestion: manifest + observables JSONL (+ metrics + config + npz sidecar).

Turns the on-disk artifacts written by :class:`cumobs.JSONLLogger` into a
per-run :class:`RunData` with everything the UI needs, and **nothing invented**:
the flattening policy in ``docs/logging.md`` is the contract.

Key handling (verified against ``docs/logging.md`` / ``docs/observables.md``):

* ``step``                    → the measurement step (x-axis).
* ``health/<key>``            → validity diagnostics (``constant_window`` etc.).
* ``<scalar>/se``             → the jackknife SE *sidecar* of ``<scalar>`` — drives
                                the uncertainty band, **not** a separate trace.
* ``group_shc/<label>`` &c.   → per-group sub-series, nested under the run.
* bare ``<scalar>``           → a primary scalar trace (gets a panel).
* ``<scalar>/max``, ``…/null_z`` etc. → other flattened scalars: loaded and kept,
                                but not given a default panel (they still resolve
                                if a user adds them to the catalog).

Ingestion is **tolerant of missing keys** (plan §1): a key absent from a row
becomes ``NaN`` at that step; a scalar absent from the whole run is simply not
present. Nothing raises on a partially-populated run.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from .catalog import GROUP_SERIES

# A default categorical palette (run color = identity across every panel).
DEFAULT_PALETTE = [
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#10b981",  # green
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#64748b",  # slate
]

_NAN = float("nan")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _group_split(key: str) -> tuple[str, str] | None:
    """If ``key`` is a per-group scalar (``group_shc/A``), return ``(base, label)``."""
    for base in GROUP_SERIES:
        prefix = base + "/"
        if key.startswith(prefix):
            return base, key[len(prefix):]
    return None


@dataclass
class RunData:
    """One loaded run = one probe set + config = one observables JSONL."""

    name: str
    color: str
    steps: list[int] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    se: dict[str, list[float]] = field(default_factory=dict)
    health: dict[str, list[float]] = field(default_factory=dict)
    groups: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    groups_se: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    metrics_steps: list[int] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    config_full: dict[str, Any] | None = None
    sidecar_path: str | None = None
    has_covariance_spectrum: bool = False
    source_path: str = ""
    key_prefix: str = ""          # probe-set prefix when from a ProbeSetCollection

    # -- derived views -----------------------------------------------------

    def primary_scalar_keys(self) -> set[str]:
        """Bare scalar keys (no ``/``) — the ones that earn a default panel."""
        return {k for k, v in self.series.items() if "/" not in k and _has_finite(v)}

    def group_labels(self, base_key: str) -> list[str]:
        return sorted(self.groups.get(base_key, {}).keys())

    def has_metrics(self) -> bool:
        return bool(self.metrics_steps) and any(
            _has_finite(v) for v in self.metrics.values()
        )

    # -- lazy spectrum sidecar (for the §4i click-to-set-reference) --------

    def load_covariance_spectra(self) -> dict[int, "list[float]"]:
        """Load ``{step: covariance_spectrum}`` from the ``.npz`` sidecar.

        Returns an empty dict if there is no sidecar or it holds no spectra.
        Requires numpy (already a ``cumobs`` dependency)."""
        if not self.has_covariance_spectrum or not self.sidecar_path:
            return {}
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            return {}
        # sidecar keys are "{step}/{full_key}"; the full key carries the probe
        # prefix for a ProbeSetCollection (e.g. "train_mix/covariance_spectrum").
        target = (f"{self.key_prefix}/covariance_spectrum" if self.key_prefix
                  else "covariance_spectrum")
        out: dict[int, list[float]] = {}
        with np.load(self.sidecar_path) as npz:
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
        return out


def _has_finite(values: "list[float]") -> bool:
    return any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in values)


# --- JSONL parsing ---------------------------------------------------------


def _iter_jsonl(lines) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a partially-written / trailing line
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return _iter_jsonl(fh)


# structured first-segments that are NOT probe-set prefixes.
_KNOWN_STRUCTURED = {"health", "block_energy"} | set(GROUP_SERIES)


def detect_probe_prefixes(rows: list[dict[str, Any]]) -> list[str]:
    """Detect ``ProbeSetCollection`` prefixes (``<probe>/<key>``).

    The monitor prefixes every key with the probe-set name when a collection is
    logged (see ``docs/logging.md``). A first segment is a probe prefix if it is
    not a known structured token / catalog scalar, and stripping it yields a
    recognisable cumobs key. Returns the sorted prefixes, or ``[]`` for a plain
    single-probe file."""
    from .catalog import SCALARS  # local import to avoid a cycle at import time

    prefixes: set[str] = set()
    for row in rows[:64]:  # keys are stable across rows; sample the head
        for key in row:
            if key == "step" or "/" not in key:
                continue
            first, rest = key.split("/", 1)
            if first in _KNOWN_STRUCTURED or first in SCALARS:
                continue
            # strip a trailing /se before testing the remainder.
            core = rest[:-3] if rest.endswith("/se") else rest
            head = core.split("/", 1)[0]
            if head in SCALARS or head in _KNOWN_STRUCTURED:
                prefixes.add(first)
    return sorted(prefixes)


def _parse_observables(rows: list[dict[str, Any]], run: RunData,
                       prefix: str | None = None) -> None:
    """Populate ``run.steps`` / ``series`` / ``se`` / ``health`` / ``groups``.

    If ``prefix`` is given, only keys under ``<prefix>/`` are consumed and the
    prefix is stripped. Every value array is aligned to ``run.steps`` with
    ``NaN`` where the key is absent for a step."""
    pfx = (prefix + "/") if prefix else None
    run.key_prefix = prefix or ""

    # 1) collect the ordered, unique step axis.
    steps: list[int] = []
    seen: set[int] = set()
    embedded_config: dict[str, Any] | None = None
    for row in rows:
        s = row.get("step")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue
        s = int(s)
        if s not in seen:
            seen.add(s)
            steps.append(s)
        for ck in ("config", "_config", "cumulant_config"):
            cv = row.get(ck)
            if isinstance(cv, dict):
                embedded_config = cv
    steps.sort()
    run.steps = steps
    index = {s: i for i, s in enumerate(steps)}
    n = len(steps)

    def _ensure(store: dict[str, list[float]], key: str) -> list[float]:
        arr = store.get(key)
        if arr is None:
            arr = [_NAN] * n
            store[key] = arr
        return arr

    def _ensure_group(store: dict[str, dict[str, list[float]]], base: str, label: str) -> list[float]:
        d = store.setdefault(base, {})
        arr = d.get(label)
        if arr is None:
            arr = [_NAN] * n
            d[label] = arr
        return arr

    for row in rows:
        s = row.get("step")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue
        i = index[int(s)]
        for key, val in row.items():
            if key in ("step", "config", "_config", "cumulant_config"):
                continue
            if pfx is not None:
                if not key.startswith(pfx):
                    continue
                key = key[len(pfx):]        # strip the probe-set prefix
            if not _is_number(val):
                continue  # aux strings/lists (BBP masks) — not scalar traces
            fval = float(val)
            if key.startswith("health/"):
                _ensure(run.health, key[len("health/"):])[i] = fval
            elif key.endswith("/se"):
                base = key[: -len("/se")]
                grp = _group_split(base)
                if grp is not None:
                    _ensure_group(run.groups_se, grp[0], grp[1])[i] = fval
                else:
                    _ensure(run.se, base)[i] = fval
            else:
                grp = _group_split(key)
                if grp is not None:
                    _ensure_group(run.groups, grp[0], grp[1])[i] = fval
                else:
                    _ensure(run.series, key)[i] = fval

    if run.config is None and embedded_config is not None:
        run.config = embedded_config


def extract_cumobs_config(raw: Any) -> dict[str, Any] | None:
    """Find the CumulantConfig-like sub-dict inside a possibly-nested config JSON.

    Real experiment configs often nest the cumobs settings (e.g.
    ``{"config": {"cumobs": {...}}}``). The config guard / validity spans read
    fields like ``window_size`` — pull out the dict that has them, tolerating
    absence."""
    if not isinstance(raw, dict):
        return None
    if "window_size" in raw:
        return raw
    sub = raw.get("cumobs")
    if isinstance(sub, dict) and "window_size" in sub:
        return sub
    nested = raw.get("config")
    if isinstance(nested, dict):
        found = extract_cumobs_config(nested)
        if found is not None:
            return found
    for v in raw.values():
        if isinstance(v, dict) and "window_size" in v:
            return v
    return raw  # fallback: guard tolerates missing fields


def _parse_metrics(path: str, run: RunData) -> None:
    rows = _read_jsonl(path)
    steps: list[int] = []
    seen: set[int] = set()
    for row in rows:
        s = row.get("step")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue
        s = int(s)
        if s not in seen:
            seen.add(s)
            steps.append(s)
    steps.sort()
    run.metrics_steps = steps
    index = {s: i for i, s in enumerate(steps)}
    n = len(steps)
    for row in rows:
        s = row.get("step")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            continue
        i = index[int(s)]
        for key, val in row.items():
            if key == "step" or not _is_number(val):
                continue
            arr = run.metrics.get(key)
            if arr is None:
                arr = [_NAN] * n
                run.metrics[key] = arr
            arr[i] = float(val)


def _detect_sidecar(observables_path: str) -> tuple[str | None, set[str]]:
    """Find a ``.npz`` array sidecar; return its path and the set of observable
    suffixes present (the part after ``{step}/``, e.g. ``covariance_spectrum`` or
    ``train_mix/covariance_spectrum`` for a collection)."""
    candidate = observables_path + ".npz"
    if not os.path.exists(candidate):
        return None, set()
    try:
        import numpy as np  # noqa: PLC0415

        suffixes: set[str] = set()
        with np.load(candidate) as npz:
            for name in npz.files:
                if "/" in name:
                    suffixes.add(name.split("/", 1)[1])
        return candidate, suffixes
    except Exception:
        return candidate, set()


# --- public API ------------------------------------------------------------


def load_runs(
    observables_path: str,
    name: str,
    color: str,
    config_path: str | None = None,
    metrics_path: str | None = None,
    base_dir: str | None = None,
) -> list[RunData]:
    """Load one observables file into one **or more** runs.

    A ``ProbeSetCollection`` logs every probe set into a single file with
    ``<probe>/<key>`` prefixes; this splits it into one :class:`RunData` per
    probe set (each prefix), stripping the prefix so keys resolve. A plain
    single-probe file yields exactly one run. Config is extracted from a possibly
    nested JSON; metrics are attached to the first split run (they are global
    training metrics, not per-probe-set)."""
    def _resolve(p: str | None) -> str | None:
        if p is None:
            return None
        if base_dir and not os.path.isabs(p):
            return os.path.join(base_dir, p)
        return p

    obs = _resolve(observables_path)
    assert obs is not None

    # config (may be nested).
    config_extracted: dict[str, Any] | None = None
    config_full: dict[str, Any] | None = None
    cfg_p = _resolve(config_path)
    if cfg_p and os.path.exists(cfg_p):
        try:
            with open(cfg_p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                config_full = raw
                config_extracted = extract_cumobs_config(raw)
        except (OSError, json.JSONDecodeError):
            pass

    rows = _read_jsonl(obs)
    prefixes = detect_probe_prefixes(rows)
    # "mix" / aggregate probe sets first — they are the most informative default.
    prefixes.sort(key=lambda p: (0 if "mix" in p.lower() else 1, p))
    sidecar_path, sidecar_suffixes = _detect_sidecar(obs)

    # training metrics are global (one training run), so parse once and share the
    # same objects across split runs — the reference panel dedupes by identity.
    m_p = _resolve(metrics_path)
    shared_metrics_steps: list[int] = []
    shared_metrics: dict[str, list[float]] = {}
    if m_p and os.path.exists(m_p):
        holder = RunData(name="_metrics", color="#000")
        _parse_metrics(m_p, holder)
        shared_metrics_steps, shared_metrics = holder.metrics_steps, holder.metrics

    def _build(run_name: str, run_color: str, prefix: str | None) -> RunData:
        run = RunData(name=run_name, color=run_color, source_path=obs)
        run.config = config_extracted
        run.config_full = config_full
        _parse_observables(rows, run, prefix=prefix)
        run.metrics_steps = shared_metrics_steps
        run.metrics = shared_metrics
        run.sidecar_path = sidecar_path
        want = (f"{prefix}/covariance_spectrum" if prefix else "covariance_spectrum")
        run.has_covariance_spectrum = want in sidecar_suffixes
        return run

    if not prefixes:
        return [_build(name, color, None)]

    runs: list[RunData] = []
    for i, pfx in enumerate(prefixes):
        run_color = color if i == 0 else DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
        run_name = pfx if len(prefixes) > 1 or pfx == name else f"{name}·{pfx}"
        runs.append(_build(run_name, run_color, pfx))
    return runs


def load_run(
    observables_path: str,
    name: str,
    color: str,
    config_path: str | None = None,
    metrics_path: str | None = None,
    base_dir: str | None = None,
) -> RunData:
    """Load a single run (the first, if the file is a collection). Kept for
    callers that expect exactly one run."""
    return load_runs(observables_path, name, color, config_path, metrics_path,
                     base_dir)[0]


def load_run_from_text(text: str, name: str, color: str) -> RunData:
    """Build a :class:`RunData` from raw JSONL text (the app's drag-drop path).

    No config / metrics / ``.npz`` sidecar is available for a dropped file, which
    is fine — ingestion tolerates their absence."""
    run = RunData(name=name, color=color, source_path="<uploaded>")
    _parse_observables(_iter_jsonl(text.splitlines()), run)
    return run


def load_manifest(path: str) -> list[RunData]:
    """Load every run in a ``runs.json`` manifest (plan §1).

    The manifest is::

        { "runs": [ {"name", "color", "observables", "config?", "metrics?"}, ... ] }

    Relative paths are resolved against the manifest's directory. Colors default
    to a categorical palette. A single run object (no ``runs`` wrapper) is also
    accepted.
    """
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if isinstance(manifest, dict) and "runs" in manifest:
        specs = manifest["runs"]
    elif isinstance(manifest, list):
        specs = manifest
    elif isinstance(manifest, dict):
        specs = [manifest]
    else:
        raise ValueError(f"unrecognised manifest structure in {path!r}")

    base_dir = os.path.dirname(os.path.abspath(path))
    runs: list[RunData] = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict) or "observables" not in spec:
            continue
        name = str(spec.get("name") or f"run{i}")
        color = str(spec.get("color") or DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        runs.extend(
            load_runs(
                observables_path=spec["observables"],
                name=name,
                color=color,
                config_path=spec.get("config"),
                metrics_path=spec.get("metrics"),
                base_dir=base_dir,
            )
        )
    return runs


def _discover_dir(directory: str) -> list[RunData]:
    """Auto-discover runs in a directory that has no ``runs.json``.

    Every ``*.jsonl`` that is not a ``*.metrics.jsonl`` / ``metrics.jsonl`` is an
    observables file; metrics / config / ``.npz`` sidecars are paired by naming
    convention (``X.metrics.jsonl`` or a lone ``metrics.jsonl``; ``X.config.json``
    or a lone ``config.json``)."""
    entries = sorted(os.listdir(directory))
    files = set(entries)
    obs_files = [
        f for f in entries
        if f.endswith(".jsonl") and not f.endswith(".metrics.jsonl")
        and f != "metrics.jsonl"
    ]
    lone_metrics = "metrics.jsonl" if "metrics.jsonl" in files else None
    lone_config = "config.json" if "config.json" in files else None

    runs: list[RunData] = []
    for i, obs in enumerate(obs_files):
        stem = obs[: -len(".jsonl")]
        metrics = next((c for c in (f"{stem}.metrics.jsonl", f"{obs}.metrics",
                                    lone_metrics) if c and c in files), None)
        config = next((c for c in (f"{stem}.config.json", lone_config)
                       if c and c in files), None)
        name = stem
        color = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
        runs.extend(
            load_runs(
                observables_path=os.path.join(directory, obs),
                name=name,
                color=color,
                config_path=os.path.join(directory, config) if config else None,
                metrics_path=os.path.join(directory, metrics) if metrics else None,
            )
        )
    return runs


def load_source(path: str) -> list[RunData]:
    """Load runs from a manifest file **or** a directory.

    * a ``runs.json`` (or any ``*.json`` manifest) file → :func:`load_manifest`;
    * a directory → its ``runs.json`` if present, else auto-discovery of the
      JSONL files inside it (:func:`_discover_dir`).
    """
    path = os.path.normpath(path)
    if os.path.isdir(path):
        manifest = os.path.join(path, "runs.json")
        if os.path.exists(manifest):
            return load_manifest(manifest)
        return _discover_dir(path)
    if os.path.isfile(path):
        return load_manifest(path)
    raise FileNotFoundError(f"no such manifest or directory: {path!r}")
