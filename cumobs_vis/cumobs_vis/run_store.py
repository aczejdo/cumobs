"""runStore — the loaded runs plus their visibility and per-group sub-series state.

Color is identity: a run keeps one color in **every** panel, so overlaying two
probe sets on one observable is two colored lines (plan §3). Toggling a run off
hides its traces, bands, and validity spans everywhere at once (plan §4b); the
panel grid reads visibility from here on every rebuild.

Per-group sub-series (``group_shc/<label>`` etc.) are a *second* form of
probe-set subset (plan §1): each run can be expanded to toggle its groups on/off,
drawn in the parent run's color with a distinct dash.
"""

from __future__ import annotations

from .data_loader import RunData


class RunStore:
    def __init__(self, runs: list[RunData] | None = None) -> None:
        self._runs: list[RunData] = []
        self._visible: dict[str, bool] = {}
        self._expanded: dict[str, bool] = {}
        # (run_name, base_key, label) -> bool ; default off.
        self._group_visible: dict[tuple[str, str, str], bool] = {}
        for r in runs or []:
            self.add_run(r)

    # -- membership --------------------------------------------------------

    def add_run(self, run: RunData) -> None:
        if any(r.name == run.name for r in self._runs):
            # keep names unique so visibility keys don't collide.
            base, i = run.name, 2
            while any(r.name == f"{base} ({i})" for r in self._runs):
                i += 1
            run.name = f"{base} ({i})"
        self._runs.append(run)
        self._visible[run.name] = True
        self._expanded[run.name] = False

    @property
    def runs(self) -> list[RunData]:
        return list(self._runs)

    def by_name(self, name: str) -> RunData | None:
        for r in self._runs:
            if r.name == name:
                return r
        return None

    # -- run visibility ----------------------------------------------------

    def is_visible(self, name: str) -> bool:
        return self._visible.get(name, True)

    def set_visible(self, name: str, value: bool) -> None:
        self._visible[name] = value

    def visible_runs(self) -> list[RunData]:
        return [r for r in self._runs if self._visible.get(r.name, True)]

    # -- per-run expansion + group sub-series ------------------------------

    def is_expanded(self, name: str) -> bool:
        return self._expanded.get(name, False)

    def set_expanded(self, name: str, value: bool) -> None:
        self._expanded[name] = value

    def group_bases(self, run: RunData) -> list[str]:
        """Per-group base observables present for a run (e.g. ``group_shc``)."""
        return sorted(run.groups.keys())

    def is_group_visible(self, run_name: str, base_key: str, label: str) -> bool:
        return self._group_visible.get((run_name, base_key, label), False)

    def set_group_visible(self, run_name: str, base_key: str, label: str, value: bool) -> None:
        self._group_visible[(run_name, base_key, label)] = value

    # -- aggregate views for the panel grid --------------------------------

    def present_scalar_keys(self, only_visible: bool = True) -> set[str]:
        """Union of primary scalar keys across (visible) runs — feeds the panel
        resolver so a panel exists iff at least one run supplies its data."""
        runs = self.visible_runs() if only_visible else self._runs
        keys: set[str] = set()
        for r in runs:
            keys |= r.primary_scalar_keys()
        return keys

    def runs_with_config(self, only_visible: bool = True) -> list[RunData]:
        runs = self.visible_runs() if only_visible else self._runs
        return [r for r in runs if r.config]
