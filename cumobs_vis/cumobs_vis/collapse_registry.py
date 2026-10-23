"""The redundancy / collapse groups (plan §4a).

Several scalars are near-duplicates along a *redundancy axis* (e.g.
``effective_rank == exp(von_neumann_entropy)`` exactly). By default the tool
shows **one representative** per axis; a group can be **expanded** to overlay its
members (to visually verify they agree) or **collapsed** back to the
representative. The z-score toggle (plan §4f) puts the members on one scale so
agreement is obvious before collapsing.

This registry is a small, editable config object: add / edit a
:class:`CollapseGroup` to change the redundancy grouping. Membership is by scalar
key; the invert-for-comparability flag (plot ``1 - value``) is read from
:mod:`cumobs_vis.catalog` so it stays consistent with the rest of the tool.

:func:`resolve_panels` turns the set of keys actually present in the loaded data
into the ordered, family-sectioned list of panel units the grid renders — each
unit is either a single scalar or one collapse group. It is pure and unit-tested.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from . import catalog


@dataclass(frozen=True)
class CollapseGroup:
    """One redundancy axis: a representative scalar plus its members."""

    gid: str
    label: str
    family: str
    representative: str
    member_keys: tuple[str, ...]
    note: str = ""


# The default groups (plan §4a). Representative is listed first in ``member_keys``.
COLLAPSE_GROUPS: list[CollapseGroup] = [
    CollapseGroup(
        gid="concentration",
        label="Concentration",
        family="concentration",
        representative="participation_ratio",
        member_keys=(
            "participation_ratio",
            "effective_rank",
            "von_neumann_entropy",
            "top_eigenvalue_fraction",
        ),
        note="effective_rank == exp(von_neumann_entropy) exactly; all are "
        "window-limited (rank <= m-1).",
    ),
    CollapseGroup(
        gid="distance_shape",
        label="Distance — shape",
        family="distances",
        representative="trace_distance",
        member_keys=(
            "trace_distance",
            "classical_fidelity",
        ),
        note="classical_fidelity is plotted as 1 - fidelity for comparability.",
    ),
    CollapseGroup(
        gid="distance_subspace",
        label="Distance — subspace",
        family="distances",
        representative="cka_distance",
        member_keys=(
            "cka_distance",
            "quantum_trace_distance",
            "cka_similarity",
            "quantum_fidelity",
        ),
        note="similarities/fidelities are plotted as 1 - value for comparability.",
    ),
]

_GROUP_BY_KEY: dict[str, CollapseGroup] = {
    k: g for g in COLLAPSE_GROUPS for k in g.member_keys
}
_GROUP_BY_ID: dict[str, CollapseGroup] = {g.gid: g for g in COLLAPSE_GROUPS}


def group_for_key(key: str) -> CollapseGroup | None:
    return _GROUP_BY_KEY.get(key)


def group_by_id(gid: str) -> CollapseGroup | None:
    return _GROUP_BY_ID.get(gid)


def claimed_keys() -> set[str]:
    """Every scalar key that belongs to some collapse group."""
    return set(_GROUP_BY_KEY)


# --- panel resolution ------------------------------------------------------


@dataclass(frozen=True)
class PanelMember:
    """One trace-source within a panel unit (a scalar key + its invert hint)."""

    key: str
    label: str
    invert: bool


@dataclass(frozen=True)
class PanelUnit:
    """A single panel in the grid: either a lone scalar or a collapse group."""

    panel_id: str                 # gid for groups, key for singletons
    kind: str                     # "group" | "scalar"
    family: str
    title: str
    representative: str           # the key shown when collapsed
    members: tuple[PanelMember, ...]
    note: str = ""

    @property
    def is_group(self) -> bool:
        return self.kind == "group"


def _member(key: str) -> PanelMember:
    info = catalog.info_for(key)
    return PanelMember(key=key, label=info.display_label, invert=info.invert)


def resolve_panels(present_keys: set[str]) -> "OrderedDict[str, list[PanelUnit]]":
    """Resolve the family-sectioned panel layout for the keys present in the data.

    Returns an ordered mapping ``family_id -> [PanelUnit]`` following
    :data:`catalog.FAMILY_ORDER`, including only families (and units) that have
    at least one present key. A collapse group appears as a single unit if *any*
    of its members is present; its representative is the configured one if
    present, else the first present member.
    """
    present = set(present_keys)

    # Order candidate keys by catalog insertion order, then any extra present
    # keys not in the catalog (appended, deterministically sorted).
    ordered_keys = list(catalog.SCALARS.keys())
    extras = sorted(present - set(ordered_keys))
    ordered_keys += extras

    sections: "OrderedDict[str, list[PanelUnit]]" = OrderedDict()
    for fam_id, _title in catalog.FAMILY_ORDER:
        sections[fam_id] = []

    emitted_groups: set[str] = set()

    for key in ordered_keys:
        if key not in present:
            continue
        grp = group_for_key(key)
        if grp is not None:
            if grp.gid in emitted_groups:
                continue
            present_members = [m for m in grp.member_keys if m in present]
            if not present_members:
                continue
            rep = grp.representative if grp.representative in present else present_members[0]
            unit = PanelUnit(
                panel_id=grp.gid,
                kind="group",
                family=grp.family,
                title=grp.label,
                representative=rep,
                members=tuple(_member(m) for m in present_members),
                note=grp.note,
            )
            sections.setdefault(grp.family, []).append(unit)
            emitted_groups.add(grp.gid)
        else:
            info = catalog.info_for(key)
            unit = PanelUnit(
                panel_id=key,
                kind="scalar",
                family=info.family,
                title=info.display_label,
                representative=key,
                members=(_member(key),),
                note="",
            )
            sections.setdefault(info.family, []).append(unit)

    # Drop empty families, preserving FAMILY_ORDER.
    return OrderedDict((f, units) for f, units in sections.items() if units)
