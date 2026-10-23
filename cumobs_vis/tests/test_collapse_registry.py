"""Tests for the collapse registry + panel resolution."""

from cumobs_vis.collapse_registry import (
    claimed_keys,
    group_for_key,
    resolve_panels,
)


def test_group_membership():
    assert group_for_key("effective_rank").gid == "concentration"
    assert group_for_key("classical_fidelity").gid == "distance_shape"
    assert group_for_key("spectral_heat_capacity") is None   # a singleton
    claimed = claimed_keys()
    assert "participation_ratio" in claimed
    assert "spectral_gap" not in claimed


def test_singletons_and_groups_and_family_order():
    present = {
        "spectral_heat_capacity",       # detector_scale singleton
        "participation_ratio", "effective_rank",  # concentration group
        "trace_distance", "classical_fidelity",   # distance_shape group
        "excess_loss",                  # misc singleton
    }
    sections = resolve_panels(present)
    # family order preserved, only non-empty families present.
    fams = list(sections.keys())
    assert fams.index("detector_scale") < fams.index("concentration")
    assert fams.index("concentration") < fams.index("distances")
    assert fams.index("distances") < fams.index("misc")

    conc = sections["concentration"]
    assert len(conc) == 1 and conc[0].is_group
    assert conc[0].panel_id == "concentration"
    assert conc[0].representative == "participation_ratio"
    member_keys = {m.key for m in conc[0].members}
    assert member_keys == {"participation_ratio", "effective_rank"}  # only present ones

    dist = sections["distances"]
    assert len(dist) == 1 and dist[0].panel_id == "distance_shape"


def test_representative_fallback_when_absent():
    # representative (participation_ratio) absent, but a member is present ->
    # the group still shows, using the first present member as representative.
    sections = resolve_panels({"effective_rank"})
    conc = sections["concentration"]
    assert len(conc) == 1
    assert conc[0].representative == "effective_rank"


def test_unknown_scalar_goes_to_other_family():
    sections = resolve_panels({"some_new_scalar"})
    assert "other" in sections
    assert sections["other"][0].panel_id == "some_new_scalar"


def test_member_invert_flag():
    sections = resolve_panels({"trace_distance", "classical_fidelity"})
    members = {m.key: m for m in sections["distances"][0].members}
    assert members["classical_fidelity"].invert is True
    assert members["trace_distance"].invert is False
