"""End-to-end monitor behavior — §10 test 12 (synthetic Fig. 1 transition)."""

from __future__ import annotations

import torch

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    LoggingConfig,
    ProbeSet,
)


def _run_synthetic(seed: int = 0):
    """Inject l(t) = a(t)*v1 + eps, with v1 supported on one planted group.

    Phase 1 (steps < 20): isotropic low-amplitude noise.
    Phase 2 (steps >= 20): a strong rank-1 mode localized on group "A".
    """
    torch.manual_seed(seed)
    n = 24
    n_a = 8
    labels = ["A"] * n_a + ["B"] * (n - n_a)

    v1 = torch.zeros(n, dtype=torch.float64)
    v1[:n_a] = torch.randn(n_a, dtype=torch.float64)
    v1 = v1 / torch.linalg.vector_norm(v1)

    state = {"loss": torch.zeros(n, dtype=torch.float64)}

    def loss_fn(model, batch_idx):
        return state["loss"][batch_idx]

    probes = ProbeSet(torch.arange(n), loss_fn, group_labels=labels, name="probe")
    cfg = CumulantConfig(
        window_size=10,
        eval_interval=1,
        update_interval=5,
        top_k=1,
        compute_eigenvectors=True,
        per_group=True,
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(
        None, probes, cfg, logger=logger,
        logging_config=LoggingConfig(vector_policy="summary"),
    )

    transition = 20
    history = []
    for step in range(60):
        if step < transition:
            state["loss"] = 0.01 * torch.randn(n, dtype=torch.float64)
        else:
            a = 5.0 * torch.randn((), dtype=torch.float64)
            state["loss"] = a * v1 + 0.01 * torch.randn(n, dtype=torch.float64)
        meas = monitor.step(step)
        if meas is not None:
            history.append((step, meas))
    return history, logger, transition


def test_end_to_end_transition_and_localization():
    history, logger, transition = _run_synthetic(seed=0)

    # window_size=10 fills by step 10; pre-transition measurement at step 10
    early = [m for s, m in history if s <= transition - 5]
    late = [m for s, m in history if s >= transition + 10]
    assert early and late

    shc_early = max(m.scalars["spectral_heat_capacity"] for m in early)
    shc_late = max(m.scalars["spectral_heat_capacity"] for m in late)

    # SHC spikes across the transition
    assert shc_late > 100 * shc_early

    # top eigenvector localizes on the planted group A (block energy)
    late_m = late[-1]
    energy_a = float(late_m.vectors["block_energy/A"].sum())
    energy_b = float(late_m.vectors["block_energy/B"].sum())
    assert energy_a > 0.95
    assert energy_a > 20 * (energy_b + 1e-12)

    # monitor + logger ran start-to-finish
    assert len(logger.history) == len(history)
    assert any("spectral_heat_capacity" in row for row in logger.history)


def test_monitor_distances_populated():
    _, logger, _ = _run_synthetic(seed=1)
    # after the first measurement a "previous"-reference distance appears
    rows_with_fidelity = [r for r in logger.history if "classical_fidelity" in r]
    assert len(rows_with_fidelity) >= 1
    for r in rows_with_fidelity:
        f = r["classical_fidelity"]
        assert 0.0 <= f <= 1.0 or f != f  # in range or NaN (dead window)


def test_quantum_distance_flavor_runs():
    torch.manual_seed(2)
    n = 12
    state = {"loss": torch.zeros(n, dtype=torch.float64)}
    probes = ProbeSet(torch.arange(n), lambda m, b: state["loss"][b], name="p")
    cfg = CumulantConfig(
        window_size=8, eval_interval=1, update_interval=4,
        materialize_matrices=True, distance_flavor="both",
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(None, probes, cfg, logger=logger)
    for step in range(40):
        state["loss"] = torch.randn(n, dtype=torch.float64)
        monitor.step(step)
    qrows = [r for r in logger.history if "quantum_fidelity" in r]
    assert len(qrows) >= 1
    for r in qrows:
        assert 0.0 <= r["quantum_fidelity"] <= 1.0
