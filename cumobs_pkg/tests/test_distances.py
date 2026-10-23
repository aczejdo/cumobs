"""Distance validation — §10 test 8, plus quantum-flavor checks (decision D2)."""

from __future__ import annotations

import math

import torch

from cumobs import (
    classical_fidelity,
    quantum_fidelity,
    quantum_trace_distance,
    trace_distance,
)


def test_identity_and_range():
    torch.manual_seed(0)
    p = torch.rand(8, dtype=torch.float64)
    assert abs(classical_fidelity(p, p) - 1.0) < 1e-12
    assert trace_distance(p, p) < 1e-12
    q = torch.rand(8, dtype=torch.float64)
    f = classical_fidelity(p, q)
    t = trace_distance(p, q)
    assert 0.0 <= f <= 1.0
    assert 0.0 <= t <= 1.0


def test_two_point_closed_form():
    a, b = 0.7, 0.25
    p = torch.tensor([a, 1 - a], dtype=torch.float64)
    q = torch.tensor([b, 1 - b], dtype=torch.float64)
    f_expected = (math.sqrt(a * b) + math.sqrt((1 - a) * (1 - b))) ** 2
    t_expected = abs(a - b)
    assert abs(classical_fidelity(p, q) - f_expected) < 1e-12
    assert abs(trace_distance(p, q) - t_expected) < 1e-12


def test_monotone_diverging_pair():
    p = torch.tensor([2.0, 2.0], dtype=torch.float64)
    prev_t, prev_f = -1.0, 2.0
    for d in [0.0, 0.5, 1.0, 1.5, 1.9]:
        q = torch.tensor([2.0 + d, 2.0 - d], dtype=torch.float64)
        t = trace_distance(p, q)
        f = classical_fidelity(p, q)
        assert t >= prev_t - 1e-12      # TV increases as the pair diverges
        assert f <= prev_f + 1e-12      # fidelity decreases
        prev_t, prev_f = t, f


def test_degenerate_returns_nan():
    p = torch.zeros(5, dtype=torch.float64)
    q = torch.rand(5, dtype=torch.float64)
    assert math.isnan(classical_fidelity(p, q))
    assert math.isnan(trace_distance(p, q))


def test_quantum_identity_and_range():
    torch.manual_seed(1)
    A = torch.randn(6, 6, dtype=torch.float64)
    rho = A @ A.t()                      # PSD
    assert abs(quantum_fidelity(rho, rho) - 1.0) < 1e-9
    assert quantum_trace_distance(rho, rho) < 1e-9


def test_quantum_is_eigenvector_sensitive():
    # same spectrum, rotated eigenvectors: classical sees them as identical,
    # quantum does not (decision D2).
    spec = torch.tensor([0.7, 0.3], dtype=torch.float64)
    rho = torch.diag(spec)
    theta = 0.6
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
    )
    sigma = R @ rho @ R.t()

    # classical fidelity on the spectra is exactly 1 (same spectrum)
    assert abs(classical_fidelity(spec, spec) - 1.0) < 1e-12
    # quantum fidelity strictly less than 1 under the rotation
    qf = quantum_fidelity(rho, sigma)
    assert qf < 1.0 - 1e-3
    assert quantum_trace_distance(rho, sigma) > 1e-3
