"""Sliding window, probe evaluation, and group slicing."""

from __future__ import annotations

import torch

from cumobs import ProbeSet, ProbeSetCollection, SlidingWindow
from cumobs.window import GaussianSource


def test_sliding_window_order_and_full():
    w = SlidingWindow(n_features=3, window_size=4, dtype=torch.float64)
    assert not w.is_full
    for t in range(4):
        w.append(torch.full((3,), float(t)))
    assert w.is_full
    out = w.as_tensor()
    assert out.shape == (4, 3)
    # oldest -> newest
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64))

    # overwrite oldest -> ring advances
    w.append(torch.full((3,), 4.0))
    out = w.as_tensor()
    assert torch.allclose(out[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))

    w.reset()
    assert not w.is_full and w.count == 0


def test_sliding_window_partial():
    w = SlidingWindow(n_features=2, window_size=5, dtype=torch.float64)
    w.append(torch.tensor([1.0, 1.0]))
    w.append(torch.tensor([2.0, 2.0]))
    out = w.as_tensor()
    assert out.shape == (2, 2)
    assert torch.allclose(out[:, 0], torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_probeset_batched_eval_matches_full():
    torch.manual_seed(0)
    n, d = 10, 4
    inputs = torch.randn(n, d, dtype=torch.float64)
    model = torch.randn(d, dtype=torch.float64)

    def loss_fn(model, batch):
        return (batch * model).sum(dim=1)

    ps = ProbeSet(inputs, loss_fn, batch_size=3, name="p")
    out = ps.evaluate(model)
    assert out.shape == (n,)
    assert torch.allclose(out, (inputs * model).sum(dim=1))


def test_probeset_groups():
    inputs = torch.arange(6)
    labels = ["a", "a", "b", "b", "b", "a"]
    ps = ProbeSet(inputs, lambda m, b: b.double(), group_labels=labels)
    groups = ps.groups()
    assert set(groups) == {"a", "b"}
    assert torch.equal(groups["a"], torch.tensor([0, 1, 5]))
    assert torch.equal(groups["b"], torch.tensor([2, 3, 4]))


def test_probeset_collection():
    p1 = ProbeSet(torch.arange(3), lambda m, b: b.double(), name="p1")
    p2 = ProbeSet(torch.arange(4), lambda m, b: b.double(), name="p2")
    coll = ProbeSetCollection([p1, p2])
    assert set(coll.names()) == {"p1", "p2"}
    assert coll["p1"] is p1


def test_gaussian_source_not_implemented():
    src = GaussianSource(sigma=0.01, n_samples=4)
    try:
        src.collect(None, None, 0)
    except NotImplementedError:
        return
    raise AssertionError("GaussianSource.collect should raise NotImplementedError")
