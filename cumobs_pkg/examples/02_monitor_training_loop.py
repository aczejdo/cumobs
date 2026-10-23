"""Example 02 — the monitor watching a real torch training loop.

A small linear-regression model is trained with SGD while a
:class:`~cumobs.CumulantMonitor` watches a fixed probe set. As the model
converges, the per-sample probe-loss fluctuations shrink and reorganize, which
the monitor reports through the spectral heat capacity (SHC), participation
ratio, and per-group block energy.

This example shows the realistic wiring:

* a ``loss_fn(model, batch) -> Tensor[(b,)]`` that returns one loss per probe,
* probes split into two groups ("near" vs "far" from the training data) so the
  per-group decomposition and ``block_energy`` are meaningful,
* an :class:`~cumobs.InMemoryLogger` whose history we print at the end.

The common, clean pattern is to let ``inputs`` be plain indices and close over
the real probe tensors — so ``loss_fn`` can look up both inputs *and* targets.

Run::

    python examples/02_monitor_training_loop.py
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    LoggingConfig,
    ProbeSet,
)


def main() -> None:
    torch.manual_seed(0)
    d = 8                                   # input dimension
    w_true = torch.randn(d, dtype=torch.float64)

    def target(x):                          # the function the model learns
        return x @ w_true

    # --- training data --------------------------------------------------------
    X_train = torch.randn(512, d, dtype=torch.float64)
    y_train = target(X_train) + 0.01 * torch.randn(512, dtype=torch.float64)

    # --- probe set: 24 "near" (in-distribution) + 16 "far" (harder) samples ---
    n_near, n_far = 24, 16
    probe_x = torch.cat(
        [
            torch.randn(n_near, d, dtype=torch.float64),         # near
            3.0 * torch.randn(n_far, d, dtype=torch.float64),    # far (larger norm)
        ]
    )
    probe_y = target(probe_x)
    labels = ["near"] * n_near + ["far"] * n_far

    # inputs are indices; loss_fn closes over probe_x / probe_y.
    def loss_fn(model, idx):
        pred = model(probe_x[idx]).squeeze(-1)
        return (pred - probe_y[idx]) ** 2           # (b,) per-sample squared error

    probes = ProbeSet(
        inputs=torch.arange(len(probe_x)),
        loss_fn=loss_fn,
        group_labels=labels,
        name="reg_probes",
    )

    # --- model + monitor ------------------------------------------------------
    model = nn.Linear(d, 1, bias=False).double()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    cfg = CumulantConfig(
        window_size=15,
        eval_interval=1,        # append a probe-loss row every step
        update_interval=5,      # compute observables every 5 steps
        order=4,
        top_k=2,
        compute_eigenvectors=True,
        per_group=True,
        distance_flavor="classical",
        distance_reference="previous",
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(
        model, probes, cfg, logger=logger,
        logging_config=LoggingConfig(vector_policy="summary"),
    )

    # --- training loop --------------------------------------------------------
    print(f"{'step':>5} {'train_loss':>11} {'SHC':>11} {'PR':>6} "
          f"{'blockE_far':>11} {'fidelity':>9}")
    for step in range(90):
        # one SGD step
        idx = torch.randint(0, len(X_train), (32,))
        pred = model(X_train[idx]).squeeze(-1)
        train_loss = ((pred - y_train[idx]) ** 2).mean()
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        meas = monitor.step(step)
        if meas is not None:
            shc = meas.scalars["spectral_heat_capacity"]
            pr = meas.scalars["participation_ratio"]
            block_far = float(meas.vectors["block_energy/far"][0])
            fid = meas.scalars.get("classical_fidelity", float("nan"))
            print(f"{step:>5} {float(train_loss.detach()):>11.4e} {shc:>11.4e} "
                  f"{pr:>6.2f} {block_far:>11.3f} {fid:>9.4f}")

    print("\nWhat to notice:")
    print("  * SHC and the covariance scale fall as the model converges")
    print("    (the probe-loss fluctuations collapse).")
    print("  * block_energy/far stays high: the larger-norm 'far' probes carry")
    print("    most of the leading loss-covariance mode.")
    print("  * classical_fidelity ~ 1 between adjacent measurements once the")
    print("    spectrum settles; it dips while the structure is still changing.")
    print("  * (run it longer and the probe loss converges to a constant: the")
    print("    window goes 'dead' and PR returns NaN -- the documented")
    print("    degenerate case in observables.md.)")
    print(f"\nlogged {len(logger.history)} measurement rows "
          f"(InMemoryLogger.as_dict() / .as_dataframe() for the full history).")


if __name__ == "__main__":
    main()
