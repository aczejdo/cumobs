"""Sliding-window buffer management and buffer sources (§2, §5, §9.2).

``SlidingWindow`` is a fixed-capacity ring buffer of per-sample loss vectors.
``BufferSource`` is the protocol the monitor calls each evaluation step; two
implementations are foreseen:

* ``DynamicalSource`` — appends the current probe loss to its window and yields
  the ``(m, n)`` buffer once full (the dynamical 2RDM).
* ``GaussianSource`` — ``k`` Gaussian-perturbation passes around the current
  ``theta`` (the Gaussian 2RDM).  Per decision D5 this is **not implemented** in
  v1 and raises ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


class SlidingWindow:
    """Fixed-capacity ring buffer of loss vectors, returned oldest -> newest."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_features = n_features
        self.window_size = window_size
        self.device = torch.device(device)
        self.dtype = dtype
        self._buf = torch.zeros(window_size, n_features, device=self.device, dtype=dtype)
        self._ptr = 0
        self._count = 0

    def append(self, loss_vector: Tensor) -> None:
        """Append a ``(n,)`` loss vector, overwriting the oldest row when full."""
        v = loss_vector.reshape(-1).to(self._buf)
        if v.numel() != self.n_features:
            raise ValueError(
                f"loss_vector has {v.numel()} features; expected {self.n_features}"
            )
        self._buf[self._ptr] = v
        self._ptr = (self._ptr + 1) % self.window_size
        self._count += 1

    @property
    def is_full(self) -> bool:
        return self._count >= self.window_size

    @property
    def count(self) -> int:
        """Number of rows currently available (capped at ``window_size``)."""
        return min(self._count, self.window_size)

    def as_tensor(self) -> Tensor:
        """Return the buffered rows ``(m, n)`` ordered oldest -> newest."""
        if self._count < self.window_size:
            return self._buf[: self._count].clone()
        # full: oldest row sits at the write pointer.
        return torch.roll(self._buf, shifts=-self._ptr, dims=0).clone()

    def reset(self) -> None:
        self._buf.zero_()
        self._ptr = 0
        self._count = 0


@runtime_checkable
class BufferSource(Protocol):
    """A source the monitor polls each evaluation step."""

    def collect(self, model, probeset, step: int) -> Tensor | None:
        """Return the current ``(m, n)`` buffer, or ``None`` if not ready."""
        ...


class DynamicalSource:
    """Dynamical 2RDM: window the per-step probe loss vectors.

    Lazily sizes its window on the first ``collect`` (it learns ``n`` from the
    probe evaluation).  Returns the ``(m, n)`` buffer once the window is full.
    """

    def __init__(
        self,
        window_size: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        self.window_size = window_size
        self.dtype = dtype
        self.device = device
        self._window: SlidingWindow | None = None

    @property
    def window(self) -> SlidingWindow | None:
        return self._window

    def collect(self, model, probeset, step: int) -> Tensor | None:
        loss = probeset.evaluate(model)               # (n,)
        if self._window is None:
            device = self.device if self.device is not None else loss.device
            self._window = SlidingWindow(
                n_features=loss.numel(),
                window_size=self.window_size,
                device=device,
                dtype=self.dtype,
            )
        self._window.append(loss)
        if self._window.is_full:
            return self._window.as_tensor()
        return None

    def reset(self) -> None:
        if self._window is not None:
            self._window.reset()


class GaussianSource:
    """Gaussian 2RDM via ``n_samples`` perturbations around the current theta.

    Per decision D5 this is a documented seam only; ``collect`` raises
    ``NotImplementedError`` in v1.  The intended behavior (for reference): draw
    ``k`` i.i.d. ``N(0, sigma^2)`` perturbations of the model parameters,
    evaluate the probe loss under each, restore the parameters, and return the
    resulting ``(k, n)`` buffer immediately (no windowing).
    """

    def __init__(self, sigma: float, n_samples: int) -> None:
        self.sigma = sigma
        self.n_samples = n_samples

    def collect(self, model, probeset, step: int) -> Tensor | None:
        raise NotImplementedError(
            "GaussianSource is a v1 extension seam (decision D5): Gaussian-"
            "perturbation sampling of the 2RDM is not implemented yet."
        )
