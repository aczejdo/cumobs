"""Probe-set management: batched per-sample loss evaluation + group slicing (§5).

A :class:`ProbeSet` bundles fixed probe ``inputs`` with a user-supplied
``loss_fn(model, batch) -> Tensor[(b,)]`` and optional per-sample group labels.
``evaluate`` mini-batches the inputs through ``loss_fn`` under ``torch.no_grad``
and concatenates to a length-``n`` per-sample loss vector — one column ("feature")
of the buffer ``B``.

The package makes **no** assumption about the loss API; ``loss_fn`` is the user's
encapsulation (e.g. for the emergent-misalignment Gaussian 2RDM it should compare
against the uncorrupted model's temperature-0 answers, per the paper).
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

import torch
from torch import Tensor


def _batch(inputs: Any, lo: int, hi: int) -> Any:
    """Slice ``inputs[lo:hi]`` for tensors, sequences, or anything sliceable."""
    return inputs[lo:hi]


class ProbeSet:
    """A named, optionally grouped collection of loss probes."""

    def __init__(
        self,
        inputs: Any,
        loss_fn: Callable[[Any, Any], Tensor],
        group_labels: Sequence[Hashable] | None = None,
        batch_size: int = 64,
        name: str = "probe",
    ) -> None:
        self.inputs = inputs
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.name = name

        self._n = _infer_len(inputs)
        if group_labels is not None:
            group_labels = list(group_labels)
            if len(group_labels) != self._n:
                raise ValueError(
                    f"group_labels has {len(group_labels)} entries; "
                    f"expected {self._n} (one per probe sample)."
                )
        self.group_labels = group_labels

    def __len__(self) -> int:
        return self._n

    @torch.no_grad()
    def evaluate(self, model) -> Tensor:
        """Per-sample loss ``(n,)``, mini-batched, under ``torch.no_grad``."""
        chunks: list[Tensor] = []
        for lo in range(0, self._n, self.batch_size):
            hi = min(lo + self.batch_size, self._n)
            out = self.loss_fn(model, _batch(self.inputs, lo, hi))
            chunks.append(out.reshape(-1))
        if not chunks:
            return torch.empty(0)
        loss = torch.cat(chunks)
        if loss.numel() != self._n:
            raise ValueError(
                f"loss_fn returned {loss.numel()} values for {self._n} probes; "
                "it must return one scalar loss per input sample."
            )
        return loss

    def groups(self) -> dict[Hashable, Tensor]:
        """Map each group label to a ``LongTensor`` of its column indices."""
        if self.group_labels is None:
            return {}
        out: dict[Hashable, list[int]] = {}
        for i, lab in enumerate(self.group_labels):
            out.setdefault(lab, []).append(i)
        return {lab: torch.tensor(idx, dtype=torch.long) for lab, idx in out.items()}


class ProbeSetCollection:
    """Several independently-windowed named :class:`ProbeSet` objects."""

    def __init__(self, probesets: Sequence[ProbeSet] | dict[str, ProbeSet]):
        if isinstance(probesets, dict):
            self._sets = dict(probesets)
        else:
            self._sets = {}
            for ps in probesets:
                if ps.name in self._sets:
                    raise ValueError(f"duplicate ProbeSet name: {ps.name!r}")
                self._sets[ps.name] = ps

    def __len__(self) -> int:
        return len(self._sets)

    def __iter__(self):
        return iter(self._sets.values())

    def items(self):
        return self._sets.items()

    def names(self) -> list[str]:
        return list(self._sets.keys())

    def __getitem__(self, name: str) -> ProbeSet:
        return self._sets[name]


def _infer_len(inputs: Any) -> int:
    try:
        return len(inputs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            "ProbeSet inputs must define __len__ (tensor, list, or dataset)."
        ) from exc
