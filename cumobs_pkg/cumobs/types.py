"""Result containers and shared-intermediate bundle for ``cumobs``.

``SharedIntermediates`` is the reuse contract from §4: it is built **once per
measurement** and every observable function reads from it so nothing large is
recomputed.  ``Measurement`` is the structured result; ``to_flat_dict`` turns it
into the flat ``{str: float | Tensor}`` mapping the loggers consume (§7).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class SharedIntermediates:
    """Everything shared across the observable catalog for one measurement.

    Built by :func:`cumobs.observables.build_intermediates` in the dependency
    order of §4.  Only ``C``/``M3``/``K`` scale as ``n**2`` (and only when
    ``materialize_matrices``); ``V`` scales as ``n * top_k``; everything else is
    ``O(m**2) + O(m n)``.
    """

    # --- scalars / bookkeeping ------------------------------------------------
    m: int
    n: int
    D: float
    use_gram: bool
    rank: int                       # # eigenvalues above eig_threshold * mu_max
    top_k_eff: int                  # principal directions actually computed

    # --- first / second order shared blocks ----------------------------------
    E_L: Tensor                     # (n,)   per-feature mean (centering mean, 1/m)
    B_c: Tensor                     # (m,n)  centered buffer
    S: Tensor                       # (m,)   aggregate variable (row sums of B_c)
    S2: Tensor                      # (m,)
    S3: Tensor                      # (m,)
    S4: Tensor                      # (m,)
    s2: float                       # ||S||^2

    # --- spectral ------------------------------------------------------------
    eigvals: Tensor                 # (n,)   covariance spectrum, padded, desc, clipped
    lam_top: Tensor                 # (top_k_eff,) leading eigenvalues
    P: Tensor                       # (m, top_k_eff) principal projections
    V: Tensor | None                # (n, top_k_eff) sample-space eigenvectors or None

    # --- fourth-order helper -------------------------------------------------
    Q: Tensor                       # (m,m) symmetric kurtosis contraction operator

    # --- optionally materialized n x n objects -------------------------------
    G0: Tensor | None = None        # (m,m) Gram matrix (gram path only)
    C: Tensor | None = None         # (n,n) covariance (materialize_matrices)
    M3: Tensor | None = None        # (n,n) contracted skew (materialize_matrices)
    K: Tensor | None = None         # (n,n) contracted kurtosis (materialize_matrices)

    # --- numerical health ----------------------------------------------------
    health: dict[str, float] = field(default_factory=dict)


@dataclass
class Measurement:
    """Structured observable result for a single computation step.

    ``factors`` carries the top-``k`` eigen-factors (``V``, ``lam_top``) when
    ``compute_eigenvectors`` is on; it is consumed by the monitor for the CKA
    distance and is **never** auto-logged (like ``matrices``).

    ``aux`` carries observables that do not fit the float ``scalars`` bucket —
    the §6 BBP ``active_probe_mask`` is a Python big-int serialized as a hex
    string, for example.  Its values are JSON-serializable primitives (``str`` /
    ``int`` / ``list``) and pass straight through ``to_flat_dict`` (the
    array-capable loggers persist them as-is).
    """

    step: int
    scalars: dict[str, float] = field(default_factory=dict)
    vectors: dict[str, Tensor] = field(default_factory=dict)
    matrices: dict[str, Tensor] = field(default_factory=dict)
    health: dict[str, float] = field(default_factory=dict)
    factors: dict[str, Tensor] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(
        self,
        vector_policy: str = "summary",
        vector_overrides: dict[str, str] | None = None,
        log_health: bool = True,
        drop_keys: Iterable[str] = (),
        drop_prefixes: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Flatten to a ``{key: value}`` mapping for a logger (§7).

        Parameters
        ----------
        vector_policy:
            ``"summary"`` (max / mean / top1 / L2 per vector), ``"full"`` (raw
            array), or ``"off"`` (skip).
        vector_overrides:
            Per-key policy overrides (e.g. always-full ``covariance_spectrum``).
        log_health:
            Include the ``health/*`` scalars.
        drop_keys / drop_prefixes:
            Keys to suppress from the flattened output (the §8 consolidation,
            opt-in via ``LoggingConfig.trim_consolidated``).  ``drop_keys`` is an
            exact-match set; ``drop_prefixes`` matches by ``startswith``.  Both
            test against the *source* key (so dropping ``"diagonal_third_cumulant"``
            also removes its ``/max`` … summaries).
        """
        overrides = vector_overrides or {}
        drop = set(drop_keys)
        prefixes = tuple(drop_prefixes)
        out: dict[str, Any] = {}

        def _dropped(key: str) -> bool:
            return key in drop or (bool(prefixes) and key.startswith(prefixes))

        # scalars logged directly
        for key, val in self.scalars.items():
            if _dropped(key):
                continue
            out[key] = val

        # vectors governed by policy
        for key, vec in self.vectors.items():
            if _dropped(key):
                continue
            policy = overrides.get(key, vector_policy)
            if policy == "off":
                continue
            if policy == "full":
                out[key] = vec
                continue
            # "summary"
            out.update(_vector_summary(key, vec))

        # aux (non-float observables) pass through verbatim
        for key, val in self.aux.items():
            if _dropped(key):
                continue
            out[key] = val

        if log_health:
            for key, val in self.health.items():
                if _dropped(f"health/{key}") or _dropped(key):
                    continue
                out[f"health/{key}"] = val

        # matrices are never auto-logged (§7)
        return out


def _vector_summary(key: str, vec: Tensor) -> dict[str, float]:
    """``max`` / ``mean`` / ``top1`` / ``L2`` summary of a vector (§7)."""
    if vec.numel() == 0:
        return {
            f"{key}/max": float("nan"),
            f"{key}/mean": float("nan"),
            f"{key}/top1": float("nan"),
            f"{key}/l2": float("nan"),
        }
    v = vec.detach().to(torch.float64)
    return {
        f"{key}/max": float(v.max()),
        f"{key}/mean": float(v.mean()),
        f"{key}/top1": float(v.reshape(-1)[0]),
        f"{key}/l2": float(torch.linalg.vector_norm(v)),
    }
