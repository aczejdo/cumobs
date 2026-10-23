"""The drop-in training monitor (§5, §5.1).

``CumulantMonitor.step(global_step)`` is the orchestrator: on evaluation steps it
polls each probe's :class:`~cumobs.window.BufferSource`; on update steps it runs
:func:`~cumobs.observables.compute_all`, computes inter-measurement distances
against a retained reference (decision D4), and logs the flattened result.

It returns the :class:`~cumobs.types.Measurement` on a computation step (or a
``{probe_name: Measurement}`` dict for a :class:`ProbeSetCollection`) and
``None`` otherwise; logging happens internally.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .config import CumulantConfig, LoggingConfig
from .distances import (
    bures_angle,
    bures_distance,
    cka_distance,
    cka_similarity,
    classical_fidelity,
    denoised_spectral_drift,
    fisher_rao_distance,
    quantum_fidelity,
    quantum_trace_distance,
    signal_energy_drift,
    spectral_drift,
    spectral_drift_snr,
    trace_distance,
)
from .logging import Logger
from .observables import compute_all, integrated_autocorr_time
from .probes import ProbeSet, ProbeSetCollection
from .types import Measurement
from .window import BufferSource, DynamicalSource

_SPECTRUM_KEY = {
    "covariance": "covariance_spectrum",
    "contracted_third_cumulant": "contracted_third_cumulant_spectrum",
    "contracted_fourth_cumulant": "contracted_fourth_cumulant_spectrum",
}
_MATRIX_KEY = {
    "covariance": "covariance",
    "contracted_third_cumulant": "contracted_third_cumulant_matrix",
    "contracted_fourth_cumulant": "contracted_fourth_cumulant_matrix",
}


class CumulantMonitor:
    """Schedule probe evaluation, observable computation, distances, logging."""

    def __init__(
        self,
        model,
        probes: ProbeSet | ProbeSetCollection,
        cfg: CumulantConfig,
        logger: Logger | None = None,
        source: BufferSource | None = None,
        logging_config: LoggingConfig | None = None,
        contraction: Tensor | None = None,
        reference_loss: Tensor | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.logging_config = logging_config or LoggingConfig()
        self.contraction = contraction
        self.reference_loss = reference_loss

        if isinstance(probes, ProbeSet):
            self._probes = {probes.name: probes}
            self._single = True
        elif isinstance(probes, ProbeSetCollection):
            self._probes = dict(probes.items())
            self._single = False
        else:
            raise TypeError("probes must be a ProbeSet or ProbeSetCollection")

        # one source per probe (independently windowed).
        self._sources: dict[str, BufferSource] = {}
        for name in self._probes:
            if source is not None and self._single:
                self._sources[name] = source
            else:
                self._sources[name] = DynamicalSource(cfg.window_size, cfg.dtype)

        self._latest: dict[str, Tensor | None] = {n: None for n in self._probes}
        # per-probe reference history (cheap n-vectors; matrices only if quantum;
        # eigen-factors only if compute_cka_distance).
        self._spec_hist: dict[str, list[Tensor]] = {n: [] for n in self._probes}
        self._mat_hist: dict[str, list[Tensor | None]] = {n: [] for n in self._probes}
        self._fact_hist: dict[str, list[dict | None]] = {n: [] for n in self._probes}

        # §7-step-2 eigenvector tracking + §5 per-mode CSD state.
        self._canon_V: dict[str, Tensor | None] = {n: None for n in self._probes}
        self._lambda_hist: dict[str, list[Tensor]] = {n: [] for n in self._probes}
        self._csd_max_hist = 64

    # -- public API --------------------------------------------------------

    def step(self, global_step: int) -> Measurement | dict[str, Measurement] | None:
        cfg = self.cfg
        if global_step % cfg.eval_interval == 0:
            for name, src in self._sources.items():
                self._latest[name] = src.collect(
                    self.model, self._probes[name], global_step
                )

        if global_step % cfg.update_interval != 0:
            return None

        measurements: dict[str, Measurement] = {}
        for name, ps in self._probes.items():
            buf = self._latest.get(name)
            if buf is None:
                continue
            groups = ps.groups() if cfg.per_group else None
            meas = compute_all(
                buf,
                cfg,
                groups=groups,
                step=global_step,
                contraction=self.contraction,
                reference_loss=self.reference_loss,
            )
            meas.scalars.update(self._distances(name, meas))
            self._track_and_csd(name, meas)
            self._record_reference(name, meas)
            self._emit(name, meas, global_step)
            measurements[name] = meas

        if not measurements:
            return None
        if self._single:
            return next(iter(measurements.values()))
        return measurements

    # -- internals ---------------------------------------------------------

    def _distances(self, name: str, meas: Measurement) -> dict[str, float]:
        cfg = self.cfg
        out: dict[str, float] = {}
        spec = meas.vectors.get(_SPECTRUM_KEY[cfg.distance_source])
        if spec is None:
            return out
        ref_spec, ref_mat, ref_fact = self._reference_for(name)
        if ref_spec is None:
            return out  # first measurement: no reference yet (logged as absent)

        if cfg.distance_flavor in ("classical", "both"):
            out["classical_fidelity"] = classical_fidelity(ref_spec, spec, cfg.eps)
            out["trace_distance"] = trace_distance(ref_spec, spec, cfg.eps)
            # scale-sensitive companion (classical distances are scale-blind).
            drift = spectral_drift(ref_spec, spec, cfg.eps)
            out["spectral_drift"] = drift
            # §2 geodesic metric form of the fidelity (preferred default).
            out["fisher_rao_distance"] = fisher_rao_distance(ref_spec, spec, cfg.eps)
            # §4 noise-floored drift — only on the covariance source, where the
            # null floor / MP edge are defined.
            if cfg.distance_source == "covariance":
                mu0 = meas.scalars.get("spectral_drift/null_mean")
                s0 = meas.scalars.get("spectral_drift/null_sd")
                if mu0 is not None:
                    out["denoised_spectral_drift"] = denoised_spectral_drift(
                        drift, mu0, cfg.eps
                    )
                    if s0 is not None:
                        out["spectral_drift_snr"] = spectral_drift_snr(
                            drift, mu0, s0, cfg.eps
                        )
                edge = meas.scalars.get("mp_edge_upper")
                if edge is not None:
                    out["signal_energy_drift"] = signal_energy_drift(
                        ref_spec, spec, edge, cfg.eps
                    )

        if cfg.distance_flavor in ("quantum", "both"):
            cur_mat = meas.matrices.get(_MATRIX_KEY[cfg.distance_source])
            # quantum distances need a genuine density matrix (PSD); only the
            # covariance qualifies, and both reference + current must be present.
            if (
                cfg.distance_source == "covariance"
                and ref_mat is not None
                and cur_mat is not None
            ):
                out["quantum_fidelity"] = quantum_fidelity(ref_mat, cur_mat, cfg.eps)
                out["quantum_trace_distance"] = quantum_trace_distance(
                    ref_mat, cur_mat, cfg.eps
                )
                # §2 geodesic metric forms of the quantum fidelity.
                out["bures_distance"] = bures_distance(ref_mat, cur_mat, cfg.eps)
                out["bures_angle"] = bures_angle(ref_mat, cur_mat, cfg.eps)

        # CKA representation drift (eigenvector-sensitive, O(n k^2)).
        if cfg.compute_cka_distance:
            cur_fact = meas.factors
            if ref_fact and "V" in ref_fact and cur_fact and "V" in cur_fact:
                args = (
                    ref_fact["V"], ref_fact["lam_top"],
                    cur_fact["V"], cur_fact["lam_top"], cfg.eps,
                )
                out["cka_distance"] = cka_distance(*args)
                out["cka_similarity"] = cka_similarity(*args)
        return out

    # -- §7 step 2: eigenvector tracking + §5 per-mode CSD -----------------

    def _track_and_csd(self, name: str, meas: Measurement) -> None:
        """Track eigenvector identity across measurements (Hungarian on
        ``|⟨v_a(t), v_b(t+1)⟩|``) and emit ``mode_permutation`` /
        ``crossing_detected``, plus the per-mode CSD series
        ``principal_autocorr_lag1`` / ``principal_autocorr_time`` over a ring
        buffer of the tracked per-mode eigenvalue (the mode's order parameter)."""
        cfg = self.cfg
        if not cfg.track_modes:
            return
        fac = meas.factors
        if "V" not in fac or fac.get("V") is None:
            return
        cur_V = fac["V"].detach().to(torch.float64)        # (n, kc)
        cur_lam = fac["lam_top"].detach().to(torch.float64)  # (kc,)
        kc = cur_V.shape[1]

        prev = self._canon_V.get(name)
        if prev is None or prev.shape[1] == 0 or kc == 0:
            canon_V, canon_lam = cur_V, cur_lam
            perm = list(range(kc))
            crossing = 0.0
        else:
            overlap = (prev.transpose(0, 1) @ cur_V).abs()  # (kp, kc)
            assign = _match_modes(overlap)                  # prev slot -> cur col
            kp = prev.shape[1]
            cols: list[int] = []
            for s in range(kp):
                if s in assign:
                    cols.append(assign[s])
            for j in range(kc):
                if j not in cols:
                    cols.append(j)
            canon_V = cur_V[:, cols]
            canon_lam = cur_lam[cols]
            perm = cols
            common = min(kp, kc)
            crossing = float(any(perm[s] != s for s in range(min(common, len(perm)))))

        self._canon_V[name] = canon_V
        meas.vectors["mode_permutation"] = torch.tensor(perm, dtype=torch.float64)
        meas.scalars["crossing_detected"] = crossing

        # CSD: ring buffer of the canonical per-mode eigenvalues (NaN-padded).
        K = cfg.top_k
        hist = self._lambda_hist[name]
        padded = torch.full((K,), float("nan"), dtype=torch.float64)
        kk = min(K, canon_lam.numel())
        padded[:kk] = canon_lam[:kk]
        hist.append(padded)
        if len(hist) > self._csd_max_hist:
            hist.pop(0)
        if len(hist) >= 4 and kc > 0:
            series = torch.stack(hist, dim=0)               # (T, K)
            lag1: list[float] = []
            tau: list[float] = []
            for s in range(kc):
                col = series[:, s]
                col = col[torch.isfinite(col)]
                if col.numel() >= 4:
                    t_s, r1 = integrated_autocorr_time(col)
                    tau.append(t_s)
                    lag1.append(r1)
                else:
                    tau.append(float("nan"))
                    lag1.append(float("nan"))
            meas.vectors["principal_autocorr_lag1"] = torch.tensor(lag1, dtype=torch.float64)
            meas.vectors["principal_autocorr_time"] = torch.tensor(tau, dtype=torch.float64)

    def _reference_for(
        self, name: str
    ) -> tuple[Tensor | None, Tensor | None, dict | None]:
        """Pick the reference spectrum/matrix/factors per ``cfg.distance_reference`` (D4).

        TODO: a transition-aware baseline that re-bases when a phase transition
        is detected (decision D4) is not implemented; ``"baseline"`` is the
        first measurement for now.
        """
        cfg = self.cfg
        specs = self._spec_hist[name]
        mats = self._mat_hist[name]
        facts = self._fact_hist[name]
        if not specs:
            return None, None, None
        if cfg.distance_reference == "previous":
            return specs[-1], mats[-1], facts[-1]
        if cfg.distance_reference == "baseline":
            return specs[0], mats[0], facts[0]
        # lag:k
        k = cfg.lag()
        if len(specs) >= k:
            return specs[-k], mats[-k], facts[-k]
        return None, None, None

    def _record_reference(self, name: str, meas: Measurement) -> None:
        cfg = self.cfg
        self._spec_hist[name].append(meas.vectors[_SPECTRUM_KEY[cfg.distance_source]])
        self._mat_hist[name].append(
            meas.matrices.get(_MATRIX_KEY[cfg.distance_source])
        )
        self._fact_hist[name].append(meas.factors if meas.factors else None)

    def _emit(self, name: str, meas: Measurement, step: int) -> None:
        if self.logger is None:
            return
        lc = self.logging_config
        drop_keys, drop_prefixes = lc.dropped()
        flat: dict[str, Any] = meas.to_flat_dict(
            vector_policy=lc.vector_policy,
            vector_overrides=lc.vector_overrides,
            log_health=lc.log_health,
            drop_keys=drop_keys,
            drop_prefixes=drop_prefixes,
        )
        if not self._single:
            flat = {f"{name}/{k}": v for k, v in flat.items()}
        self.logger.log(flat, step)


def _match_modes(overlap: Tensor) -> dict[int, int]:
    """Assign previous mode slots to current eigenvector columns maximizing total
    ``|overlap|`` (Hungarian via SciPy if available, else a greedy fallback).

    Returns ``{prev_slot: cur_col}``."""
    a, b = overlap.shape
    if a == 0 or b == 0:
        return {}
    try:  # optimal assignment
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        rows, cols = linear_sum_assignment((-overlap).detach().cpu().numpy())
        return {int(i): int(j) for i, j in zip(rows, cols)}
    except Exception:  # greedy fallback (no SciPy)
        ranked = sorted(
            ((float(overlap[i, j]), i, j) for i in range(a) for j in range(b)),
            reverse=True,
        )
        used_i: set[int] = set()
        used_j: set[int] = set()
        out: dict[int, int] = {}
        for _, i, j in ranked:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            out[i] = j
        return out
