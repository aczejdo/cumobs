"""``cumobs`` — cumulant observables for deep-learning phase-transition monitoring.

A lean, framework-free training monitor that tracks scalar cumulants up to 4th
order plus spectral diagnostics of the per-sample loss covariance (the "2RDM")
over a sliding window.  See ``cumulant_observables_spec.md`` for the full design.

Quick start (§5.1)::

    from cumobs import CumulantConfig, CumulantMonitor, ProbeSet, JSONLLogger

    probes = ProbeSet(inputs=probe_inputs, loss_fn=my_loss,
                      group_labels=labels, name="mix")
    cfg = CumulantConfig(window_size=20, update_interval=10, top_k=3,
                         compute_eigenvectors=True, per_group=True)
    monitor = CumulantMonitor(model, probes, cfg, logger=JSONLLogger("run.jsonl"))

    for step in range(num_steps):
        train_step(...)
        monitor.step(step)
"""

from __future__ import annotations

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
from .logging import (
    InMemoryLogger,
    JSONLLogger,
    Logger,
    StdoutLogger,
    TensorBoardLogger,
    WandbLogger,
)
from .monitor import CumulantMonitor
from .observables import (
    alignment,
    autocorrelation,
    bbp_bitmask,
    build_intermediates,
    compute_all,
    contracted,
    derived_cumulants,
    factorization_residual,
    integrated_autocorr_time,
    leading_fisher_zero,
    leading_nongaussian_order,
    moments_from_cumulants,
    null_model,
    per_group,
    scalar_cumulants,
    spectral,
    total_scalar_cumulants,
)
from .probes import ProbeSet, ProbeSetCollection
from .types import Measurement, SharedIntermediates
from .window import BufferSource, DynamicalSource, GaussianSource, SlidingWindow

__version__ = "0.3.0"

__all__ = [
    "CumulantConfig",
    "LoggingConfig",
    "Measurement",
    "SharedIntermediates",
    "build_intermediates",
    "compute_all",
    "scalar_cumulants",
    "spectral",
    "contracted",
    "per_group",
    "derived_cumulants",
    "autocorrelation",
    "alignment",
    "null_model",
    "bbp_bitmask",
    "total_scalar_cumulants",
    "moments_from_cumulants",
    "factorization_residual",
    "leading_fisher_zero",
    "leading_nongaussian_order",
    "integrated_autocorr_time",
    "classical_fidelity",
    "trace_distance",
    "spectral_drift",
    "fisher_rao_distance",
    "quantum_fidelity",
    "quantum_trace_distance",
    "bures_distance",
    "bures_angle",
    "denoised_spectral_drift",
    "spectral_drift_snr",
    "signal_energy_drift",
    "cka_similarity",
    "cka_distance",
    "SlidingWindow",
    "BufferSource",
    "DynamicalSource",
    "GaussianSource",
    "ProbeSet",
    "ProbeSetCollection",
    "Logger",
    "InMemoryLogger",
    "JSONLLogger",
    "StdoutLogger",
    "WandbLogger",
    "TensorBoardLogger",
    "CumulantMonitor",
]
