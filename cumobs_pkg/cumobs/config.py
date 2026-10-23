"""Configuration dataclasses for ``cumobs``.

A single :class:`CumulantConfig` carries every knob the observable core,
the monitor, and the logging policy need.  See the package spec (§5) for the
authoritative description of each field; the most important invariants are:

* ``D = window_size - ddof`` is the *one* normalization divisor used in every
  moment and (in ``cumulant_estimator="biased"`` mode) every cumulant
  subtraction (decision D1).  PR / fidelity / trace-distance are ``D``-invariant.
* ``float64`` is the default dtype because the asymmetric ``eig`` on the
  contracted-cumulant Gram partners (§3.4-3.5) is the numerically sensitive
  part of the pipeline.

Three defaults changed in v0.2 to fix statistical / interpretation issues found
in the audit (see ``docs/configuration.md`` for the rationale):

* ``cumulant_estimator="unbiased"`` — the named 3rd/4th cumulants now use Fisher
  k-statistics, so a Gaussian window reads ≈ 0 (the biased plug-in had an
  ``O(1/m)`` baseline of order ``-c·κ₂²/m`` that co-varied with the variance).
* ``shc_mode="rank"`` — spectral heat capacity is the variance over the
  **nonzero** spectrum, so it no longer shrinks as you add dead/extra probes.
* ``detrend_order=0`` — opt-in within-window polynomial detrending (set ``1`` to
  strip a linear trajectory drift that would otherwise masquerade as a mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CumulantConfig:
    """Knobs for the cumulant-observable pipeline.

    Attributes
    ----------
    window_size:
        Number of rows ``m`` in the sliding buffer ``B``.
    eval_interval:
        Steps between probe evaluations / window appends.
    update_interval:
        Steps between observable computations.  Keep ``< window_size`` so the
        window meaningfully refreshes between measurements.
    ddof:
        Delta-degrees-of-freedom for the single divisor ``D = window_size - ddof``.
    cumulant_estimator:
        ``"unbiased"`` (default) computes the named 3rd/4th cumulants
        (``total_*``/``diagonal_*``/``principal_*``) with Fisher **k-statistics**,
        the unique unbiased estimators — a Gaussian window then reads ≈ 0.
        ``"biased"`` uses the plug-in central-moment formulas with the single
        divisor ``D`` (lower variance, but an ``O(1/m)`` Gaussian baseline that
        scales with ``κ₂²``).  Orders ≥ 5 (``total_cumulant_p``) are plug-in in
        **both** modes (no closed-form k-statistic ships; mirrors ``scipy``,
        which caps ``kstat`` at order 4).
    order:
        Max cumulant order to compute (2, 3, or 4).
    max_scalar_cumulant_order:
        If ``> 4`` (and ``order == 4``), also emit the general scalar cumulants
        ``total_cumulant_{5..max}`` of the aggregate ``S`` via the moment-cumulant
        recurrence (§3.8).  Orders 2-4 are already the named totals
        (``total_second_cumulant`` / ``total_third_cumulant`` /
        ``total_fourth_cumulant``), so only orders ``>= 5`` add new keys.  Cheap
        (``O(m * max)``) and intended as a mathematical check / heavy-tail probe;
        ``float64`` recommended.
    detrend_order:
        Degree of the within-window least-squares polynomial (in the row/time
        index) removed before computing the centered buffer ``B_c``.  ``0``
        (default) is plain per-probe mean-centering — the original behaviour.
        ``1`` removes a linear trajectory drift (recommended for the *dynamical*
        2RDM, whose rows are consecutive training steps: a smooth drift otherwise
        shows up as a spurious dominant covariance mode).  ``E_L`` (the reported
        first cumulant) is always the plain mean.  Detrending of degree ``d``
        removes ``d+1`` degrees of freedom, so for unbiased second-order scale
        you may want ``ddof = d + 1``.
    top_k:
        Number of principal directions for projection cumulants / IPR / CKA.
    eig_threshold:
        Relative cutoff for a "nonzero" eigenvalue, multiplied by the largest
        eigenvalue of the Gram matrix.  Also defines the ``rank`` over which
        ``shc_mode="rank"`` takes the spectral variance.
    eps:
        Denominator / sqrt guard.
    shc_mode:
        ``"rank"`` (default) computes ``spectral_heat_capacity`` as the population
        variance over the ``rank`` **nonzero** eigenvalues (invariant to the
        number of probes / structural zeros).  ``"full"`` is the legacy variance
        over the full zero-padded length-``n`` spectrum (``∝ 1/n``).
    materialize_matrices:
        Build the opt-in ``n x n`` matrices ``C``, ``M3``, ``K``.
    compute_eigenvectors:
        Build the sample-space eigenvectors ``V`` (needed for IPR, block energy,
        eigenvector export, the quantum distances and the CKA distance).
    per_group:
        Compute the §3.7 per-group observables (needs probe group labels).
    distance_reference:
        ``"previous"`` | ``"baseline"`` | ``"lag:k"`` (decision D4).
    distance_source:
        Which spectrum feeds the distances; ``"covariance"`` by default, also
        accepts ``"contracted_third_cumulant"`` / ``"contracted_fourth_cumulant"``
        (decision D3).  Note the contracted spectra are **signed**; the classical
        distances are only strictly meaningful for the PSD covariance source.
    distance_flavor:
        ``"classical"`` (spectrum only), ``"quantum"`` (density matrix), or
        ``"both"`` (decision D2).  Quantum requires ``materialize_matrices``.
    compute_cka_distance:
        Also emit ``cka_distance`` / ``cka_similarity`` between successive
        measurements — a linear-CKA representation-similarity on the top-``k``
        eigen-factors (eigenvector-sensitive, ``O(n·k²)``, cheaper than the
        ``O(n³)`` quantum distances).  Requires ``compute_eigenvectors``.
    error_bars:
        ``"off"`` (default) or ``"jackknife"``.  Jackknife re-runs the scalar
        observables leave-one-row-out (``m`` extra passes — expensive) and emits a
        ``{scalar}/se`` standard-error estimate for every scalar observable.
    dtype:
        Working dtype; ``float64`` recommended for the asymmetric eigs.
    sign_gauge:
        Deterministic eigenvector sign convention (decision D6): flip each
        eigenvector column so its largest-magnitude entry is positive.
    imag_ratio_warn:
        Threshold on ``max|Im|/(max|Re|+eps)`` above which a numerical-health
        warning is emitted for the asymmetric contracted spectra.
    root_order:
        Truncation order of the partition function ``Z_P(t)`` whose leading
        complex root is the Fisher / Lee-Yang zero (§1c).  ``None`` (default)
        uses ``min(4, P_tot)`` where ``P_tot`` is the highest available aggregate
        cumulant order (``max(order, max_scalar_cumulant_order)``); an explicit
        value is clamped to ``[2, P_tot]`` (warns if it exceeds ``P_tot``).
    principal_cumulant_order:
        Highest per-mode cumulant order for ``principal_standardized_cumulant`` /
        ``principal_cumulant_{p}`` (§1).  ``4`` (default) adds no state — the
        per-mode spectrum stops at kurtosis.  ``> 4`` computes per-mode cumulants
        of the principal projections ``P_a`` up to this order (plug-in recurrence,
        the *one* genuinely new primitive in §1); ``float64`` recommended.
    null_model:
        Background null for the spectral observables (§3).  ``"off"`` (default),
        ``"surrogate"`` (block-bootstrap circular column shifts → ``<scalar>/
        null_mean|null_sd|null_z`` and the ``spectral_drift`` null floor),
        ``"mp"`` (Marchenko-Pastur / Tracy-Widom edge → ``mp_edge_upper`` /
        ``n_spikes`` / ``spike_energy_fraction`` / ``mp_bulk_ks``), or ``"both"``.
        The MP edge also enables the §6 BBP bitmask and the §4 ``signal_energy_drift``.
    null_samples:
        Number of surrogates per measurement for ``null_model`` ∈ {surrogate,
        both} — the bootstrap re-runs the (stripped) scalar pipeline this many
        times, so it is roughly ``null_samples``× the base cost.
    rho_keep:
        Energy target for the §6 BBP support set: ``S_a`` is the smallest index
        set with ``Σ_{i∈S_a} v_a[i]² ≥ rho_keep`` (default ``0.9``).
    track_modes:
        Track eigenvector identity across measurements (§7 step 2): Hungarian
        matching on ``|⟨v_a(t), v_b(t+1)⟩|`` → ``mode_permutation`` /
        ``crossing_detected`` and the per-mode CSD series
        ``principal_autocorr_lag1`` / ``principal_autocorr_time``.  Requires
        ``compute_eigenvectors``; lives in the monitor (needs cross-measurement
        history).
    """

    window_size: int = 20
    eval_interval: int = 10
    update_interval: int = 10
    ddof: int = 1
    cumulant_estimator: str = "unbiased"
    order: int = 4
    max_scalar_cumulant_order: int = 4
    detrend_order: int = 0
    top_k: int = 3
    eig_threshold: float = 1e-7
    eps: float = 1e-13
    shc_mode: str = "rank"
    materialize_matrices: bool = False
    compute_eigenvectors: bool = False
    per_group: bool = False
    distance_reference: str = "previous"
    distance_source: str = "covariance"
    distance_flavor: str = "classical"
    compute_cka_distance: bool = False
    error_bars: str = "off"
    dtype: torch.dtype = torch.float64
    sign_gauge: bool = True
    imag_ratio_warn: float = 1e-3
    # --- v0.3 extension knobs (cumobs_extension_spec) -----------------------
    root_order: int | None = None
    principal_cumulant_order: int = 4
    null_model: str = "off"
    null_samples: int = 50
    rho_keep: float = 0.9
    track_modes: bool = False

    def __post_init__(self) -> None:
        if self.order not in (2, 3, 4):
            raise ValueError(f"order must be 2, 3, or 4; got {self.order}")
        if self.cumulant_estimator not in ("biased", "unbiased"):
            raise ValueError(
                "cumulant_estimator must be 'biased' or 'unbiased'; "
                f"got {self.cumulant_estimator!r}"
            )
        if self.max_scalar_cumulant_order < 2:
            raise ValueError(
                "max_scalar_cumulant_order must be >= 2; got "
                f"{self.max_scalar_cumulant_order}"
            )
        if self.detrend_order < 0:
            raise ValueError(
                f"detrend_order must be >= 0; got {self.detrend_order}"
            )
        if self.detrend_order + 1 >= self.window_size:
            raise ValueError(
                f"detrend_order ({self.detrend_order}) leaves too few rows: "
                f"need window_size > detrend_order + 1 (window_size={self.window_size})"
            )
        if self.window_size - (self.ddof + self.detrend_order) <= 0:
            raise ValueError(
                "D = window_size - (ddof + detrend_order) must be > 0 "
                f"(window_size={self.window_size}, ddof={self.ddof}, "
                f"detrend_order={self.detrend_order})"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {self.top_k}")
        if self.shc_mode not in ("rank", "full"):
            raise ValueError(
                f"shc_mode must be 'rank' or 'full'; got {self.shc_mode!r}"
            )
        if self.error_bars not in ("off", "jackknife"):
            raise ValueError(
                f"error_bars must be 'off' or 'jackknife'; got {self.error_bars!r}"
            )
        if self.compute_cka_distance and not self.compute_eigenvectors:
            raise ValueError(
                "compute_cka_distance=True requires compute_eigenvectors=True "
                "(the CKA distance is built from the top-k eigen-factors)."
            )
        if self.distance_flavor not in ("classical", "quantum", "both"):
            raise ValueError(
                f"distance_flavor must be 'classical', 'quantum', or 'both'; "
                f"got {self.distance_flavor!r}"
            )
        if self.distance_source not in (
            "covariance",
            "contracted_third_cumulant",
            "contracted_fourth_cumulant",
        ):
            raise ValueError(
                "distance_source must be 'covariance', "
                "'contracted_third_cumulant', or 'contracted_fourth_cumulant'; "
                f"got {self.distance_source!r}"
            )
        if not (
            self.distance_reference in ("previous", "baseline")
            or self.distance_reference.startswith("lag:")
        ):
            raise ValueError(
                "distance_reference must be 'previous', 'baseline', or 'lag:k'; "
                f"got {self.distance_reference!r}"
            )
        if self.null_model not in ("off", "surrogate", "mp", "both"):
            raise ValueError(
                "null_model must be 'off', 'surrogate', 'mp', or 'both'; "
                f"got {self.null_model!r}"
            )
        if self.null_samples < 1:
            raise ValueError(f"null_samples must be >= 1; got {self.null_samples}")
        if not 0.0 < self.rho_keep <= 1.0:
            raise ValueError(
                f"rho_keep must be in (0, 1]; got {self.rho_keep}"
            )
        if self.principal_cumulant_order < 4:
            raise ValueError(
                "principal_cumulant_order must be >= 4; got "
                f"{self.principal_cumulant_order}"
            )
        if self.root_order is not None and self.root_order < 2:
            raise ValueError(
                f"root_order must be None or >= 2; got {self.root_order}"
            )
        if self.track_modes and not self.compute_eigenvectors:
            raise ValueError(
                "track_modes=True requires compute_eigenvectors=True "
                "(mode tracking matches sample-space eigenvectors across steps)."
            )

    @property
    def divisor(self) -> int:
        """The single normalization divisor ``D = window_size - (ddof + detrend_order)``.

        Decision D1 / correctness fix R1: detrending of degree ``d`` is a
        rank-``(d+1)`` projection, so the unbiased second-order divisor is
        ``m - (ddof + detrend_order)`` (it reduces to ``m - ddof`` at
        ``detrend_order = 0``).
        """
        return self.window_size - (self.ddof + self.detrend_order)

    def lag(self) -> int:
        """Parse the ``k`` in a ``"lag:k"`` reference policy (default 1)."""
        if self.distance_reference.startswith("lag:"):
            return int(self.distance_reference.split(":", 1)[1])
        return 1


# Keys the §8 consolidation drops from the *default* emitted set when
# ``LoggingConfig.trim_consolidated`` is on.  Everything is still computed and
# present on the ``Measurement``; this only suppresses it at flatten time, and
# only on request (the default keeps full backward-compatible output).
CONSOLIDATED_DROP_KEYS: tuple[str, ...] = (
    "von_neumann_entropy",   # == log(effective_rank); keep effective_rank
    "classical_fidelity",    # recoverable from fisher_rao_distance
    "quantum_fidelity",      # recoverable from bures_distance / bures_angle
)
# Prefix-matched drops (the general higher-order scalar cumulants).
CONSOLIDATED_DROP_PREFIXES: tuple[str, ...] = (
    "total_cumulant_",       # total_cumulant_5, _6, … — on-demand diagnostic
)


@dataclass
class LoggingConfig:
    """Policy for flattening a :class:`~cumobs.types.Measurement` for logging.

    ``vector_policy`` is the global default; ``vector_overrides`` lets a single
    key opt into a different policy (e.g. always-full for ``covariance_spectrum``).

    ``trim_consolidated`` opts into the §8 default-set consolidation: when ``True``
    the deprecated-default keys (:data:`CONSOLIDATED_DROP_KEYS` /
    :data:`CONSOLIDATED_DROP_PREFIXES`) are suppressed from the flattened output
    in favour of their kept companions (``effective_rank``, the geodesic
    distances, the on-demand cumulant routine).  Default ``False`` keeps the full
    backward-compatible output.  ``drop_keys`` adds further user-specified keys.
    """

    vector_policy: str = "summary"  # "summary" | "full" | "off"
    vector_overrides: dict[str, str] = field(default_factory=dict)
    log_health: bool = True
    trim_consolidated: bool = False
    drop_keys: set[str] = field(default_factory=set)

    def policy_for(self, key: str) -> str:
        return self.vector_overrides.get(key, self.vector_policy)

    def dropped(self) -> tuple[set[str], tuple[str, ...]]:
        """Resolve ``(drop_keys, drop_prefixes)`` for :meth:`Measurement.to_flat_dict`."""
        keys = set(self.drop_keys)
        prefixes: tuple[str, ...] = ()
        if self.trim_consolidated:
            keys |= set(CONSOLIDATED_DROP_KEYS)
            prefixes = CONSOLIDATED_DROP_PREFIXES
        return keys, prefixes
