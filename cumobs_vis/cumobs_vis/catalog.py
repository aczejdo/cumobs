"""The scalar-observable catalog.

The single source of truth for *how* each scalar ``cumobs`` key is presented:
its human label, which panel family it belongs to, a tooltip (the meaning, taken
from ``docs/concepts.md`` / ``docs/observables.md``), and small display hints
(invert-for-comparability, window-cap caveat, whether log-y is a sensible
default).

This is a plain, editable registry — add a row to :data:`SCALARS` to surface a
new scalar, or move it between families by changing its ``family``. Keys that
appear in the data but are *not* in the catalog still render (under the
``"other"`` family) so an unusually-rich config is never silently dropped; the
catalog only governs labelling and grouping.

Everything here is scalar-only by construction. Vector/matrix keys
(``covariance_spectrum``, ``*_matrix``, per-mode ribbons) are out of scope and
are handled — if at all — only as the summary scalars ``cumobs`` already
flattens (``<vector>/max`` etc.), which a user may add to the catalog manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- panel families, top to bottom (plan §3) -------------------------------
#
# The reference family is pinned separately (see referencePanel); the remaining
# families stack below it in this order, each with a light section separator.

FAMILY_ORDER: list[tuple[str, str]] = [
    ("detector_scale", "Detector & scale"),
    ("concentration", "Concentration"),
    ("nongaussianity", "Non-Gaussianity"),
    ("distances", "Distances"),
    ("dynamics", "Dynamics & early-warning"),
    ("misc", "Misc"),
    ("other", "Other scalars"),
]

FAMILY_TITLES: dict[str, str] = dict(FAMILY_ORDER)


@dataclass(frozen=True)
class ScalarInfo:
    """Presentation metadata for one scalar key."""

    key: str
    label: str
    family: str
    tooltip: str
    invert: bool = False          # plot ``1 - value`` for comparability (fidelities)
    log_default: bool = False     # a sensible default for the per-panel log-y toggle
    window_capped: bool = False   # append the "capped by the window (rank <= m-1)" note
    reference_ok: bool = False    # usable as the top reference fallback series

    @property
    def display_label(self) -> str:
        return f"1 - {self.label}" if self.invert else self.label


# The window-cap caveat, appended to the tooltip of window-limited scalars
# (plan §4e): PR / effective_rank / etc. are bounded by rank <= m-1, so their
# level partly reflects the window length, not just the model.
WINDOW_CAP_NOTE = (
    " Note: window-limited — the covariance rank is <= m-1 (m = window_size) "
    "regardless of probe count, so this level partly reflects the window length, "
    "not just the model."
)


def _rows() -> list[ScalarInfo]:
    return [
        # --- reference / first order --------------------------------------
        ScalarInfo(
            "total_first_cumulant", "total_first_cumulant", "misc",
            "Sum of the per-probe mean loss (E_L.sum()) — the baseline loss level. "
            "Used as the reference-panel fallback when no metrics file is supplied.",
            reference_ok=True,
        ),
        # --- detector & scale ---------------------------------------------
        ScalarInfo(
            "spectral_heat_capacity", "spectral_heat_capacity (SHC)", "detector_scale",
            "Variance of the covariance spectrum — the headline transition "
            "diagnostic; spikes when one or a few modes detach from the bulk. "
            "Scales as D^-2, so ddof AND detrend_order must match across compared "
            "runs.",
            log_default=True,
        ),
        ScalarInfo(
            "total_second_cumulant", "total_second_cumulant", "detector_scale",
            "kappa_2 of the aggregate S = sum of the covariance entries "
            "(1^T C 1) — the energy-fluctuation / total loss variance.",
            log_default=True,
        ),
        ScalarInfo(
            "spectral_gap", "spectral_gap", "detector_scale",
            "lambda_1 - lambda_2 — the separation of the leading covariance mode "
            "from the rest; a condensation order parameter.",
            log_default=True,
        ),
        ScalarInfo(
            "normalized_shc", "normalized_shc", "detector_scale",
            "Var(lambda)/mean(lambda)^2 = r/PR - 1 over the nonzero spectrum — a "
            "scale-free condensation index (a transform of PR/rank).",
        ),
        ScalarInfo(
            "log_det", "log_det", "detector_scale",
            "Sum of log(lambda_i) over the nonzero eigenvalues — the "
            "log-determinant (a volume/uncertainty scalar). NaN when rank is 0.",
        ),
        ScalarInfo(
            "edge_gap", "edge_gap", "detector_scale",
            "lambda_1 - mp_edge_upper — the re-grounded condensation gap versus "
            "the Marchenko-Pastur noise edge (preferred over spectral_gap once the "
            "MP null is on).",
        ),
        ScalarInfo(
            "spike_energy_fraction", "spike_energy_fraction", "detector_scale",
            "Fraction of total variance carried by modes above the MP noise edge.",
        ),
        ScalarInfo(
            "n_spikes", "n_spikes", "detector_scale",
            "Count of detached modes above the Tracy-Widom-gated MP edge.",
        ),
        # --- concentration -------------------------------------------------
        ScalarInfo(
            "participation_ratio", "participation_ratio (PR)", "concentration",
            "(sum lambda)^2 / sum lambda^2 in [1, r] — the effective number of "
            "active modes. ~1: one rank-1 mode dominates; ~r: isotropic. NaN for a "
            "dead window.",
            window_capped=True,
        ),
        ScalarInfo(
            "effective_rank", "effective_rank", "concentration",
            "exp(von_neumann_entropy) in [1, r] — a smooth mode count. Equals "
            "exp(von_neumann_entropy) EXACTLY (not independent state).",
            window_capped=True,
        ),
        ScalarInfo(
            "von_neumann_entropy", "von_neumann_entropy", "concentration",
            "-sum p_i log p_i for p = lambda/sum lambda — the spectral (von "
            "Neumann) entropy. effective_rank is its exponential.",
            window_capped=True,
        ),
        ScalarInfo(
            "top_eigenvalue_fraction", "top_eigenvalue_fraction", "concentration",
            "lambda_1 / sum lambda in [0, 1] — the variance fraction carried by "
            "the leading mode (-> 1 as one mode dominates). A condensation "
            "complement to SHC.",
        ),
        # --- non-Gaussianity ----------------------------------------------
        ScalarInfo(
            "total_fourth_cumulant", "total_fourth_cumulant", "nongaussianity",
            "Connected kappa_4 of the aggregate S. ~0 for a Gaussian window with "
            "the unbiased estimator; grows with heavy tails / non-Gaussianity.",
        ),
        ScalarInfo(
            "total_third_cumulant", "total_third_cumulant", "nongaussianity",
            "kappa_3 of the aggregate S — asymmetry of the joint loss "
            "fluctuations (a raw cumulant, not standardized skewness).",
        ),
        ScalarInfo(
            "total_cumulant_5", "total_cumulant_5", "nongaussianity",
            "kappa_5 of the aggregate S (plug-in; diagnostic). Numerically "
            "delicate — keep dtype=float64.",
        ),
        ScalarInfo(
            "total_cumulant_6", "total_cumulant_6", "nongaussianity",
            "kappa_6 of the aggregate S (plug-in; diagnostic).",
        ),
        # --- distances -----------------------------------------------------
        ScalarInfo(
            "trace_distance", "trace_distance", "distances",
            "1/2 sum |p_i - q_i| (total variation) on normalized spectra vs the "
            "reference measurement. Eigenvector-blind and scale-blind. 0 iff "
            "equal.",
        ),
        ScalarInfo(
            "classical_fidelity", "classical_fidelity", "distances",
            "(sum sqrt(p_i q_i))^2 Bhattacharyya overlap on normalized spectra. "
            "1 iff equal — plotted as 1 - fidelity for comparability.",
            invert=True,
        ),
        ScalarInfo(
            "spectral_drift", "spectral_drift", "distances",
            "||lambda - mu||_2 / max(||lambda||, ||mu||) on RAW spectra — the "
            "scale-sensitive companion to the classical distances (responds to a "
            "pure variance change).",
        ),
        ScalarInfo(
            "fisher_rao_distance", "fisher_rao_distance", "distances",
            "2 arccos(sqrt(classical_fidelity)) — the geodesic (proper-metric) "
            "form of the classical fidelity.",
        ),
        ScalarInfo(
            "cka_distance", "cka_distance", "distances",
            "1 - linear CKA on the top-k eigen-factors — eigenvector-sensitive "
            "representation drift between measurements (cheap: O(n k^2)).",
        ),
        ScalarInfo(
            "cka_similarity", "cka_similarity", "distances",
            "Linear CKA (Frobenius cosine) on the top-k eigen-factors. 1 iff "
            "proportional — plotted as 1 - similarity for comparability.",
            invert=True,
        ),
        ScalarInfo(
            "quantum_trace_distance", "quantum_trace_distance", "distances",
            "1/2 ||rho - sigma||_1 on rho = C/tr C — eigenvector-sensitive "
            "(O(n^3); needs materialize_matrices).",
        ),
        ScalarInfo(
            "quantum_fidelity", "quantum_fidelity", "distances",
            "Uhlmann fidelity on rho = C/tr C — plotted as 1 - fidelity for "
            "comparability. Needs materialize_matrices.",
            invert=True,
        ),
        ScalarInfo(
            "bures_distance", "bures_distance", "distances",
            "sqrt(2(1 - sqrt(quantum_fidelity))) — the Bures geodesic form of the "
            "quantum fidelity.",
        ),
        ScalarInfo(
            "bures_angle", "bures_angle", "distances",
            "arccos(sqrt(quantum_fidelity)) — the operator analog of the "
            "Fisher-Rao distance.",
        ),
        ScalarInfo(
            "denoised_spectral_drift", "denoised_spectral_drift", "distances",
            "spectral_drift with the surrogate-null floor subtracted; 0 means "
            "'indistinguishable from noise' (needs null_model surrogate/both).",
        ),
        ScalarInfo(
            "signal_energy_drift", "signal_energy_drift", "distances",
            "spectral_drift computed only on the above-MP-edge spectrum (needs "
            "null_model mp/both).",
        ),
        # --- dynamics / early-warning -------------------------------------
        ScalarInfo(
            "autocorr_lag1", "autocorr_lag1", "dynamics",
            "Lag-1 autocorrelation of the aggregate S — RISING is the critical "
            "slowing-down early warning (the autocorrelation half of the "
            "early-warning story; SHC is the variance half).",
        ),
        ScalarInfo(
            "autocorr_time", "autocorr_time", "dynamics",
            "Integrated autocorrelation time tau_int of S (Sokal auto-window) — "
            "the honest sample size m_eff scales with 1/tau_int.",
        ),
        ScalarInfo(
            "fisher_zero_gap", "fisher_zero_gap", "dynamics",
            "Distance of the leading Fisher/Lee-Yang zero to the positive real "
            "axis (-> 0 near a transition). Mostly tracks 1/sqrt(kappa_2).",
        ),
        ScalarInfo(
            "leading_fisher_zero_arg", "leading_fisher_zero_arg", "dynamics",
            "Phase (arg) of the leading Fisher/Lee-Yang zero — the genuinely new "
            "content, switched on by kappa_3 (asymmetry tilts the zero off the "
            "imaginary axis).",
        ),
        ScalarInfo(
            "leading_fisher_zero_abs", "leading_fisher_zero_abs", "dynamics",
            "Magnitude of the leading Fisher/Lee-Yang zero.",
        ),
        ScalarInfo(
            "align_cov_skew", "align_cov_skew", "dynamics",
            "cos(C, M3) in [-1, 1] — does the skew structure co-locate with the "
            "variance condensation?",
        ),
        ScalarInfo(
            "align_cov_kurtosis", "align_cov_kurtosis", "dynamics",
            "cos(C, K) in [-1, 1] — does the connected-kurtosis structure "
            "co-locate with the variance condensation?",
        ),
        ScalarInfo(
            "crossing_detected", "crossing_detected", "dynamics",
            "1.0 if eigenvector tracking permuted the descending-eigenvalue order "
            "this step (a mode crossing).",
        ),
        # --- misc ----------------------------------------------------------
        ScalarInfo(
            "excess_loss", "excess_loss", "misc",
            "sum_i (E_L[i] - reference_loss[i]) — the summed probe loss in excess "
            "of a converged-model reference (a loss-gap diagnostic).",
        ),
    ]


SCALARS: dict[str, ScalarInfo] = {r.key: r for r in _rows()}


# --- per-group sub-series mapping (plan §1, §4b) ---------------------------
#
# ``group_*/<label>`` keys are surfaced as toggle-able sub-series nested under
# their parent run, drawn on the panel of the base observable they decompose.
GROUP_SERIES: dict[str, str] = {
    "group_shc": "spectral_heat_capacity",
    "group_pr": "participation_ratio",
    "group_total_variance": "total_second_cumulant",
}


def info_for(key: str) -> ScalarInfo:
    """Return the :class:`ScalarInfo` for ``key``, synthesising a generic entry
    (family ``"other"``) for keys not in the curated catalog so nothing logged
    is silently dropped."""
    hit = SCALARS.get(key)
    if hit is not None:
        return hit
    return ScalarInfo(key, key, "other", f"Scalar key '{key}' (not in the curated catalog).")


def tooltip_for(key: str) -> str:
    """Tooltip text for ``key``, with the window-cap note appended when relevant."""
    info = info_for(key)
    text = info.tooltip
    if info.window_capped:
        text += WINDOW_CAP_NOTE
    return text


def family_of(key: str) -> str:
    return info_for(key).family
