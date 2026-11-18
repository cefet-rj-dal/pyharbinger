"""Harbinger detection utilities.

Python port of the R ``harbinger_utils*`` helpers: deviation measures, filter
criteria, and candidate selection. These are the building blocks that detectors
compose through the base :class:`~pyharbinger.harbinger.Harbinger` hooks
(``har_distance`` / ``har_outliers`` / ``har_outliers_check``).

Pipeline (as in R):

#. a *deviation measure* turns raw residuals/values into a non-negative score
   per observation (:func:`har_deviation_l2`, ...);
#. a *filter criterion* selects candidate outlier positions from the score and
   reports the lower/upper thresholds (:func:`har_filter_gaussian`, ...), as a
   :class:`FilterResult`;
#. a *candidate selection* rule collapses consecutive candidate runs into the
   final boolean anomaly mask (:func:`har_candidate_selection_firstgroup`, ...),
   as a :class:`SelectionResult`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "FilterResult",
    "SelectionResult",
    "har_candidate_selection_firstgroup",
    "har_candidate_selection_highgroup",
    "har_candidate_selection_referencedistribution",
    "har_deviation_huber",
    "har_deviation_l1",
    "har_deviation_l2",
    "har_filter_boxplot",
    "har_filter_gaussian",
    "har_filter_grubbs",
    "har_filter_mad",
    "har_filter_none",
    "har_filter_ratio",
]


@dataclass
class FilterResult:
    """Candidate outlier positions and the (inf, sup) threshold that produced them.

    ``index`` holds 0-based integer positions (the R ``which(...)`` result).
    """

    index: np.ndarray
    threshold: tuple[float, float]


@dataclass
class SelectionResult:
    """Final boolean anomaly mask and the threshold carried over from filtering."""

    mask: np.ndarray
    threshold: tuple[float, float]


def _rowwise(values: np.ndarray) -> np.ndarray:
    """Collapse a 2-D deviation array to per-row sums (R ``rowSums``)."""
    if values.ndim == 2:
        return values.sum(axis=1)
    return values


# --- deviation measures (harbinger_utils_deviation_measures.R) --------------


def har_deviation_l1(values) -> np.ndarray:
    """Absolute deviation (L1); row-sums for matrices."""
    return _rowwise(np.abs(np.asarray(values, dtype=float)))


def har_deviation_l2(values) -> np.ndarray:
    """Squared deviation (L2); row-sums for matrices."""
    return _rowwise(np.asarray(values, dtype=float) ** 2)


def _huber_rho(values: np.ndarray, k: float) -> np.ndarray:
    abs_values = np.abs(values)
    return np.where(abs_values <= k, 0.5 * values**2, k * (abs_values - 0.5 * k))


def har_deviation_huber(values, k: float = 1.345) -> np.ndarray:
    """Huber loss deviation; row-sums for matrices."""
    return _rowwise(_huber_rho(np.asarray(values, dtype=float), k))


# --- filter criteria (harbinger_utils_filter_criteria.R) --------------------


def har_filter_none(res) -> FilterResult:
    """No filtering: select nothing, with an unbounded threshold."""
    return FilterResult(np.array([], dtype=int), (-np.inf, np.inf))


def har_filter_boxplot(res) -> FilterResult:
    """Tukey boxplot rule: flag values beyond 1.5*IQR of the quartiles."""
    res = np.asarray(res, dtype=float)
    q1, q3 = np.nanquantile(res, [0.25, 0.75])
    iqr = q3 - q1
    inf = float(q1 - 1.5 * iqr)
    sup = float(q3 + 1.5 * iqr)
    index = np.where((res > sup) | (res < inf))[0]
    return FilterResult(index, (inf, sup))


def har_filter_gaussian(res) -> FilterResult:
    """Three-sigma rule: flag values beyond mean +/- 3*sd."""
    res = np.asarray(res, dtype=float)
    sd = np.std(res, ddof=1)
    mean = np.mean(res)
    sup = float(mean + 3 * sd)
    inf = float(mean - 3 * sd)
    index = np.where((res > sup) | (res < inf))[0]
    return FilterResult(index, (inf, sup))


def har_filter_mad(res, k: float = 3, constant: float = 1.4826) -> FilterResult:
    """Median absolute deviation rule: flag values beyond median +/- k*MAD."""
    res = np.asarray(res, dtype=float)
    center = np.median(res)
    scale = np.median(np.abs(res - center)) * constant

    if not np.isfinite(center) or not np.isfinite(scale):
        return FilterResult(np.array([], dtype=int), (np.nan, np.nan))

    if scale <= 0:
        index = np.where(np.isfinite(res) & (res != center))[0]
        return FilterResult(index, (float(center), float(center)))

    inf = float(center - k * scale)
    sup = float(center + k * scale)
    index = np.where((res > sup) | (res < inf))[0]
    return FilterResult(index, (inf, sup))


def har_filter_grubbs(res, alpha: float = 0.05, two_sided: bool = True) -> FilterResult:
    """Iterative Grubbs test: repeatedly remove the most extreme significant value."""
    values = np.asarray(res, dtype=float)
    valid = np.where(np.isfinite(values))[0]
    threshold = [np.nan, np.nan]

    if len(valid) < 3:
        return FilterResult(np.array([], dtype=int), (np.nan, np.nan))

    selected: list[int] = []
    remaining = list(valid)
    while True:
        current = values[remaining]
        n = len(current)
        if n < 3:
            break

        mu = np.mean(current)
        sigma = np.std(current, ddof=1)
        if not np.isfinite(sigma) or sigma <= 0:
            break

        deviation = np.abs(current - mu)
        pos = int(np.argmax(deviation))
        g_stat = deviation[pos] / sigma

        alpha_adj = alpha / (2 * n) if two_sided else alpha / n
        t_crit = stats.t.ppf(1 - alpha_adj, df=n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

        if not np.isfinite(g_stat) or not np.isfinite(g_crit) or g_stat <= g_crit:
            break

        idx = remaining[pos]
        selected.append(idx)

        candidate = values[idx]
        # Keep the least extreme detected value on each side as the summary cutoff.
        if candidate >= mu:
            threshold[1] = candidate if np.isnan(threshold[1]) else min(threshold[1], candidate)
        else:
            threshold[0] = candidate if np.isnan(threshold[0]) else max(threshold[0], candidate)

        remaining.pop(pos)

    selected.sort()
    return FilterResult(np.array(selected, dtype=int), (threshold[0], threshold[1]))


def har_filter_ratio(res) -> FilterResult:
    """Ratio-to-maximum rule: three-sigma on the normalized (1 - res/max) score."""
    res = np.asarray(res, dtype=float)
    if res.size == 0:
        return FilterResult(np.array([], dtype=int), (np.nan, np.nan))

    max_res = np.nanmax(res)
    if not np.isfinite(max_res) or max_res <= 0:
        return FilterResult(np.array([], dtype=int), (np.nan, np.nan))

    ratio = 1 - res / max_res
    sd = np.std(ratio, ddof=1)
    mean = np.mean(ratio)
    sup = float((mean + 3 * sd - 1) * max_res)
    inf = float((mean - 3 * sd - 1) * max_res)
    index = np.where((res > sup) | (res < inf))[0]
    return FilterResult(index, (inf, sup))


# --- candidate selection (harbinger_utils_candidate_selection.R) ------------


def _split_consecutive(idx: np.ndarray) -> list[np.ndarray]:
    """Split sorted integer positions into runs of consecutive values."""
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return np.split(idx, breaks)


def _as_index(outliers) -> tuple[np.ndarray, tuple[float, float]]:
    """Accept a :class:`FilterResult` (or a bare index array) uniformly."""
    if isinstance(outliers, FilterResult):
        return np.asarray(outliers.index, dtype=int), outliers.threshold
    return np.asarray(outliers, dtype=int), (np.nan, np.nan)


def _abs_residuals(residuals) -> np.ndarray:
    residuals = np.abs(np.asarray(residuals, dtype=float))
    return _rowwise(residuals)


def har_candidate_selection_firstgroup(outliers, residuals, values) -> SelectionResult:
    """Keep the first position of each consecutive run of candidates."""
    idx, threshold = _as_index(outliers)
    residuals = _abs_residuals(residuals)
    size = len(residuals)
    mask = np.zeros(size, dtype=bool)
    for group in _split_consecutive(idx):
        mask[int(group.min())] = True
    return SelectionResult(mask, threshold)


def har_candidate_selection_highgroup(outliers, residuals, values) -> SelectionResult:
    """Keep the highest-residual position of each consecutive run (first on ties)."""
    idx, threshold = _as_index(outliers)
    residuals = _abs_residuals(residuals)
    size = len(residuals)
    mask = np.zeros(size, dtype=bool)
    for group in _split_consecutive(idx):
        max_val = residuals[group].max()
        candidates = group[residuals[group] == max_val]
        mask[int(candidates.min())] = True
    return SelectionResult(mask, threshold)


def har_candidate_selection_referencedistribution(
    outliers,
    residuals,
    values,
    history_size: int = 30,
    sigma_level: float = 3,
) -> SelectionResult:
    """Confirm candidates against a Gaussian reference of the preceding window."""
    idx, threshold = _as_index(outliers)
    residuals = _abs_residuals(residuals)
    values = _rowwise(np.asarray(values, dtype=float))
    size = len(residuals)
    mask = np.zeros(size, dtype=bool)

    for group in _split_consecutive(idx):
        start_idx = int(group.min())
        ref_start = max(0, start_idx - history_size)
        ref_values = values[ref_start:start_idx]
        ref_values = ref_values[np.isfinite(ref_values)]

        if len(ref_values) < history_size:
            mask[start_idx] = True
            continue

        mu = np.mean(ref_values)
        sigma = np.std(ref_values, ddof=1)

        for candidate_idx in group:
            candidate_value = values[candidate_idx]
            if not np.isfinite(candidate_value):
                continue
            if not np.isfinite(mu):
                is_outlier = False
            elif not np.isfinite(sigma) or sigma <= 0:
                is_outlier = candidate_value != mu
            else:
                is_outlier = abs(candidate_value - mu) > sigma_level * sigma
            if is_outlier:
                mask[int(candidate_idx)] = True

    return SelectionResult(mask, threshold)
