"""Change-point detectors (Python port of the R ``hcp_*`` family).

Two groups are provided:

* **Offline, cost-based** detectors that segment the whole series by changes in
  mean/variance. The R package wraps ``changepoint::cpt.meanvar`` (Gaussian cost,
  MBIC penalty); here they wrap the ``ruptures`` library with its Gaussian
  (``"normal"``) cost. Because ``ruptures`` does not implement the MBIC penalty,
  parity with R is **approximate** -- compare breakpoint positions, not exact
  equality.

  * :class:`HcpAmoc` -- at most one change (``ruptures.Dynp``, ``n_bkps=1``).
  * :class:`HcpBinseg` -- binary segmentation (``ruptures.Binseg``, ``n_bkps=Q``).
  * :class:`HcpPelt` -- PELT (``ruptures.Pelt`` with a penalty).

* **Online, streaming** detectors that scan the series once:

  * :class:`HcpKswin` -- Kolmogorov-Smirnov windowing (uses a random subsample, so
    results are RNG-dependent; set ``random_state`` for reproducibility).
  * :class:`HcpPageHinkley` -- the classical Page-Hinkley cumulative statistic
    (fully deterministic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from pytspredit import ts_data
from scipy import stats
from scipy.special import gammaln

from .harbinger import Harbinger, as_series_array
from .utils import har_candidate_selection_highgroup, har_filter_boxplot, mas

__all__ = [
    "HcpAmoc",
    "HcpBinseg",
    "HcpBocpd",
    "HcpCfArima",
    "HcpCfEts",
    "HcpCfLr",
    "HcpChow",
    "HcpGft",
    "HcpJoinpoint",
    "HcpKswin",
    "HcpPageHinkley",
    "HcpPelt",
    "HcpScp",
]


def _detrend_residuals(y: np.ndarray) -> np.ndarray:
    """Residuals of an OLS fit ``y ~ t`` with ``t = 1..len(y)`` (R ``lm(x ~ t)``)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)
    t_mean = t.mean()
    denom = float(((t - t_mean) ** 2).sum())
    if denom == 0:
        return y - y.mean()
    slope = float(((t - t_mean) * (y - y.mean())).sum() / denom)
    intercept = float(y.mean() - slope * t_mean)
    return y - (intercept + slope * t)


class _RupturesChangePoint(Harbinger):
    """Shared machinery for the ruptures-backed offline detectors."""

    #: ruptures cost model (Gaussian mean/variance, like R's test.stat="Normal").
    model = "normal"
    min_size = 2

    def _predict_bkps(self, signal: np.ndarray) -> list[int]:  # pragma: no cover
        raise NotImplementedError

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        signal = self.serie.astype(float)
        n = len(signal)

        bkps = self._predict_bkps(signal)
        cp = np.zeros(n, dtype=bool)
        for b in bkps:
            # ruptures breakpoints are segment end indices (exclusive); the last
            # one equals n and is dropped. A breakpoint b marks a change after
            # position b-1 (0-based), matching R's cpts[i] (1-based) -> b-1.
            if 0 < b < n:
                cp[b - 1] = True

        return self._restore_refs(change_points=cp)


class HcpAmoc(_RupturesChangePoint):
    """At Most One Change (AMOC): the single most significant change in mean/variance."""

    def _predict_bkps(self, signal):
        algo = rpt.Dynp(model=self.model, min_size=self.min_size, jump=1).fit(signal)
        return algo.predict(n_bkps=1)


class HcpBinseg(_RupturesChangePoint):
    """Binary Segmentation (BinSeg).

    Parameters
    ----------
    Q:
        Maximum number of change points to search for.
    """

    def __init__(self, Q: int = 2):
        super().__init__()
        self.Q = Q

    def _predict_bkps(self, signal):
        algo = rpt.Binseg(model=self.model, min_size=self.min_size, jump=1).fit(signal)
        return algo.predict(n_bkps=self.Q)


class HcpPelt(_RupturesChangePoint):
    """Pruned Exact Linear Time (PELT).

    Parameters
    ----------
    penalty:
        Penalty value passed to ``ruptures.Pelt.predict``. When ``None`` a default
        of ``3 * log(n)`` is used -- a BIC-style penalty for the Gaussian
        mean/variance cost (two parameters per segment), chosen to give a
        practically useful segmentation. R uses the MBIC penalty, which is not
        reproduced exactly; tune ``penalty`` to control granularity.
    """

    def __init__(self, penalty: float | None = None):
        super().__init__()
        self.penalty = penalty

    def _predict_bkps(self, signal):
        n = len(signal)
        pen = self.penalty if self.penalty is not None else 3.0 * float(np.log(n))
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=1).fit(signal)
        return algo.predict(pen=pen)


class HcpKswin(Harbinger):
    """KSWIN: Kolmogorov-Smirnov windowing change detector (online, univariate).

    A sliding window is split into an early part and the most recent
    ``stat_size`` observations; a KS test between a random subsample of the early
    part and the recent part flags a change. Because of the random subsample the
    result depends on ``random_state`` and is only approximately comparable to R.

    Parameters
    ----------
    window_size:
        Size of the sliding window.
    stat_size:
        Size of the statistic subwindow used for the KS test.
    alpha:
        Significance level for the KS test.
    random_state:
        Seed for the random subsample.
    """

    def __init__(
        self,
        window_size: int = 100,
        stat_size: int = 30,
        alpha: float = 0.005,
        random_state: int | None = None,
    ):
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1.")
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")
        if stat_size <= 0:
            raise ValueError("stat_size must be greater than 0.")
        if window_size <= 2 * stat_size:
            raise ValueError("window_size must be greater than 2 * stat_size.")
        self.window_size = window_size
        self.stat_size = stat_size
        self.alpha = alpha
        self.random_state = random_state

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        serie = as_series_array(serie)
        serie = serie[~np.isnan(serie)]
        n = len(serie)
        if n == 0:
            raise ValueError("No non-missing observations were provided.")

        rng = np.random.default_rng(self.random_state)
        window = np.array([], dtype=float)
        drift_active = False
        result = np.zeros(n, dtype=bool)

        for i in range(n):
            window = np.append(window, serie[i])
            if len(window) < self.window_size:
                continue
            if len(window) > self.window_size:
                window = window[-self.window_size :]

            early = window[: self.window_size - self.stat_size]
            recent = window[self.window_size - self.stat_size :]
            sample = rng.choice(early, size=self.stat_size, replace=True)
            ks = stats.ks_2samp(sample, recent, method="exact")

            drift_now = np.isfinite(ks.pvalue) and ks.pvalue < self.alpha and ks.statistic > 0.1
            pred = bool(drift_now and not drift_active)
            drift_active = bool(drift_now)
            if pred:
                window = recent
            result[i] = pred

        return self._build_cp_table(n, result)

    @staticmethod
    def _build_cp_table(n, result):
        type_arr = np.array([""] * n, dtype=object)
        type_arr[result] = "changepoint"

        return pd.DataFrame({"idx": np.arange(n), "event": result, "type": type_arr})


class HcpPageHinkley(Harbinger):
    """Page-Hinkley online change-point detector (univariate, deterministic).

    Parameters
    ----------
    min_instances:
        Minimum number of observations before a change can be reported.
    delta:
        Slack term subtracted from the deviation score.
    threshold:
        Detection threshold for the cumulative statistic.
    alpha:
        Forgetting factor applied to the cumulative score.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50,
        alpha: float = 1 - 1e-4,
    ):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        serie = as_series_array(serie)
        serie = serie[~np.isnan(serie)]
        n = len(serie)
        if n == 0:
            raise ValueError("No non-missing observations were provided.")

        state = {"x_mean": 0.0, "sum": 0.0, "sample_count": 1}

        def update(x):
            state["x_mean"] += (x - state["x_mean"]) / state["sample_count"]
            state["sum"] = max(0.0, self.alpha * state["sum"] + (x - state["x_mean"] - self.delta))
            state["sample_count"] += 1
            if state["sample_count"] < self.min_instances:
                return False
            if state["sum"] > self.threshold:
                state["x_mean"] = 0.0
                state["sum"] = 0.0
                state["sample_count"] = 1
                return True
            return False

        result = np.zeros(n, dtype=bool)
        # R primes the statistic with the first observation before the main loop.
        update(serie[0])
        for i in range(n):
            result[i] = update(serie[i])

        type_arr = np.array([""] * n, dtype=object)
        type_arr[result] = "changepoint"

        return pd.DataFrame({"idx": np.arange(n), "event": result, "type": type_arr})


# --- hcp_cf_lr (ChangeFinder with linear regression) ------------------------


class HcpCfLr(Harbinger):
    """ChangeFinder using linear regression (Python port of R ``hcp_cf_lr``).

    Two-stage ChangeFinder: a linear trend is fit to the series and the squared
    residuals are scored for anomalies; the residual signal is then smoothed
    (moving average), re-fit, and re-scored to flag change points.

    Parameters
    ----------
    sw_size:
        Sliding-window size for the moving-average smoothing / warm-up.
    """

    def __init__(self, sw_size: int = 30):
        super().__init__()
        self.sw_size = sw_size

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        x = self.serie
        n = len(x)
        sw = self.sw_size

        res = self.har_distance(_detrend_residuals(x))
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, x)
        anomalies = np.asarray(selection.mask, dtype=bool).copy()
        anomalies[:sw] = False

        y = mas(res, sw)
        u = self.har_distance(_detrend_residuals(y))
        u = mas(u, sw)
        cp_outliers = self.har_outliers(u)
        cp_selection = self.har_outliers_check(cp_outliers, u, u)
        cp = np.asarray(cp_selection.mask, dtype=bool).copy()
        cp[:sw] = False
        cp = np.concatenate([np.zeros(n - len(u), dtype=bool), cp])

        return self._restore_refs(anomalies=anomalies, change_points=cp, res=res)


# --- hcp_scp (seminal change point) -----------------------------------------


class HcpScp(Harbinger):
    """Seminal change point via windowed local regression (port of R ``hcp_scp``).

    For each sliding window a linear model on the whole window is compared against
    two linear models fitted on the two halves split around the window's central
    (seminal) observation. The drop in mean deviation is the change score, scored
    with the boxplot filter and high-group candidate selection.

    Parameters
    ----------
    sw_size:
        Sliding-window size.
    """

    def __init__(self, sw_size: int = 30):
        super().__init__()
        self.sw_size = sw_size
        self.har_outliers = har_filter_boxplot
        self.har_outliers_check = har_candidate_selection_highgroup

    def _window_score(self, data: np.ndarray, offset: int) -> float:
        err = float(np.mean(self.har_distance(_detrend_residuals(data))))
        left = data[: offset - 1]
        right = data[offset:]
        res_ad = self.har_distance(
            np.concatenate([_detrend_residuals(left), _detrend_residuals(right)])
        )
        err_ad = float(np.mean(res_ad))
        return err - err_ad

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        sw = self.sw_size

        windows = ts_data(self.serie, sw)
        windows = windows.to_numpy() if hasattr(windows, "to_numpy") else np.asarray(windows)
        offset = round(sw / 2)

        res = np.array([self._window_score(windows[i], offset) for i in range(windows.shape[0])])
        values = windows[:, offset]

        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)
        cp = np.asarray(selection.mask, dtype=bool)

        front = np.zeros(offset, dtype=bool)
        back = np.zeros(sw - offset - 1, dtype=bool)
        cp_full = np.concatenate([front, cp, back])
        res_full = np.concatenate([np.zeros(offset), res, np.zeros(sw - offset - 1)])

        return self._restore_refs(
            change_points=cp_full, res=res_full, threshold=selection.threshold
        )


# --- hcp_joinpoint (Joinpoint Regression++) ---------------------------------


def _segment_rss(y: np.ndarray, x: np.ndarray) -> float:
    """Residual sum of squares of an OLS fit ``y ~ x`` over a segment.

    Uses a QR least-squares solve (``numpy.linalg.lstsq``) to mirror R's ``lm``:
    on a perfectly linear segment this leaves the same tiny (non-zero) round-off
    residual R does, which matters for the ``log(RSS)`` term in the BIC selection.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.inf
    x = np.asarray(x, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def _bic(rss: float, n: int, k: int, factor: int) -> float:
    if not np.isfinite(rss) or rss <= 0:
        return np.inf
    p = factor * k + 2
    return float(np.log(rss / n) + (p / n) * np.log(n))


def _joinpoint_weight(y: np.ndarray, cps: list[int]) -> float:
    if len(cps) == 0:
        return 0.0
    diffs = []
    for tau in sorted(cps):
        left = np.arange(1, tau)  # R seq_len(tau - 1)
        right = np.arange(tau, len(y) + 1)  # R tau:length(y)
        if len(left) < 3 or len(right) < 3:
            continue
        r1 = np.corrcoef(y[left - 1], left)[0, 1] ** 2
        r2 = np.corrcoef(y[right - 1], right)[0, 1] ** 2
        diffs.append(abs(r2 - r1))
    if not diffs:
        return 0.0
    w = float(np.mean(diffs))
    if not np.isfinite(w):
        return 0.0
    return min(max(w, 0.0), 1.0)


class HcpJoinpoint(Harbinger):
    """Joinpoint Regression++ change-point detector (port of R ``hcp_joinpoint``).

    Searches for a globally optimal set of joinpoints with dynamic programming,
    fits linear segments between candidate breaks, and selects the number of
    joinpoints (0..``k_max``) by a weighted BIC (a blend of BIC and BIC3).

    Parameters
    ----------
    min_between:
        Minimum number of observations between consecutive joinpoints.
    min_end:
        Minimum number of observations in the first and last segments.
    k_max:
        Maximum number of joinpoints considered during model selection.
    log_transform:
        Whether to log-transform the (strictly positive) series before fitting.
    """

    def __init__(
        self,
        min_between: int = 2,
        min_end: int = 2,
        k_max: int = 5,
        log_transform: bool = False,
    ):
        super().__init__()
        if min_between < 1:
            raise ValueError("min_between must be a positive number.")
        if min_end < 1:
            raise ValueError("min_end must be a positive number.")
        if k_max < 0:
            raise ValueError("k_max must be a non-negative number.")
        self.min_between = int(min_between)
        self.min_end = int(min_end)
        self.k_max = int(k_max)
        self.log_transform = bool(log_transform)
        self.model: dict | None = None

    def _max_jp(self, n: int) -> int:
        if n < 2 * self.min_end:
            raise ValueError("The series is too short for the requested segment constraints.")
        max_possible = max(0, (n - 2 * self.min_end) // self.min_between)
        return min(self.k_max, max_possible)

    def _dp(self, y: np.ndarray, x: np.ndarray, max_jp: int):
        n = len(y)
        # 1-based indexing emulated with length n+1 arrays (index 0 unused).
        dp_rss = [np.full(n + 1, np.inf) for _ in range(max_jp + 1)]
        dp_cps: list[list] = [[None] * (n + 1) for _ in range(max_jp + 1)]

        for j in range(1, n + 1):
            if j >= self.min_end:
                dp_rss[0][j] = _segment_rss(y[:j], x[:j])
                dp_cps[0][j] = []

        for k in range(1, max_jp + 1):
            min_j = max(self.min_end + self.min_between, 2 * self.min_end)
            for j in range(min_j, n + 1):
                best_rss = np.inf
                best_cps: list[int] = []
                i_min = self.min_end + 1
                i_max = j - self.min_end
                if i_min > i_max:
                    continue
                for i in range(i_min, i_max + 1):
                    if j - i + 1 < self.min_end:
                        continue
                    prev_cps = dp_cps[k - 1][i - 1]
                    prev_rss = dp_rss[k - 1][i - 1]
                    if prev_cps is None or not np.isfinite(prev_rss):
                        continue
                    if len(prev_cps) > 0 and (i - prev_cps[-1]) < self.min_between:
                        continue
                    seg_rss = _segment_rss(y[i - 1 : j], x[i - 1 : j])
                    total = prev_rss + seg_rss
                    if total < best_rss:
                        best_rss = total
                        best_cps = [*prev_cps, i]
                dp_rss[k][j] = best_rss
                dp_cps[k][j] = best_cps
        return dp_rss, dp_cps

    def fit(self, serie, **kwargs) -> HcpJoinpoint:
        if serie is None:
            raise ValueError("No data was provided.")
        self._store_refs(serie)
        y = np.asarray(self.serie, dtype=float)
        n = len(y)
        if n < 2 * self.min_end:
            raise ValueError("The series is too short for the requested min_end value.")
        if self.log_transform:
            if np.any(y <= 0):
                raise ValueError("log_transform=True requires strictly positive values.")
            y = np.log(y)
        x = np.arange(1, n + 1, dtype=float)

        max_jp = self._max_jp(n)
        dp_rss, dp_cps = self._dp(y, x, max_jp)

        best_wbic = np.inf
        best_cps: list[int] = []
        for k in range(max_jp + 1):
            cps = dp_cps[k][n] or []
            rss = dp_rss[k][n]
            bic = _bic(rss, n, k, factor=2)
            bic3 = _bic(rss, n, k, factor=3)
            w = _joinpoint_weight(y, cps)
            wbic = (1 - w) * bic + w * bic3
            if np.isfinite(wbic) and wbic < best_wbic:
                best_wbic = wbic
                best_cps = cps
        if not np.isfinite(best_wbic):
            raise ValueError("No valid joinpoint model could be fitted with the current constraints.")
        self.model = {"cps": best_cps, "wbic": best_wbic}
        return self

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided.")
        self._store_refs(serie)
        self.fit(self.serie)
        n = len(self.serie)
        cp = np.zeros(n, dtype=bool)
        for raw in self.model["cps"] if self.model else []:
            c = round(raw)
            if 1 <= c <= n:
                cp[c - 1] = True
        return self._restore_refs(change_points=cp)


# --- hcp_chow / hcp_gft (structural change, "++" re-engineering) --------------
#
# Origin: R ``hcp_chow`` / ``hcp_gft`` rely on the ``strucchange`` package --
# ``Fstats`` (a homoskedastic Chow/sup-F test on the *mean* regression ``y~t``)
# and ``breakpoints`` (the Bai & Perron least-squares dynamic program with a
# Gaussian BIC). The 0.4.1 faithful port reproduced that: prefix-sum OLS RSS,
# the Bai-Perron DP, and a reverse-engineered ``strucchange`` BIC
# (``n*(log(2*pi)+1+log(RSS/n)) + log(n)*((m+1)*k + m + 1)``, k = 2).
#
# "++" ADAPTATION (this release): we generalise the *least-squares (mean-only)*
# criterion to a **Gaussian mean-AND-variance segmentation cost**, so the
# detectors also find scale/variance-regime changes that the original mean-shift
# model is blind to (e.g. volatility / amplitude shifts). The segment cost is the
# Gaussian maximum-likelihood deviance ``m * log(RSS_seg / m)`` (per-segment
# variance), inspired by the normal mean+variance change-point segmentation of
# Chen & Gupta and Killick et al. For the number of breaks, ``HcpGft`` adopts the
# conservative LWZ information criterion (Liu, Wu & Zidek 1997) in place of plain
# BIC, which curbs over-segmentation; ``HcpChow`` selects the single break that
# maximises the mean+variance log-likelihood ratio (a scale-aware generalisation
# of Chow's homoskedastic F-test, scanned over the unknown break à la Andrews).
# The Bai-Perron DP inner loop is vectorised with numpy (prefix sums), keeping it
# O(max_breaks * n) per candidate count rather than a triple Python loop.
#
# References (origin + inspiration for the ++):
#   - Chow GC (1960). Tests of equality between sets of coefficients in two
#     linear regressions. Econometrica 28(3):591-605.
#   - Bai J, Perron P (1998). Estimating and testing linear models with multiple
#     structural changes. Econometrica 66(1):47-78. (2003) J Appl Econometrics
#     18:1-22 -- dynamic-programming computation of multiple breakpoints.
#   - Zeileis A, Leisch F, Hornik K, Kleiber C (2002). strucchange: An R package
#     for testing for structural change in linear regression models. JSS 7(2).
#   - Andrews DWK (1993). Tests for parameter instability and structural change
#     with unknown change point. Econometrica 61(4):821-856. (sup-type scan)
#   - Liu J, Wu S, Zidek JV (1997). On segmented multivariate regression.
#     Statistica Sinica 7:497-525. (LWZ information criterion)
#   - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.
#     Birkhauser. (Gaussian mean+variance change points)
#   - Killick R, Fearnhead P, Eckley IA (2012). Optimal detection of
#     changepoints with a linear computational cost. JASA 107(500):1590-1598.


def _mv_prefix(y):
    """Prefix sums for O(1) OLS ``y~t`` RSS of 1-based segments, scalar + vectorised.

    Returns ``(rss, rss_vec, n)`` where ``rss(i, j)`` is the residual sum of
    squares of the trend fit on the segment ``i..j`` (1-based inclusive) and
    ``rss_vec(i_arr, j)`` evaluates it for an array of left endpoints ``i`` at a
    fixed right endpoint ``j`` (used to vectorise the Bai-Perron DP).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(1, n + 1, dtype=float)
    sx = np.concatenate(([0.0], np.cumsum(x)))
    sy = np.concatenate(([0.0], np.cumsum(y)))
    sxx = np.concatenate(([0.0], np.cumsum(x * x)))
    syy = np.concatenate(([0.0], np.cumsum(y * y)))
    sxy = np.concatenate(([0.0], np.cumsum(x * y)))

    def rss(i: int, j: int) -> float:
        m = j - i + 1
        if m <= 0:
            return 0.0
        a = sx[j] - sx[i - 1]
        b = sy[j] - sy[i - 1]
        sxx_c = (sxx[j] - sxx[i - 1]) - a * a / m
        syy_c = (syy[j] - syy[i - 1]) - b * b / m
        sxy_c = (sxy[j] - sxy[i - 1]) - a * b / m
        r = syy_c if sxx_c <= 0 else syy_c - sxy_c * sxy_c / sxx_c
        return max(float(r), 0.0)

    def rss_vec(i_arr: np.ndarray, j: int) -> np.ndarray:
        i_arr = np.asarray(i_arr, dtype=int)
        m = j - i_arr + 1
        a = sx[j] - sx[i_arr - 1]
        b = sy[j] - sy[i_arr - 1]
        sxx_c = (sxx[j] - sxx[i_arr - 1]) - a * a / m
        syy_c = (syy[j] - syy[i_arr - 1]) - b * b / m
        sxy_c = (sxy[j] - sxy[i_arr - 1]) - a * b / m
        r = np.where(sxx_c <= 0, syy_c, syy_c - sxy_c * sxy_c / np.where(sxx_c <= 0, 1.0, sxx_c))
        return np.maximum(r, 0.0)

    return rss, rss_vec, n


def _mv_seg_cost(r, m):
    """Gaussian mean+variance ML deviance of a segment: ``m * log(RSS/m)``."""
    return m * np.log(np.maximum(r, 1e-12) / m)


def _break_penalty(n: int, m: int, kind: str, k: int = 2) -> float:
    """Model-selection penalty for ``m`` breaks (``(m+1)`` segments).

    Each segment has ``k`` trend parameters plus one variance; ``m`` break
    locations are also counted. ``"lwz"`` is the conservative Liu-Wu-Zidek
    criterion ``c * (ln n)^(2+d) * p`` (c=0.299, d=0.1); ``"bic"`` is ``ln(n)*p``.
    """
    p = (m + 1) * (k + 1) + m
    if kind == "lwz":
        return 0.299 * (np.log(n) ** 2.1) * p
    return np.log(n) * p


def _meanvar_breakpoints(y, h_frac: float = 0.15, penalty: str = "lwz") -> list[int]:
    """Bai-Perron breakpoints under a Gaussian mean+variance cost (1-based).

    Vectorised dynamic program minimising the total segment deviance
    ``sum_s m_s*log(RSS_s/m_s)``; the number of breaks is chosen by the LWZ
    (default) or BIC penalty. This is the "++" generalisation of
    ``strucchange``'s least-squares ``breakpoints``.
    """
    rss, rss_vec, n = _mv_prefix(y)
    h = int(np.floor(n * h_frac))
    if h < 1:
        return []
    max_m = n // h - 1
    if max_m < 1:
        return []

    opt = np.full((max_m + 1, n + 1), np.inf)
    prev = np.full((max_m + 1, n + 1), -1, dtype=int)
    for j in range(h, n + 1):
        opt[0, j] = _mv_seg_cost(rss(1, j), j)
    for m in range(1, max_m + 1):
        for j in range((m + 1) * h, n + 1):
            b = np.arange(m * h, j - h + 1)
            prev_cost = opt[m - 1, b]
            finite = np.isfinite(prev_cost)
            if not finite.any():
                continue
            seg = _mv_seg_cost(rss_vec(b + 1, j), j - b)  # m_seg = j-(b+1)+1 = j-b
            total = np.where(finite, prev_cost + seg, np.inf)
            k = int(np.argmin(total))
            opt[m, j] = total[k]
            prev[m, j] = b[k]

    best_m = 0
    best_crit = opt[0, n] + _break_penalty(n, 0, penalty)
    for m in range(1, max_m + 1):
        if not np.isfinite(opt[m, n]):
            continue
        crit = opt[m, n] + _break_penalty(n, m, penalty)
        if crit < best_crit:
            best_crit, best_m = crit, m

    cps: list[int] = []
    j, m = n, best_m
    while m > 0:
        b = int(prev[m, j])
        cps.append(b)
        j, m = b, m - 1
    return sorted(cps)


def _meanvar_single_break(y, h_frac: float = 0.15) -> int | None:
    """Single break maximising the Gaussian mean+variance log-likelihood ratio.

    Scans the trimmed range for the split that most reduces the segment deviance
    ``m*log(RSS/m)`` -- a scale-aware generalisation of Chow's homoskedastic
    sup-F, which only tests the *mean* coefficients (Chow 1960; Andrews 1993).
    Returns the 1-based break index or ``None``.
    """
    rss, rss_vec, n = _mv_prefix(y)
    h = int(np.floor(n * h_frac))
    if h < 1 or h > n - h:
        return None
    full = _mv_seg_cost(rss(1, n), n)
    i = np.arange(h, n - h + 1)
    if len(i) == 0:
        return None
    # right segment (i+1 .. n) vectorises over the left endpoint at fixed j = n;
    # the left segment (1 .. i) is a cheap prefix evaluation.
    left = _mv_seg_cost(np.array([rss(1, ii) for ii in i]), i)
    right = _mv_seg_cost(rss_vec(i + 1, n), n - i)
    lr = full - (left + right)
    return int(i[int(np.argmax(lr))])


class HcpChow(Harbinger):
    """Scale-aware structural-break detector ("++" evolution of R ``hcp_chow``).

    How it works / theory
    ---------------------
    Reports the single most significant structural break in the series by scanning
    a trimmed range for the split point that maximises the Gaussian **mean +
    variance** log-likelihood ratio (the ``m*log(RSS/m)`` deviance of the two
    sub-segments vs the whole). This is a scale-aware generalisation of the
    classical Chow test: Chow's F-statistic only tests the *mean* regression
    coefficients (homoskedastic errors), so it is blind to a pure variance/scale
    break; scanning the likelihood ratio over an unknown break date follows the
    sup-type construction of Andrews (1993).

    Origin & adaptation
    -------------------
    Origin: R ``hcp_chow`` uses ``strucchange::Fstats`` (homoskedastic sup-F on
    ``y ~ t``). **"++" adaptation:** the homoskedastic F-statistic is replaced by
    the mean+variance Gaussian log-likelihood ratio, so the detector also catches
    variance-regime changes. Measured on ``examples_changepoints`` (SoftED-F1 vs
    ground truth) this lifts the summed F1 over the seven series from the faithful
    port's 3.40 (R ``strucchange`` 3.31) to **3.73**.

    Parameters
    ----------
    h:
        Minimal segment size as a fraction of the series length (trimming).

    References
    ----------
    - Chow GC (1960). Tests of equality between sets of coefficients in two linear
      regressions. Econometrica 28(3):591-605.
    - Andrews DWK (1993). Tests for parameter instability and structural change
      with unknown change point. Econometrica 61(4):821-856.
    - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.
    - Zeileis A et al. (2002). strucchange. J. Statistical Software 7(2).

    Porting
    -------
    Mirrors ``hcp_chow.R``. Fidelity: adapted (++).
    - Replaces ``strucchange``'s homoskedastic Chow sup-F (mean coefficients only)
      with a scale-aware mean+variance Gaussian LR scan over the unknown break.
      Parity with R is intentionally not preserved; the acceptance criterion is
      detection quality vs ground truth (see ``develop/benchmark_backimpl.py``).
    """

    def __init__(self, h: float = 0.15):
        super().__init__()
        self.h = h

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        n = len(self.serie)
        cp = np.zeros(n, dtype=bool)
        bp = _meanvar_single_break(self.serie, self.h)
        if bp is not None:
            cp[bp - 1] = True
        return self._restore_refs(change_points=cp)


class HcpGft(Harbinger):
    """Multiple structural-break detector ("++" evolution of R ``hcp_gft``).

    How it works / theory
    ---------------------
    Finds multiple structural breaks with the Bai & Perron dynamic program, but
    under a Gaussian **mean + variance** segment cost -- each segment pays the ML
    deviance ``m*log(RSS/m)`` (per-segment variance), so the optimal partition
    places breaks at level/trend *and* scale changes. The number of breaks is
    chosen by the conservative LWZ information criterion (Liu, Wu & Zidek 1997),
    ``c*(ln n)^(2+d)*p`` (c=0.299, d=0.1), which penalises extra breaks more
    heavily than BIC and curbs over-segmentation. The DP inner loop is vectorised
    with prefix sums.

    Origin & adaptation
    -------------------
    Origin: R ``hcp_gft`` uses ``strucchange::breakpoints`` -- the least-squares
    (mean-only) Bai-Perron program with a Gaussian BIC. **"++" adaptation:** (1)
    the least-squares cost is generalised to the mean+variance Gaussian deviance
    (cf. normal mean+variance change-point segmentation, Chen & Gupta 2000;
    Killick et al. 2012), capturing variance regimes the mean-shift model misses;
    (2) BIC is replaced by the conservative LWZ criterion; (3) the DP is
    vectorised. Measured on ``examples_changepoints`` (SoftED-F1 vs ground truth)
    this lifts the summed F1 over the seven series from the faithful port's 2.44
    (R ``strucchange`` 2.20) to **3.17**.

    Parameters
    ----------
    h:
        Minimal segment size as a fraction of the series length.
    penalty:
        Break-count information criterion: ``"lwz"`` (default, conservative) or
        ``"bic"``.

    References
    ----------
    - Bai J, Perron P (1998, 2003). Estimating/testing linear models with multiple
      structural changes; dynamic-programming computation. Econometrica 66(1);
      J. Applied Econometrics 18.
    - Liu J, Wu S, Zidek JV (1997). On segmented multivariate regression.
      Statistica Sinica 7:497-525. (LWZ criterion)
    - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.
    - Killick R, Fearnhead P, Eckley IA (2012). Optimal detection of changepoints
      with a linear computational cost. JASA 107(500):1590-1598.
    - Zeileis A et al. (2002). strucchange. J. Statistical Software 7(2).

    Porting
    -------
    Mirrors ``hcp_gft.R``. Fidelity: adapted (++).
    - Generalises ``strucchange``'s least-squares Bai-Perron breakpoints to a
      Gaussian mean+variance cost with an LWZ penalty. Parity with R is
      intentionally not preserved; the acceptance criterion is detection quality
      vs ground truth (see ``develop/benchmark_backimpl.py``).
    """

    def __init__(self, h: float = 0.15, penalty: str = "lwz"):
        super().__init__()
        self.h = h
        self.penalty = penalty

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        n = len(self.serie)
        cp = np.zeros(n, dtype=bool)
        for bp in _meanvar_breakpoints(self.serie, self.h, self.penalty):
            if 1 <= bp <= n:
                cp[bp - 1] = True
        return self._restore_refs(change_points=cp)


# --- hcp_cf_arima / hcp_cf_ets (robust ChangeFinder, "++" re-engineering) ------
#
# Origin: R ``hcp_cf_arima`` / ``hcp_cf_ets`` are two-stage ChangeFinder-inspired
# detectors that score the residuals of a ``forecast`` model (``auto.arima`` /
# ``ets``), smooth them, re-fit, and re-score. They cite Takeuchi & Yamanishi
# (2006) but do not implement its SDAR core; they also depend on ``forecast``.
#
# "++" ADAPTATION (this release): a native, self-contained **robust ChangeFinder**
# that fuses two complementary signals and takes their UNION:
#   1. the *canonical* ChangeFinder of Takeuchi & Yamanishi (2006): a two-stage
#      online **SDAR** (Sequentially Discounting AutoRegressive) model with a
#      forgetting factor, scored by the logarithmic loss -log p(x_t | past). This
#      is strong on abrupt, variance and multi-change regimes. ``HcpCfArima`` uses
#      an autoregressive SDAR predictor; ``HcpCfEts`` uses a discounted
#      exponential-smoothing (Holt level+trend) predictor.
#   2. a Gaussian **mean+variance structural break** (the ``_meanvar_single_break``
#      engine shared with HcpChow/HcpGft), which recovers *smooth trend/slope*
#      changes that the AR-surprise score is blind to.
#
# Pure numpy -- no ``forecast`` and no ``statsmodels``. Because this diverges from
# ``forecast``, the acceptance criterion is detection quality vs ground truth
# (SoftED-F1), not R parity. Measured on ``examples_changepoints`` the summed F1
# over the seven series rises well above the faithful-port / R baselines:
#   * HcpCfArima: ~3.94  (faithful port 2.78 / R forecast 2.92)
#   * HcpCfEts:   ~3.74  (faithful port 2.66 / R forecast 2.66)
#
# References:
#   - Takeuchi J, Yamanishi K (2006). A unifying framework for detecting outliers
#     and change points from time series. IEEE TKDE 18(4):482-492. (ChangeFinder /
#     SDAR)
#   - Yamanishi K, Takeuchi J (2002). A unifying framework for detecting outliers
#     and change points from non-stationary time series data. KDD.
#   - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.
#     (mean+variance structural break used for the trend component)


def _causal_ma(x, w: int) -> np.ndarray:
    """Trailing (causal) moving average of window ``w`` (same length as ``x``)."""
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - w + 1)
        out[t] = (cs[t + 1] - cs[lo]) / (t + 1 - lo)
    return out


def _sdar_ar_scores(x: np.ndarray, r: float, order: int) -> np.ndarray:
    """Online SDAR autoregressive log-loss scores (Takeuchi & Yamanishi 2006).

    Maintains a discounted mean, discounted autocovariances (forgetting factor
    ``r``) and a discounted residual variance; the AR coefficients are the
    discounted Yule-Walker solution. The score is the logarithmic loss
    ``-log N(x_t; x_hat_t, sigma^2)`` of each observation under the one-step
    predictive Gaussian.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = 0.0
    sigma = 1.0
    c = np.zeros(order + 1)
    scores = np.zeros(n)
    for t in range(n):
        xt = x[t]
        mu = (1 - r) * mu + r * xt
        if t < order:
            continue
        for i in range(1, order + 1):
            c[i] = (1 - r) * c[i] + r * (xt - mu) * (x[t - i] - mu)
        c[0] = (1 - r) * c[0] + r * (xt - mu) ** 2
        if c[0] <= 1e-12:
            a = np.zeros(order)
        elif order == 1:
            a = np.array([c[1] / c[0]])
        else:
            idx = np.abs(np.subtract.outer(np.arange(order), np.arange(order)))
            toeplitz = c[idx]
            a = np.linalg.solve(toeplitz + 1e-8 * np.eye(order), c[1:order + 1])
        xhat = mu + float(np.dot(a, x[t - order:t][::-1] - mu))
        sigma = (1 - r) * sigma + r * (xt - xhat) ** 2
        sigma = max(sigma, 1e-12)
        scores[t] = 0.5 * np.log(2 * np.pi * sigma) + (xt - xhat) ** 2 / (2 * sigma)
    return scores


def _sdar_holt_scores(x: np.ndarray, alpha: float, beta: float, r: float) -> np.ndarray:
    """Online discounted Holt (level+trend) log-loss scores.

    An exponential-smoothing analogue of SDAR: the one-step forecast is the Holt
    level+trend prediction, the residual variance is tracked with forgetting
    factor ``r``, and the score is the predictive logarithmic loss.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    level = float(x[0])
    trend = float(x[1] - x[0]) if n > 1 else 0.0
    sigma = 1.0
    scores = np.zeros(n)
    for t in range(n):
        xt = x[t]
        if t < 2:
            level = xt
            continue
        xhat = level + trend
        e = xt - xhat
        sigma = (1 - r) * sigma + r * e * e
        sigma = max(sigma, 1e-12)
        scores[t] = 0.5 * np.log(2 * np.pi * sigma) + e * e / (2 * sigma)
        prev_level = level
        level = alpha * xt + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return scores


class _HcpRobustChangeFinder(Harbinger):
    """Robust ChangeFinder base: two-stage SDAR surprise UNION mean+variance break.

    Subclasses provide :meth:`_online_scores` (the one-step predictive log-loss of
    the online model). The change-point signal is the two-stage ChangeFinder score
    (score -> smooth -> re-score -> smooth), selected with the usual outlier
    filter + candidate selection, and then combined (union) with the single
    Gaussian mean+variance structural break so smooth trend/slope changes are also
    captured. The first-stage smoothed score yields the anomaly flags.
    """

    def _online_scores(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _changefinder(self, x: np.ndarray, smooth: int, h: float):
        n = len(x)
        smooth = max(1, int(smooth))

        s1 = self._online_scores(x)
        sm1 = _causal_ma(s1, smooth)

        a_out = self.har_outliers(sm1)
        a_sel = self.har_outliers_check(a_out, sm1, x)
        anomalies = np.asarray(a_sel.mask, dtype=bool).copy()
        anomalies[:smooth] = False

        s2 = self._online_scores(sm1)
        cp_score = _causal_ma(s2, max(1, round(smooth / 2)))

        cp_out = self.har_outliers(cp_score)
        cp_sel = self.har_outliers_check(cp_out, cp_score, cp_score)
        cp = np.asarray(cp_sel.mask, dtype=bool).copy()
        cp[:smooth] = False

        # Union with the mean+variance structural break (recovers trend changes).
        bp = _meanvar_single_break(x, h)
        if bp is not None and n >= bp >= 1:
            cp[bp - 1] = True

        if n > 0:
            cp[0] = False

        return self._restore_refs(anomalies=anomalies, change_points=cp, res=cp_score)


class HcpCfArima(_HcpRobustChangeFinder):
    """Robust ChangeFinder with an autoregressive SDAR model ("++" of R ``hcp_cf_arima``).

    How it works / theory
    ---------------------
    The canonical two-stage ChangeFinder of Takeuchi & Yamanishi (2006): an online
    **SDAR** (Sequentially Discounting AutoRegressive) model learns an AR(``order``)
    predictor with a forgetting factor ``r`` and scores each point by its
    predictive logarithmic loss; the scores are smoothed, re-scored by a second
    SDAR, and smoothed again. These change scores are strong on abrupt / variance /
    multi-change regimes. Their union with a Gaussian **mean+variance structural
    break** (shared with :class:`HcpChow`/:class:`HcpGft`) recovers the smooth
    trend/slope changes the AR-surprise score is blind to.

    Origin & adaptation
    -------------------
    Origin: R ``hcp_cf_arima`` scores ``forecast::auto.arima`` residuals (it cites
    ChangeFinder but does not implement SDAR, and depends on ``forecast``).
    **"++" adaptation:** a native, pure-numpy implementation of the *canonical*
    SDAR ChangeFinder fused with the mean+variance break -- no ``forecast``, no
    ``statsmodels``. Judged by detection quality vs ground truth (not R parity):
    on ``examples_changepoints`` the summed SoftED-F1 is ~3.94 vs the faithful
    port's 2.78 and R ``forecast``'s 2.92.

    Parameters
    ----------
    sw_size:
        Smoothing window for the two-stage scores / warm-up.
    r:
        SDAR forgetting (discounting) factor.
    order:
        Autoregressive order of the SDAR model.
    h:
        Minimal segment fraction for the mean+variance structural break.

    References
    ----------
    - Takeuchi J, Yamanishi K (2006). A unifying framework for detecting outliers
      and change points from time series. IEEE TKDE 18(4):482-492.
    - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.

    Porting
    -------
    Mirrors ``hcp_cf_arima.R``. Fidelity: adapted (++).
    - Replaces the ``forecast``-residual approximation with the canonical SDAR
      ChangeFinder (Takeuchi & Yamanishi 2006) unioned with a mean+variance break;
      pure numpy, no ``forecast``/``statsmodels``. Parity with R is intentionally
      not preserved; see ``develop/benchmark_backimpl.py``.
    """

    def __init__(self, sw_size: int = 5, r: float = 0.07, order: int = 1, h: float = 0.15):
        super().__init__()
        self.sw_size = sw_size
        self.r = r
        self.order = order
        self.h = h

    def _online_scores(self, x):
        return _sdar_ar_scores(np.asarray(x, dtype=float), self.r, self.order)

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        return self._changefinder(np.asarray(self.serie, dtype=float), self.sw_size, self.h)


class HcpCfEts(_HcpRobustChangeFinder):
    """Robust ChangeFinder with a discounted Holt model ("++" of R ``hcp_cf_ets``).

    How it works / theory
    ---------------------
    The same two-stage robust ChangeFinder as :class:`HcpCfArima`, but the online
    predictive model is a **discounted Holt** (exponential-smoothing level+trend)
    forecaster -- the exponential-smoothing analogue of SDAR -- scored by its
    predictive logarithmic loss. The SDAR change scores are unioned with the
    Gaussian mean+variance structural break so both abrupt/variance and smooth
    trend changes are captured.

    Origin & adaptation
    -------------------
    Origin: R ``hcp_cf_ets`` scores ``forecast::ets`` residuals (depends on
    ``forecast``). **"++" adaptation:** a native, pure-numpy discounted-Holt
    ChangeFinder fused with the mean+variance break -- no ``forecast``, no
    ``statsmodels``. Judged by detection quality vs ground truth: on
    ``examples_changepoints`` the summed SoftED-F1 is ~3.74 vs the faithful port's
    2.66 and R ``forecast``'s 2.66.

    Parameters
    ----------
    sw_size:
        Smoothing window for the two-stage scores / warm-up.
    alpha, beta:
        Holt level and trend smoothing factors.
    r:
        Forgetting (discounting) factor for the residual variance.
    h:
        Minimal segment fraction for the mean+variance structural break.

    References
    ----------
    - Takeuchi J, Yamanishi K (2006). A unifying framework for detecting outliers
      and change points from time series. IEEE TKDE 18(4):482-492.
    - Holt CC (1957/2004). Forecasting seasonals and trends by exponentially
      weighted moving averages. Int. J. Forecasting 20(1):5-10.
    - Chen J, Gupta AK (2000). Parametric Statistical Change Point Analysis.

    Porting
    -------
    Mirrors ``hcp_cf_ets.R``. Fidelity: adapted (++).
    - Replaces the ``forecast``-residual approximation with a discounted-Holt
      ChangeFinder unioned with a mean+variance break; pure numpy, no
      ``forecast``/``statsmodels``. Parity with R intentionally not preserved.
    """

    def __init__(
        self,
        sw_size: int = 5,
        alpha: float = 0.8,
        beta: float = 0.1,
        r: float = 0.05,
        h: float = 0.15,
    ):
        super().__init__()
        self.sw_size = sw_size
        self.alpha = alpha
        self.beta = beta
        self.r = r
        self.h = h

    def _online_scores(self, x):
        return _sdar_holt_scores(np.asarray(x, dtype=float), self.alpha, self.beta, self.r)

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        return self._changefinder(np.asarray(self.serie, dtype=float), self.sw_size, self.h)


# --- hcp_bocpd (Bayesian Online Change Point Detection) ---------------------

_BOCPD_XMIN = np.finfo(float).tiny          # R ``.Machine$double.xmin``
_BOCPD_LOG_XMIN = np.log(_BOCPD_XMIN)


def _bocpd_studentpdf(x, mu, var, nu):
    """Student-t predictive density (port of ``ocp:::studentpdf``).

    ``c * (1 + (x-mu)^2 / (nu*var))^(-(nu+1)/2)`` with
    ``c = exp(lgamma(nu/2+0.5) - lgamma(nu/2)) / sqrt(nu*pi*var)``.
    """
    cc = np.exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2)) / np.sqrt(nu * np.pi * var)
    return cc * (1.0 + (1.0 / (nu * var)) * (x - mu) ** 2) ** (-(nu + 1) / 2)


def _bocpd_run(serie, hazard, m, k, a, b, cpthreshold, min_rlength,
               max_rlength, trunc_rlim):
    """Native reimplementation of ``ocp::onlineCPD`` (gaussian UPM, univariate)
    plus the ``changepoint_lists`` voting of R ``hcp_bocpd``.

    Reproduces the Adams & MacKay run-length posterior recursion with a
    Normal-Inverse-Gamma conjugate prior (Student-t predictive), and the three
    change-point extraction lists ``ocp`` maintains -- ``colmaxes`` (most likely
    non-zero run length per step), ``threshcps`` (run lengths whose posterior
    mass exceeds ``cpthreshold``) and ``maxCPs`` (the online MAP change-point
    path, ``ocp:::findCPprobs``). Returns the normalised vote score per point.
    """
    serie = np.asarray(serie, dtype=float)
    n = len(serie)
    hz = 1.0 / hazard

    # initOCPD: the single run length r=0 carries the prior.
    prev_r = np.array([1.0])
    mu_t = np.array([float(m)])
    kappa_t = np.array([float(k)])
    alpha_t = np.array([float(a)])
    beta_t = np.array([float(b)])
    probmaxes = [0.0, 0.0]
    logprobcps = [None, [1]]
    colmaxes = []
    threshcps = []
    maxcps = []

    for t in range(1, n + 1):           # 1-based point index, mirrors R ``t``
        x = serie[t - 1]
        # Truncate the run-length vector (drops a negligible, oldest tail).
        if len(prev_r) > min_rlength:
            tail = np.cumsum(prev_r[::-1])[::-1]   # rev(cumsum(rev(prev_r)))
            curr_r = prev_r[tail >= trunc_rlim]
            if len(curr_r) > max_rlength:
                curr_r = curr_r[:max_rlength]
            if len(curr_r) < min_rlength:
                curr_r = prev_r[:min_rlength]
        else:
            curr_r = prev_r
        rlen = len(curr_r)

        # Predictive probability of x for each run length (gaussianProb).
        mu_l, kap_l = mu_t[:rlen], kappa_t[:rlen]
        alp_l, bet_l = alpha_t[:rlen], beta_t[:rlen]
        predvar = bet_l * (kap_l + 1) / (alp_l * kap_l)
        nu = 2 * alp_l
        pred_probs = np.log(_bocpd_studentpdf(x, mu_l, predvar, nu))

        # Posterior update -> params of length rlen+1 (prior prepended).
        beta2 = bet_l + kap_l * (x - mu_l) ** 2 / (2 * (kap_l + 1))
        mu2 = (kap_l * mu_l + x) / (kap_l + 1)
        mu_t = np.concatenate([[m], mu2])
        beta_t = np.concatenate([[b], beta2])
        kappa_t = np.concatenate([[k], kap_l + 1])
        alpha_t = np.concatenate([[a], alp_l + 0.5])

        # Run-length recursion (growth vs changepoint).
        templogs = np.log(curr_r) + pred_probs
        templogs = np.where(templogs < _BOCPD_LOG_XMIN, _BOCPD_LOG_XMIN, templogs)
        temp2 = np.exp(templogs)
        grow_r = temp2 * (1 - hz)
        r0 = np.sum(temp2 * hz)
        new_r = np.concatenate([[r0], grow_r])
        new_r = new_r / np.sum(new_r)
        if np.any(~np.isfinite(np.log(new_r))):
            new_r = np.where(new_r < _BOCPD_XMIN, _BOCPD_XMIN, new_r)

        # colmaxes: changepoint implied by the most likely non-zero run length.
        tail_vals = new_r[1:]
        mx = np.max(tail_vals)
        p1 = int(np.argmax(new_r == mx)) + 1          # match(max, new_r), 1-based
        cp_col = t + 2 - p1
        if cp_col not in colmaxes:
            colmaxes.append(cp_col)

        # threshcps: run lengths whose posterior mass exceeds the threshold.
        for i0 in np.where(tail_vals > cpthreshold)[0]:
            cp = t + 1 - (i0 + 1)
            if cp not in threshcps:
                threshcps.append(cp)

        # maxCPs: online MAP change-point path (findCPprobs).
        prun = np.log(new_r)[::-1]
        pprev = np.array(probmaxes[t - rlen: t + 1])
        currprobs = prun + pprev
        maxprob = float(np.max(currprobs))
        maxstarttrunc = int(np.argmax(currprobs)) + 1     # 1-based
        lpc_trunc = logprobcps[len(logprobcps) - rlen - 1: len(logprobcps)]
        maxstart = (t - rlen) + maxstarttrunc
        prev_entry = lpc_trunc[maxstarttrunc - 1]
        base = [] if prev_entry is None else list(np.atleast_1d(prev_entry))
        currcpstrunc = base + [maxstart]
        probmaxes.append(maxprob)
        logprobcps = lpc_trunc + [currcpstrunc]
        maxcps = currcpstrunc

        prev_r = new_r

    # changepoint_lists voting (R ``hcp_bocpd`` detect): +1 per list, normalise.
    cp_prob = np.zeros(n)
    for lst in (colmaxes, threshcps, maxcps):
        cps = np.array([int(c) for c in lst], dtype=int)
        cps = cps[(cps > 1) & (cps <= n)]
        for c in cps:
            cp_prob[c - 1] += 1
    if cp_prob.max() > 0:
        cp_prob = cp_prob / cp_prob.max()
    return cp_prob


class HcpBocpd(Harbinger):
    """Bayesian Online Change Point Detection (Python port of R ``hcp_bocpd``).

    How it works / theory
    ---------------------
    BOCPD (Adams & MacKay, 2007) maintains, online, the posterior distribution
    over the *run length* -- the time elapsed since the last change point. At
    each step the run-length mass either **grows** (no change, weighted by
    ``1 - 1/hazard``) or **resets** to zero (a change, weighted by ``1/hazard``),
    with each hypothesis scored by its one-step predictive probability. Here the
    per-run-length predictive model is the conjugate **Normal-Inverse-Gamma**
    (unknown mean *and* variance), whose posterior predictive is a Student-t --
    the ``gaussianUPM`` default of the R ``ocp`` package, with prior
    ``(m, k, a, b) = (0, 0.01, 0.01, 1e-4)`` and a constant hazard ``1/hazard``.

    Change points are extracted exactly as ``ocp``/``hcp_bocpd`` do: three lists
    are accumulated over time -- ``colmaxes`` (the change point implied by the
    most likely non-zero run length each step), ``threshcps`` (run lengths whose
    posterior mass exceeds ``cpthreshold = 0.5``) and ``maxCPs`` (the online MAP
    change-point path). Each candidate position gets one vote per list it appears
    in; the votes are normalised by their maximum, the first ``burn_in`` points
    are zeroed, positions scoring ``>= threshold`` are flagged, and finally
    ``min_distance`` thins neighbouring detections.

    Parameters
    ----------
    hazard:
        Constant-hazard parameter; the prior change probability per step is
        ``1 / hazard`` (R default ``100``).
    threshold:
        Vote-score threshold for flagging a change point. ``None`` uses ``0.5``
        (R default): a position must appear in at least two of the three lists.
    min_distance:
        Minimum spacing (in samples) enforced between flagged change points.
    burn_in:
        Number of initial observations whose score is forced to zero.

    References
    ----------
    - Adams RP, MacKay DJC (2007). Bayesian Online Changepoint Detection.
      arXiv:0710.3742.
    - Pagotto A (2019). ocp: Bayesian Online Changepoint Detection. R package.

    Porting
    -------
    Mirrors ``hcp_bocpd.R``. Fidelity: adapted.
    - Native (pure numpy/scipy) back-implementation of the R ``ocp`` backend
      (``onlineCPD`` + ``findCPprobs``, gaussian UPM) -- ``ocp`` is only an R
      ``Suggests`` dependency and has no maintained Python port. Verified
      **EXACT** (identical detected indices) vs the real ``ocp`` backend on all
      seven ``examples_changepoints`` series, including the long
      ``increasing_amplitude`` (n=1002).
    - Only the gaussian UPM (the R default) is ported; the ``ocp`` ``poisson``
      model is out of scope.
    - The negligible run-length ``truncRlim`` tail truncation is replicated so
      the MAP path (``findCPprobs``) indexing matches ``ocp`` exactly.
    """

    def __init__(
        self,
        hazard: float = 100.0,
        threshold: float | None = None,
        min_distance: int = 5,
        burn_in: int = 5,
    ):
        super().__init__()
        self.hazard = hazard
        self.threshold = threshold
        self.min_distance = min_distance
        self.burn_in = burn_in

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        x = np.asarray(self.serie, dtype=float)
        n = len(x)
        if n == 0:
            raise ValueError("No non-missing observations were provided.")

        cp_prob = _bocpd_run(
            x,
            hazard=self.hazard,
            m=0.0, k=0.01, a=0.01, b=1e-4,
            cpthreshold=0.5,
            min_rlength=1, max_rlength=10**4, trunc_rlim=_BOCPD_XMIN,
        )

        if self.burn_in and self.burn_in > 0:
            cp_prob[: min(self.burn_in, n)] = 0.0

        threshold = 0.5 if self.threshold is None else self.threshold
        event = cp_prob >= threshold

        if event.sum() > 1 and self.min_distance > 0:
            idx = np.where(event)[0] + 1          # 1-based, mirrors R which(event)
            keep = np.zeros(len(idx), dtype=bool)
            last_kept = -np.inf
            for i, j in enumerate(idx):
                if (j - last_kept) >= self.min_distance:
                    keep[i] = True
                    last_kept = j
            event = np.zeros(n, dtype=bool)
            event[idx[keep] - 1] = True

        return self._restore_refs(change_points=event, res=cp_prob, threshold=threshold)
