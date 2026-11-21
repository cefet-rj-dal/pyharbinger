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

from .harbinger import Harbinger, as_series_array
from .utils import har_candidate_selection_highgroup, har_filter_boxplot, mas

__all__ = [
    "HcpAmoc",
    "HcpBinseg",
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


# --- hcp_chow / hcp_gft (structural change, native Bai-Perron) ---------------
#
# These are NATIVE re-implementations that replace R's dependency on the
# ``strucchange`` package (back-implementation initiative): a prefix-sum OLS
# residual-sum-of-squares, the Bai & Perron dynamic program for multiple breaks,
# and the Gaussian BIC selection reverse-engineered from ``strucchange``
# (``n*(log(2*pi)+1+log(RSS/n)) + log(n)*((m+1)*k + m + 1)``, k = 2).


def _rss_prefix(y):
    """Return ``rss(i, j)`` (O(1) OLS ``y~x`` RSS of the 1-based segment i..j) and n.

    Uses prefix sums of x, y, x^2, y^2, xy so the Bai-Perron DP is O(n^2).
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

    return rss, n


def _structural_bic(rss_m: float, n: int, m: int, k: int = 2) -> float:
    r = max(rss_m, 1e-300)
    return float(
        n * (np.log(2 * np.pi) + 1 + np.log(r / n)) + np.log(n) * ((m + 1) * k + m + 1)
    )


def _bai_perron_breakpoints(y, h_frac: float = 0.15) -> list[int]:
    """Bai & Perron optimal breakpoints with BIC-selected count (1-based)."""
    rss, n = _rss_prefix(y)
    h = int(np.floor(n * h_frac))
    if h < 1:
        return []
    max_breaks = n // h - 1
    if max_breaks < 1:
        return []

    opt = [np.full(n + 1, np.inf) for _ in range(max_breaks + 1)]
    prev = [np.full(n + 1, -1, dtype=int) for _ in range(max_breaks + 1)]
    for j in range(h, n + 1):
        opt[0][j] = rss(1, j)
    for m in range(1, max_breaks + 1):
        for j in range((m + 1) * h, n + 1):
            best = np.inf
            best_b = -1
            for b in range(m * h, j - h + 1):
                prev_rss = opt[m - 1][b]
                if not np.isfinite(prev_rss):
                    continue
                value = prev_rss + rss(b + 1, j)
                if value < best:
                    best = value
                    best_b = b
            opt[m][j] = best
            prev[m][j] = best_b

    best_m = 0
    best_bic = _structural_bic(opt[0][n], n, 0)
    for m in range(1, max_breaks + 1):
        if not np.isfinite(opt[m][n]):
            continue
        candidate = _structural_bic(opt[m][n], n, m)
        if candidate < best_bic:
            best_bic = candidate
            best_m = m

    cps: list[int] = []
    j, m = n, best_m
    while m > 0:
        b = int(prev[m][j])
        cps.append(b)
        j, m = b, m - 1
    return sorted(cps)


def _chow_breakpoint(y, h_frac: float = 0.15) -> int | None:
    """Single sup-F (Chow) breakpoint over the trimmed range (1-based) or None."""
    rss, n = _rss_prefix(y)
    h = int(np.floor(n * h_frac))
    k = 2
    if h < 1 or h > n - h or n - 2 * k <= 0:
        return None
    rss0 = rss(1, n)
    best_i = None
    best_f = -np.inf
    for i in range(h, n - h + 1):
        r = rss(1, i) + rss(i + 1, n)
        f = np.inf if r <= 0 else ((rss0 - r) / k) / (r / (n - 2 * k))
        if f > best_f:
            best_f = f
            best_i = i
    return best_i


class HcpChow(Harbinger):
    """Chow structural-break detector (native port of R ``hcp_chow``).

    Finds the single most significant structural break in the linear model
    ``y ~ t`` via the sup-F (Chow) statistic over a trimmed range. Native
    re-implementation that replaces R's ``strucchange`` dependency.

    Parameters
    ----------
    h:
        Minimal segment size as a fraction of the series length (trimming).
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
        bp = _chow_breakpoint(self.serie, self.h)
        if bp is not None:
            cp[bp - 1] = True
        return self._restore_refs(change_points=cp)


class HcpGft(Harbinger):
    """Generalized fluctuation / structural-change detector (port of R ``hcp_gft``).

    Finds multiple structural breaks in ``y ~ t`` by the Bai & Perron dynamic
    program, with the number of breaks chosen by BIC. Native re-implementation
    that replaces R's ``strucchange`` dependency.

    Parameters
    ----------
    h:
        Minimal segment size as a fraction of the series length.
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
        for bp in _bai_perron_breakpoints(self.serie, self.h):
            if 1 <= bp <= n:
                cp[bp - 1] = True
        return self._restore_refs(change_points=cp)
