"""Anomaly detectors (Python port of the R ``hanr_*`` family).

This module provides the statistical / machine-learning anomaly detectors that
are thin wrappers over numpy / pandas / pytspredit:

* :class:`HanrMl` -- forecasting-based: wraps a ``pytspredit`` regressor, predicts
  one step ahead over sliding windows, and flags large residuals. This is the
  bridge that proves the pydaltoolbox -> pytspredit -> pyharbinger stack.
* :class:`HanrFft` -- high-pass FFT filtering: flags high-frequency departures.
* :class:`HanrHistogram` -- histogram density: flags low-density / out-of-range bins.
* :class:`HanrFbiad` -- Forward and Backward Inertial Anomaly Detector.

All of them reuse the shared :class:`~pyharbinger.harbinger.Harbinger` hooks
(``har_distance`` / ``har_outliers`` / ``har_outliers_check``).
"""
from __future__ import annotations

from math import log

import numpy as np
import ruptures as rpt
from pytspredit import ts_data, ts_projection

from .harbinger import Harbinger
from .utils import har_candidate_selection_highgroup, mas

__all__ = [
    "HanrArima",
    "HanrFbiad",
    "HanrFft",
    "HanrFftAmoc",
    "HanrFftAmocCusum",
    "HanrFftBinseg",
    "HanrFftBinsegCusum",
    "HanrFftSma",
    "HanrHistogram",
    "HanrMl",
]


# --- hanr_ml ----------------------------------------------------------------


class HanrMl(Harbinger):
    """Anomaly detector based on ML regression (wraps a pytspredit regressor).

    Parameters
    ----------
    model:
        A ``pytspredit`` regression model (e.g. ``TsElm``, ``TsKnn``, ``TsRf``).
    sw_size:
        Sliding-window size used to build the supervised one-step problem.
    """

    def __init__(self, model, sw_size: int = 15):
        super().__init__()
        self.model = model
        self.sw_size = sw_size

    def fit(self, serie, **kwargs) -> HanrMl:
        arr = _as_array(serie)
        ts = ts_data(arr, self.sw_size)
        io = ts_projection(ts)
        self.model = self.model.fit(x=io.input, y=io.output)
        return self

    def detect(self, serie, **kwargs):
        self._store_refs(serie)
        ts = ts_data(self.serie, self.sw_size)
        io = ts_projection(ts)

        adjust = np.asarray(self.model.predict(io.input)).ravel()
        values = np.asarray(io.output).ravel()
        res = values - adjust

        res = self.har_distance(res)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)

        res = np.concatenate([np.zeros(self.sw_size - 1), res])
        anomalies = np.concatenate([np.zeros(self.sw_size - 1, dtype=bool), selection.mask])

        return self._restore_refs(anomalies=anomalies, res=res, threshold=selection.threshold)


# --- hanr_fft ---------------------------------------------------------------


def _compute_cut_index(freqs: np.ndarray) -> int:
    """High-pass cutoff index (1-based, as in R ``compute_cut_index``)."""
    cutindex = int(np.argmax(freqs)) + 1
    if freqs.min() != freqs.max():
        threshold = freqs.mean() + 2.698 * np.std(freqs, ddof=1)
        f = freqs.copy()
        f[f < threshold] = freqs.min() + freqs.max()
        cutindex = int(np.argmin(f)) + 1
    return cutindex


class HanrFft(Harbinger):
    """Anomaly detector using high-pass FFT filtering."""

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)

        fft_signal = np.fft.fft(self.serie)
        spectrum = np.abs(fft_signal) ** 2
        half_spectrum = spectrum[: len(self.serie) // 2 + 1]

        cutindex = _compute_cut_index(half_spectrum)
        n = len(fft_signal)

        # Zero out low frequencies (high-pass), both symmetric halves.
        fft_signal[:cutindex] = 0
        fft_signal[n - cutindex - 1 :] = 0

        filtered_series = np.real(np.fft.ifft(fft_signal))

        res = self.har_distance(filtered_series)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, self.serie)

        return self._restore_refs(anomalies=selection.mask, res=res, threshold=selection.threshold)


# --- hanr_fft_* (spectral change-point cutoff) ------------------------------


def _spectrum_cut_index(freqs: np.ndarray, binseg: bool) -> int:
    """Cutoff index from a change point in the power spectrum (R ``compute_cut_index``).

    R uses ``changepoint::cpt.mean`` (change in mean) with method AMOC or BinSeg
    and returns the *last* detected change point. Here ``ruptures`` reproduces it
    with the L2 (mean-shift) cost: AMOC = a single breakpoint; BinSeg = a
    ``3*log(n)`` penalty (matching :class:`HcpPelt`'s calibration and R's cpts).
    """
    k = len(freqs)
    signal = np.asarray(freqs, dtype=float)
    if binseg:
        bkps = rpt.Binseg(model="l2", min_size=1, jump=1).fit(signal).predict(pen=3.0 * log(k))
    else:
        bkps = rpt.Dynp(model="l2", min_size=1, jump=1).fit(signal).predict(n_bkps=1)
    cpts = [b for b in bkps if b < k]
    return cpts[-1] if cpts else 1


class _HanrFftCut(Harbinger):
    """Shared machinery for the FFT high-pass detectors with a spectral cutoff.

    Subclasses set :attr:`_binseg` (AMOC vs. BinSeg change point) and
    :attr:`_cusum` (whether the cutoff is found on the cumulative spectrum).
    """

    _binseg = False
    _cusum = False

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)

        fft_signal = np.fft.fft(self.serie)
        spectrum = np.abs(fft_signal) ** 2
        half_spectrum = spectrum[: len(self.serie) // 2 + 1]

        freqs = np.cumsum(half_spectrum) if self._cusum else half_spectrum
        cutindex = _spectrum_cut_index(freqs, self._binseg)
        n = len(fft_signal)

        # Zero out low frequencies (high-pass), both symmetric halves.
        fft_signal[:cutindex] = 0
        fft_signal[n - cutindex - 1 :] = 0

        filtered_series = np.real(np.fft.ifft(fft_signal))

        res = self.har_distance(filtered_series)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, self.serie)

        return self._restore_refs(anomalies=selection.mask, res=res, threshold=selection.threshold)


class HanrFftAmoc(_HanrFftCut):
    """FFT high-pass detector; cutoff = AMOC change point of the power spectrum."""

    _binseg = False
    _cusum = False


class HanrFftBinseg(_HanrFftCut):
    """FFT high-pass detector; cutoff = last BinSeg change point of the power spectrum."""

    _binseg = True
    _cusum = False


class HanrFftAmocCusum(_HanrFftCut):
    """FFT high-pass detector; cutoff = AMOC change point of the cumulative spectrum."""

    _binseg = False
    _cusum = True


class HanrFftBinsegCusum(_HanrFftCut):
    """FFT high-pass detector; cutoff = last BinSeg change point of the cumulative spectrum.

    Uses the high-group candidate selection (as in R ``hanr_fft_binseg_cusum``).
    """

    _binseg = True
    _cusum = True

    def __init__(self):
        super().__init__()
        self.har_outliers_check = har_candidate_selection_highgroup


# --- hanr_fft_sma (adaptive moving-average residual) ------------------------


def _nextn(n: int, factors: tuple[int, ...] = (2, 3, 5)) -> int:
    """Smallest highly composite integer ``>= n`` (R ``stats::nextn``).

    ``spec.pgram(..., fast = TRUE)`` zero-pads the series to this length so the
    FFT runs on a product of the given small primes.
    """
    if n <= 1:
        return 1
    m = n
    while True:
        k = m
        for f in factors:
            while k % f == 0:
                k //= f
        if k == 1:
            return m
        m += 1


def _spec_taper(x: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Split cosine-bell taper over a fraction ``p`` of each end (R ``spec.taper``)."""
    x = x.copy()
    n = x.size
    m = int(np.floor(n * p))
    if m == 0:
        return x
    w = 0.5 * (1 - np.cos(np.pi * np.arange(1, 2 * m, 2) / (2 * m)))
    taper = np.concatenate([w, np.ones(n - 2 * m), w[::-1]])
    return x * taper


def _fft_sma_window(series: np.ndarray) -> int:
    """Best moving-average window = period of the dominant spectral frequency.

    Faithful port of ``find_best_moving_average`` in R ``hanr_fft_sma``: run the
    raw periodogram (``stats::spec.pgram`` defaults -- linear detrend, 10% split
    cosine-bell taper, zero-pad to ``nextn``), take the frequency of the largest
    spectral peak, convert it to an integer period ``round(1 / freq)``, and clamp
    to ``[3, len(series) - 1]``. Only the argmax matters, so the global spectral
    scaling constants of ``spec.pgram`` are dropped.
    """
    x = np.asarray(series, dtype=float)
    n0 = x.size
    n = n0
    # Linear detrend (spec.pgram default detrend = TRUE).
    t = np.arange(1, n + 1) - (n + 1) / 2.0
    sumt2 = n * (n**2 - 1) / 12.0
    x = x - x.mean() - (np.sum(x * t) / sumt2) * t
    # Taper, then zero-pad to the next highly composite length.
    x = _spec_taper(x, 0.1)
    new_n = _nextn(n)
    if new_n > n:
        x = np.concatenate([x, np.zeros(new_n - n)])
    n = x.size
    nspec = n // 2
    fft = np.fft.fft(x)
    spectrum = (fft * np.conj(fft)).real / n0
    # freq[k] = k / n for k = 1 .. nspec (the zero-frequency bin is dropped).
    kmax = int(np.argmax(spectrum[1 : nspec + 1])) + 1
    best = round(n / kmax)
    return min(max(3, best), n0 - 1)


class HanrFftSma(Harbinger):
    """Anomaly detector using an adaptive FFT-selected moving average.

    Theory:
        The periodogram reveals the series' dominant periodicity; a simple moving
        average of that period smooths the recurring component away. What remains
        -- the trailing-window residual ``serie[t] - mean(serie[t-w+1 : t])`` --
        isolates the departures from the local trend, which are scored as
        anomalies with the shared Harbinger outlier heuristics. The first ``w - 1``
        points are a warm-up window and never flagged.

    References:
        - Sobrinho EP, Souza J, Lima J, Giusti L, Bezerra E, Coutinho R, Baroni L,
          Pacitti E, Porto F, Belloze K, Ogasawara E (2025). Fine-Tuning Detection
          Criteria for Enhancing Anomaly Detection in Time Series. SBBD.
          doi:10.5753/sbbd.2025.247063

    Porting:
        Mirrors ``hanr_fft_sma.R``. Fidelity: exact. The window is picked by a
        faithful port of ``stats::spec.pgram`` (:func:`_fft_sma_window`); the
        residual alignment follows the R recycling-bug fix (``serie[w:n] - sma``
        with the first ``w - 1`` points as warm-up).
    """

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        values = self.serie
        n = len(values)

        window = _fft_sma_window(values)
        sma = mas(values, window)  # mean of each length-``window`` trailing window

        # Align the moving average to each observation: sma[i] is the mean of the
        # window ending at position i + window - 1, so it lines up with
        # values[window - 1:]. The first window - 1 points are a warm-up.
        filtered = np.zeros(n)
        filtered[window - 1 :] = values[window - 1 :] - sma

        res = self.har_distance(filtered)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)

        mask = np.asarray(selection.mask, dtype=bool).copy()
        mask[: window - 1] = False  # R: anomalies[1:(optimal_window - 1)] <- FALSE

        return self._restore_refs(anomalies=mask, res=res, threshold=selection.threshold)


# --- hanr_histogram ---------------------------------------------------------


def _nclass_sturges(n: int) -> int:
    """R ``nclass.Sturges``: ceil(log2(n) + 1)."""
    return int(np.ceil(np.log2(n) + 1))


def r_pretty(lo: float, hi: float, n: int) -> np.ndarray:
    """Equally spaced 'nice' breakpoints covering [lo, hi], like R ``pretty()``.

    A compact port of the common path of R's ``pretty`` / ``hist`` break
    computation: pick a step among {1, 2, 2.5, 5} * 10^k that yields about ``n``
    intervals, then extend the range outward to multiples of that step.
    """
    if hi == lo:
        hi = lo + 1
    dx = hi - lo
    raw_step = dx / max(n, 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    # R favors these nice multiples (with a mild bias toward larger steps).
    if norm < 1.5:
        nice = 1.0
    elif norm < 3:
        nice = 2.0
    elif norm < 7:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * mag
    start = np.floor(lo / step) * step
    end = np.ceil(hi / step) * step
    # Guard floating point so the last break covers hi.
    n_steps = int(np.round((end - start) / step))
    return start + step * np.arange(n_steps + 1)


class HanrHistogram(Harbinger):
    """Anomaly detector using histogram bin density.

    Parameters
    ----------
    density_threshold:
        Minimum bin density for an observation not to be flagged (default 0.05).
    """

    def __init__(self, density_threshold: float = 0.05):
        super().__init__()
        self.density_threshold = density_threshold

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        values = self.serie
        n = len(values)

        bin_edges = r_pretty(float(values.min()), float(values.max()), _nclass_sturges(n))
        counts, _ = np.histogram(values, bins=bin_edges)
        widths = np.diff(bin_edges)
        density = counts / (n * widths)
        n_bins = len(density)

        anomalies = np.zeros(n, dtype=bool)
        for i in range(n):
            # findInterval(..., all.inside=TRUE) clamped to [1, n_bins] (1-based).
            bin_index = int(np.searchsorted(bin_edges, values[i], side="right"))
            bin_index = min(max(bin_index, 1), n_bins)
            lower = bin_edges[bin_index - 1]
            upper = bin_edges[bin_index]
            if values[i] < lower or values[i] > upper or density[bin_index - 1] < self.density_threshold:
                anomalies[i] = True

        return self._restore_refs(anomalies=anomalies)


# --- hanr_fbiad -------------------------------------------------------------


class HanrFbiad(Harbinger):
    """Forward and Backward Inertial Anomaly Detector.

    Parameters
    ----------
    sw_size:
        Window size for the forward/backward inertial context.
    """

    def __init__(self, sw_size: int = 30):
        super().__init__()
        self.sw_size = sw_size

    def _direction(self, serie: np.ndarray):
        sx = np.asarray(ts_data(serie, self.sw_size))
        ma = sx.mean(axis=1)
        values = sx[:, -1]
        res = self.har_distance(sx[:, -1] - ma)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)
        return selection.mask, res

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        pad = self.sw_size - 1

        i_f, res_f = self._direction(self.serie)
        i_f = np.concatenate([np.zeros(pad, dtype=bool), i_f])
        res_f = np.concatenate([np.zeros(pad), res_f])

        i_b, res_b = self._direction(self.serie[::-1])
        i_b = np.concatenate([i_b[::-1], np.zeros(pad, dtype=bool)])
        res_b = np.concatenate([res_b[::-1], np.zeros(pad)])

        res = (res_b + res_f) / 2
        anomalies = i_f | i_b

        return self._restore_refs(anomalies=anomalies, res=res)


# --- hanr_arima (ARIMA++ anomaly detector) ----------------------------------


class HanrArima(Harbinger):
    """Anomaly detector based on ARIMA residuals (port of R ``hanr_arima``, ``++``).

    Theory:
        An ARIMA(p, d, q) model captures the normal autoregressive / moving-average
        dynamics of a series. Once fitted, observations whose in-sample residual is
        unusually large are flagged as anomalies: the model predicts what the series
        "should" do next, and a point it cannot explain is a candidate event.
        Residual magnitudes are turned into a distance and thresholded with the
        shared Harbinger outlier heuristics; the first ``sw_size`` points are a
        warm-up window and never flagged.

    "++" adaptation:
        The original R ``hanr_arima`` picks the order with ``forecast::auto.arima``
        (stepwise AICc). This releitura reuses the DAL ARIMA auto-selector
        :func:`pytspredit.fit_arima` -- in particular its ``auto="cv"`` *ARIMA++*
        mode, which draws the ``(p, q)`` candidates from the significant ACF/PACF
        lags (Box-Jenkins identification, not a fixed cap of 5) and ranks them by
        rolling-origin cross-validated forecast error rather than the in-sample AICc
        proxy. Reusing the DAL ARIMA removes the dependency on R's ``forecast``
        package (its last ``harbinger`` user), so the R backport can drop it.

    References:
        - Box GEP, Jenkins GM, Reinsel GC, Ljung GM (2015). Time Series Analysis:
          Forecasting and Control. Wiley.
        - Hyndman RJ, Khandakar Y (2008). Automatic time series forecasting: the
          forecast package for R. Journal of Statistical Software, 27(3).
        - Tashman LJ (2000). Out-of-sample tests of forecasting accuracy: an
          analysis and review. International Journal of Forecasting, 16(4)
          (rolling-origin evaluation).

    Porting:
        Mirrors ``hanr_arima.R``. Fidelity: adapted (++). Order selection via
        ``pytspredit.fit_arima`` (the DAL ARIMA) instead of ``forecast::auto.arima``;
        the detection path (in-sample residuals -> ``har_distance`` -> ``har_outliers``
        -> warm-up mask of ``sw_size = max(p, d + 1, q)``) follows the R original.

    Parameters
    ----------
    auto:
        Order-selection mode forwarded to :func:`pytspredit.fit_arima`: ``"aicc"``
        (compact AICc search, the default -- fast and already beats the R detector)
        or ``"cv"`` (the heavier ARIMA++ rolling-origin cross-validated selector).
    max_order, cv_folds, horizon:
        Forwarded to :func:`pytspredit.fit_arima` when ``auto="cv"``.
    """

    def __init__(
        self,
        auto: str = "aicc",
        max_order: int = 12,
        cv_folds: int = 2,
        horizon=None,
    ):
        super().__init__()
        self.auto = auto
        self.max_order = max_order
        self.cv_folds = cv_folds
        self.horizon = horizon
        self.p = self.d = self.q = 0
        self.sw_size = 1
        self._model = None

    def fit(self, serie, **kwargs) -> HanrArima:
        from pytspredit import fit_arima

        arr = _as_array(serie)
        (self.p, self.d, self.q), _trend, self._model = fit_arima(
            arr,
            auto=self.auto,
            max_order=self.max_order,
            cv_folds=self.cv_folds,
            horizon=self.horizon,
        )
        self.sw_size = max(self.p, self.d + 1, self.q)
        return self

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        values = np.asarray(self.serie, dtype=float)
        self.fit(values)

        resid = np.asarray(self._model.resid, dtype=float).ravel()
        # statsmodels returns one residual per point; guard against any length drift.
        if len(resid) != len(values):
            pad = max(len(values) - len(resid), 0)
            resid = np.concatenate([np.zeros(pad), resid])[: len(values)]

        # Neutralise the warm-up window *before* the outlier statistics. Unlike R's
        # ``forecast`` Kalman residuals, statsmodels' in-sample residuals carry a
        # large diffuse-initialisation transient in the first few points; left in,
        # it dominates the threshold and hides the real anomalies. Zeroing the first
        # ``sw_size`` residuals (the window R flags as warm-up anyway) removes the
        # transient from both the statistics and the final detection.
        warm = min(self.sw_size, len(resid))
        resid[:warm] = 0.0

        res = self.har_distance(resid)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)

        mask = np.asarray(selection.mask, dtype=bool).copy()
        mask[:warm] = False  # R: anomalies[1:sw_size] <- FALSE

        return self._restore_refs(anomalies=mask, res=res, threshold=selection.threshold)


def _as_array(serie) -> np.ndarray:
    from .harbinger import as_series_array

    return as_series_array(serie)
