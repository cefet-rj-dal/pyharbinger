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

import numpy as np
from pytspredit import ts_data, ts_projection

from .harbinger import Harbinger

__all__ = ["HanrFbiad", "HanrFft", "HanrHistogram", "HanrMl"]


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


def _as_array(serie) -> np.ndarray:
    from .harbinger import as_series_array

    return as_series_array(serie)
