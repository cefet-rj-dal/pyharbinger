"""Decomposition-based anomaly detectors (wavelet / EMD).

Python port of the R harbinger decomposition detectors. Instead of the external
R packages (``wavelets``, ``hht``), these reuse the **native** multiresolution
and empirical-mode decompositions already implemented in ``pytspredit`` -- so no
heavy third-party dependency is pulled in (mirrors how :class:`HanrArima`
reuses ``pytspredit.fit_arima``).

Currently provides :class:`HanrWavelet`; the EMD family follows.

References
----------
- Mallat S (1989). A theory for multiresolution signal decomposition: the
  wavelet representation. IEEE TPAMI 11(7):674-693.

Porting
-------
Mirrors ``hanr_wavelet.R``. Fidelity: exact (``pytspredit.wavelet.modwt``
reproduces R ``wavelets::modwt``; identical detected sets on
``examples_anomalies``).
"""
from __future__ import annotations

import math

import numpy as np
from pytspredit import wavelet as _wavelet

from .harbinger import Harbinger

__all__ = ["HanrWavelet"]


class HanrWavelet(Harbinger):
    """Anomaly detector using a wavelet (MODWT) multiresolution decomposition.

    The series is decomposed with the MODWT and the detail bands are summed into
    a magnitude signal; large deviations of that signal are flagged. Reuses the
    native ``pytspredit`` MODWT (no external wavelet dependency).

    Parameters
    ----------
    filter:
        Wavelet filter: ``"haar"`` (default), ``"d4"``, ``"la8"``, ``"bl14"``,
        or ``"c6"``.
    n_levels:
        Number of decomposition levels. ``None`` uses ``floor(log2(n))``,
        matching R ``wavelets::modwt``.

    Porting
    -------
    Mirrors ``hanr_wavelet.R``. Fidelity: exact.
    """

    def __init__(self, filter: str = "haar", n_levels: int | None = None):
        super().__init__()
        if filter not in _wavelet.modwt_filters():
            raise ValueError(f"filter must be one of {_wavelet.modwt_filters()}")
        self.filter = filter
        self.n_levels = n_levels

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)
        values = self.serie

        levels = self.n_levels if self.n_levels is not None else math.floor(math.log2(len(values)))
        details, _smooth = _wavelet.modwt(values, self.filter, levels)
        w_component = np.sum(np.column_stack(details), axis=1)

        res = self.har_distance(w_component)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)

        return self._restore_refs(anomalies=selection.mask, res=res, threshold=selection.threshold)
