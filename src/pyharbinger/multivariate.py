"""Multivariate anomaly detector (Python port of the R ``hmu_pca``).

:class:`HmuPca` standardizes a multivariate series, projects it onto principal
components, reconstructs it, and flags rows with a large reconstruction error.
It wraps :class:`sklearn.decomposition.PCA`, so it adds **no new dependency**.

.. note::
   This is a faithful port of R's ``hmu_pca``, which keeps **all** principal
   components. With all components retained the reconstruction is exact, so the
   residual is reduced to floating-point round-off and the detector is
   effectively degenerate (it flags round-off artefacts). The ``n_components``
   parameter below is a **pyharbinger extension** (not present in R): keeping
   fewer components than the input dimensionality turns it into a genuine
   reconstruction-error detector. With ``n_components=None`` the R behaviour is
   reproduced exactly (degenerate; residuals ~0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .harbinger import Harbinger

__all__ = ["HmuPca"]


class HmuPca(Harbinger):
    """PCA reconstruction-error anomaly detector for multivariate series.

    Parameters
    ----------
    n_components:
        Number of principal components to keep. ``None`` (default) keeps all
        components, faithfully reproducing R's ``hmu_pca`` (degenerate; residuals
        collapse to round-off). An integer ``< n_features`` yields a genuine
        reconstruction-error detector (a pyharbinger extension over R).
    """

    def __init__(self, n_components: int | None = None):
        super().__init__()
        self.n_components = n_components

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")

        self._store_refs(serie)
        data = self.serie
        matrix = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(
            data, dtype=float
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        # Standardize per column (z-score with ddof=1, like R's scale()).
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)
        scaled = (matrix - mean) / std

        # PCA projection + reconstruction; residual = scaled - reconstructed.
        n_comp = self.n_components if self.n_components is not None else min(scaled.shape)
        pca = PCA(n_components=n_comp)
        reconstructed = pca.inverse_transform(pca.fit_transform(scaled))

        res = self.har_distance(scaled - reconstructed)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, self.serie)

        return self._restore_refs(
            anomalies=selection.mask, res=res, threshold=selection.threshold
        )
