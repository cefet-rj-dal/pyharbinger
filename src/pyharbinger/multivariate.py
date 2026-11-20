"""Multivariate anomaly detector (Python port of the R ``hmu_pca``).

:class:`HmuPca` standardizes a multivariate series, projects it onto the leading
principal components, reconstructs it from a **reduced** set of components, and
flags rows with a large reconstruction error. It wraps
:class:`sklearn.decomposition.PCA`, so it adds **no new dependency**.

Keeping *all* components would reconstruct the data exactly, leaving only
floating-point round-off as the residual (a degenerate detector). By default the
detector therefore keeps the leading components that explain ``variance`` of the
total variance, capped at ``n_features - 1`` so at least one component is always
dropped and the residual is meaningful. This mirrors the fixed R ``hmu_pca``.
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
    num_components:
        Number of principal components to retain for reconstruction. When
        ``None`` (default) the number is chosen automatically from ``variance``.
    variance:
        Target cumulative proportion of variance used to pick ``num_components``
        automatically when it is ``None``.
    """

    def __init__(self, num_components: int | None = None, variance: float = 0.9):
        super().__init__()
        self.num_components = num_components
        self.variance = variance

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")

        self._store_refs(serie)
        data = self.serie
        matrix = (
            data.to_numpy(dtype=float)
            if isinstance(data, pd.DataFrame)
            else np.asarray(data, dtype=float)
        )
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        p = matrix.shape[1]

        # Standardize per column (z-score with ddof=1, like R's scale()).
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)
        scaled = (matrix - mean) / std

        # Full PCA, then choose how many leading components to retain.
        pca = PCA().fit(scaled)
        if self.num_components is None:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            reached = np.where(cumulative >= self.variance)[0]
            k = int(reached[0]) + 1 if len(reached) else p
        else:
            k = int(self.num_components)
        # Always drop at least one component so the residual is meaningful.
        k = max(1, min(k, p - 1)) if p > 1 else 1

        # Reconstruct from the retained components only; the discarded ones form
        # the reconstruction error used as the anomaly score.
        scores = pca.transform(scaled)[:, :k]
        reconstructed = scores @ pca.components_[:k] + pca.mean_

        res = self.har_distance(scaled - reconstructed)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, self.serie)

        return self._restore_refs(
            anomalies=selection.mask, res=res, threshold=selection.threshold
        )
