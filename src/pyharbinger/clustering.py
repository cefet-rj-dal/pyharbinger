"""Clustering-based anomaly/discord detector (Python port of ``hanct_kmeans``).

:class:`HanctKmeans` clusters the sliding-window subsequences of a series with
k-means and flags the ones that are far from their nearest centroid. With
``seq == 1`` it labels isolated observations as anomalies; with ``seq > 1`` it
labels whole subsequences as discords.

Because scikit-learn's k-means uses a different initialization and RNG than R's
``stats::kmeans``, results are only approximately comparable to R; set
``random_state`` for reproducibility.
"""
from __future__ import annotations

import numpy as np
from pytspredit import ts_data
from sklearn.cluster import KMeans

from .harbinger import Harbinger, as_series_array

__all__ = ["HanctKmeans"]


class HanctKmeans(Harbinger):
    """k-means distance-based anomaly (``seq == 1``) / discord (``seq > 1``) detector.

    Parameters
    ----------
    seq:
        Subsequence (sliding-window) size.
    centers:
        Number of centroids. When ``None`` it defaults to ``ceil(log10(n))``.
    random_state:
        Seed for the k-means initialization.
    """

    def __init__(self, seq: int = 1, centers: int | None = None, random_state: int | None = None):
        super().__init__()
        self.seq = seq
        self.centers = centers
        self.random_state = random_state
        self.centroids: np.ndarray | None = None

    def fit(self, serie, **kwargs) -> HanctKmeans:
        arr = as_series_array(serie)
        arr = arr[~np.isnan(arr)]
        centers = self.centers
        if centers is None:
            centers = int(np.ceil(np.log10(len(arr))))
        data = np.asarray(ts_data(arr, self.seq))
        km = KMeans(n_clusters=centers, n_init=1, random_state=self.random_state).fit(data)
        self.centers = centers
        self.centroids = km.cluster_centers_
        return self

    def detect(self, serie, **kwargs):
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)

        sx = np.asarray(ts_data(self.serie, self.seq))
        values = sx[:, -1]

        # Distance from each window to its nearest centroid.
        d2 = ((sx[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        res = np.sqrt(d2.min(axis=1))

        res = self.har_distance(res)
        outliers = self.har_outliers(res)
        selection = self.har_outliers_check(outliers, res, values)
        threshold = selection.threshold

        pad = self.seq - 1
        res = np.concatenate([np.zeros(pad), res])
        anomalies = np.concatenate([np.zeros(pad, dtype=bool), selection.mask])

        detection = self._restore_refs(anomalies=anomalies, res=res, threshold=threshold)

        if self.seq != 1:
            mask = detection["type"] == "anomaly"
            detection.loc[mask, "type"] = "discord"
            detection["seq"] = np.where(mask, self.seq, np.nan)
            detection["seqlen"] = np.where(mask, self.seq, np.nan)

        return detection
