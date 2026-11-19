"""Classification-based anomaly detector (Python port of the R ``hanc_ml``).

:class:`HancMl` is **supervised**: it wraps a pydaltoolbox classification model
(e.g. :class:`pydaltoolbox.ClaKnn`, ``ClaRf``, ``ClaSvm``, ``ClaDtree``) trained
on labeled observations. At detection time the positive-class probability is
thresholded and consecutive candidates are collapsed by the candidate-selection
hook (first-of-group by default), exactly as in R's ``hanc_ml``.

Because the whole classifier family wraps scikit-learn, this detector adds **no
new dependency** beyond the ones already required by ``pyharbinger``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .harbinger import Harbinger
from .utils import FilterResult

__all__ = ["HancMl"]


class HancMl(Harbinger):
    """Supervised anomaly detector wrapping a pydaltoolbox classifier.

    Parameters
    ----------
    model:
        A configured pydaltoolbox classification model whose ``attribute`` names
        the target column, e.g. ``ClaKnn("event", ["FALSE", "TRUE"])``. The
        feature columns (``x``) are learned at ``fit`` time. The second class
        level is treated as the positive (event) class, matching R's ``adjust[, 2]``.
    threshold:
        Positive-class probability at/above which an observation is a candidate
        event.
    """

    def __init__(self, model, threshold: float = 0.5):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def fit(self, serie=None, **kwargs) -> HancMl:
        """Train the underlying classifier on a labeled frame (features + target)."""
        if serie is None:
            raise ValueError("No data was provided for computation")
        self.model = self.model.fit(serie)
        return self

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")

        # Normalize indexing / omit NA rows (multivariate-aware).
        self._store_refs(serie)
        frame = self.serie
        if not isinstance(frame, pd.DataFrame):
            name = self.model.x[0] if self.model.x else "serie"
            frame = pd.DataFrame({name: np.asarray(frame, dtype=float)})

        # Positive-class probability (second level, mirroring R's adjust[, 2]).
        proba = self.model.predict(frame[self.model.x])
        res = np.asarray(proba.iloc[:, 1], dtype=float)

        # Candidate = probability at/above threshold; then candidate selection.
        index = np.where(res >= self.threshold)[0]
        outliers = FilterResult(index, (self.threshold, self.threshold))
        selection = self.har_outliers_check(outliers, res, res)

        return self._restore_refs(
            anomalies=selection.mask, res=res, threshold=self.threshold
        )
