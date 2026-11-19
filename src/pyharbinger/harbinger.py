"""Base class for the Harbinger event-detection framework.

Python port of the R ``harbinger.R`` contract. :class:`Harbinger` extends the
pydaltoolbox ``DalBase`` (the Experiment Line base) and defines the uniform
detection interface shared by every anomaly, change-point, and motif detector:

* ``fit(serie)`` -- train/adjust the detector (often a no-op); returns ``self``.
* ``detect(serie)`` -- return a detection table.
* ``evaluate(detection, event)`` -- score a detection against ground truth.

The detection table is a :class:`pandas.DataFrame` with columns:

* ``idx`` -- 0-based position in the original series.
* ``event`` -- boolean flag of a detected event.
* ``type`` -- ``"anomaly"``, ``"changepoint"``, ``"discord"``, ``"motif"`` or ``""``.

Concrete detectors reuse the internal helpers :meth:`Harbinger._store_refs` and
:meth:`Harbinger._restore_refs` to omit missing observations during computation
and restore detections to the original series indexing. The shared hooks
``har_distance`` / ``har_outliers`` / ``har_outliers_check`` default to the L2
deviation, the three-sigma Gaussian filter, and the first-of-group candidate
selection, exactly as in R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalBase

from .evaluation import HarEval
from .utils import (
    har_candidate_selection_firstgroup,
    har_deviation_l2,
    har_filter_gaussian,
)

__all__ = ["Harbinger", "as_series_array"]


def as_series_array(serie) -> np.ndarray:
    """Coerce a univariate series (Series / array / list / 1-col DataFrame) to float ndarray."""
    if isinstance(serie, pd.DataFrame):
        if serie.shape[1] != 1:
            raise ValueError("Only univariate series are supported in this version.")
        serie = serie.iloc[:, 0]
    if isinstance(serie, pd.Series):
        serie = serie.to_numpy()
    return np.asarray(serie, dtype=float).ravel()


class Harbinger(DalBase):
    """Base detector: shared state handling and the default ``detect`` contract."""

    def __init__(self):
        # Shared detection hooks (mirror R harbinger()).
        self.har_distance = har_deviation_l2
        self.har_outliers = har_filter_gaussian
        self.har_outliers_check = har_candidate_selection_firstgroup
        # State filled by _store_refs during detection.
        self.serie: np.ndarray | None = None
        self.non_na: np.ndarray | None = None
        self._n: int | None = None

    # --- contract -----------------------------------------------------------

    def fit(self, serie=None, **kwargs) -> Harbinger:
        """Train/adjust the detector. Default is a no-op returning ``self``."""
        return self

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        """Return a detection table. Base behaviour flags no events."""
        n = len(as_series_array(serie))
        return pd.DataFrame(
            {"idx": np.arange(n), "event": np.zeros(n, dtype=bool), "type": [""] * n}
        )

    def evaluate(self, detection, event, evaluation=None):
        """Evaluate a detection against ground-truth events (defaults to :class:`HarEval`)."""
        if evaluation is None:
            evaluation = HarEval()
        return evaluation.evaluate(detection, event)

    # --- shared helpers (har_store_refs / har_restore_refs) -----------------

    def _store_refs(self, serie) -> Harbinger:
        """Store the NA-omitted series and the positions of non-missing values.

        Univariate inputs are stored as a float ndarray (``self.serie``). A
        **multivariate** input (a :class:`pandas.DataFrame` with more than one
        column) is kept as a DataFrame so detectors can address columns by name;
        a row is "present" only when *all* its columns are non-missing, mirroring
        R's ``!is.na(apply(serie, 1, max))`` in ``har_store_refs``.
        """
        if isinstance(serie, pd.DataFrame) and serie.shape[1] > 1:
            frame = serie.reset_index(drop=True)
            present = ~frame.isna().any(axis=1).to_numpy()
            self._n = len(frame)
            self.non_na = np.where(present)[0]
            self.serie = frame.iloc[self.non_na].reset_index(drop=True)
        else:
            arr = as_series_array(serie)
            self._n = len(arr)
            self.non_na = np.where(~np.isnan(arr))[0]
            self.serie = arr[self.non_na]
        return self

    def _restore_refs(self, anomalies=None, change_points=None, res=None, threshold=None):
        """Map detections back to the original indexing and build the detection table."""
        n = self._n
        anomalies_full = np.full(n, np.nan)
        cps_full = np.full(n, np.nan)
        res_full = np.full(n, np.nan)

        startup = anomalies_full
        if change_points is not None:
            cps_full[self.non_na] = np.asarray(change_points, dtype=float)
            startup = cps_full
        if anomalies is not None:
            anomalies_full[self.non_na] = np.asarray(anomalies, dtype=float)
            startup = anomalies_full
        if res is not None:
            res_full[self.non_na] = np.asarray(res, dtype=float)

        event = np.where(np.isnan(startup), False, startup).astype(bool)
        type_arr = np.array([""] * n, dtype=object)

        anom_mask = np.where(np.isnan(anomalies_full), False, anomalies_full).astype(bool)
        cp_mask = np.where(np.isnan(cps_full), False, cps_full).astype(bool)

        type_arr[anom_mask] = "anomaly"
        event[cp_mask] = True
        type_arr[cp_mask] = "changepoint"

        detection = pd.DataFrame({"idx": np.arange(n), "event": event, "type": type_arr})
        detection.attrs["res"] = res_full
        detection.attrs["threshold"] = threshold
        return detection
