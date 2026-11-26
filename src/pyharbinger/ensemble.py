"""Ensemble detectors for Harbinger: majority vote and fuzzy consensus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .harbinger import Harbinger, as_series_array

__all__ = ["HarEnsemble", "HarEnsembleFuzzy"]


class HarEnsemble(Harbinger):
    """Majority-vote ensemble across several Harbinger detectors.

    Each member detector votes per observation; a position is flagged when more
    than half of the members agree. The event type is ``"changepoint"`` when any
    voting member reports one, otherwise ``"anomaly"``.

    Theory
    ------
    For ``m`` detectors the vote count ``res`` is the row-wise sum of the member
    event masks, and an event fires where ``res > m / 2``.

    References
    ----------
    - Ogasawara E, Salles R, Porto F, Pacitti E (2025). Event Detection in Time
      Series. Springer. doi:10.1007/978-3-031-75941-3

    Porting
    -------
    Mirrors ``har_ensemble.R``. Fidelity: exact (same detected sets when the
    member detectors match R).
    """

    def __init__(self, *models: Harbinger):
        super().__init__()
        if not models:
            raise ValueError("At least one detector is required.")
        self.models = list(models)
        self.har_outliers_check = None

    def fit(self, serie, **kwargs) -> HarEnsemble:
        if serie is None:
            raise ValueError("No data was provided for computation")
        clean = as_series_array(serie)
        clean = clean[~np.isnan(clean)]
        self.models = [model.fit(clean, **kwargs) for model in self.models]
        return self

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)

        vote_cols = []
        type_cols = []
        for model in self.models:
            detection = model.detect(self.serie, **kwargs)
            evt = detection["event"].fillna(False).to_numpy(dtype=bool)
            typ = detection["type"].fillna("").to_numpy(dtype=object)
            vote_cols.append(evt.astype(float))
            type_cols.append(typ)

        res = np.sum(np.column_stack(vote_cols), axis=1)
        events = res > len(self.models) / 2

        type_matrix = np.column_stack(type_cols)
        consensus = np.full(len(res), "", dtype=object)
        for i, row in enumerate(type_matrix):
            present = row[row != ""]
            if "changepoint" in present:
                consensus[i] = "changepoint"
            elif present.size:
                consensus[i] = "anomaly"
        consensus[~events] = ""

        anomalies = (consensus == "anomaly") & events
        change_points = (consensus == "changepoint") & events

        detection = self._restore_refs(
            anomalies=anomalies, change_points=change_points, res=res
        )
        detection.attrs["score"] = detection.attrs["res"]
        return detection


def _fuzzify_triangle(event: np.ndarray, event_type: np.ndarray, tolerance: int):
    """Expand binary detections with the triangular membership used by Harbinger R."""
    fuzzy = event.astype(float, copy=True)
    fuzzy_type = event_type.astype(object, copy=True)
    if tolerance <= 1:
        return fuzzy, fuzzy_type
    radius = int(tolerance)
    for center in np.flatnonzero(event):
        for distance in range(1, radius):
            weight = (radius - distance) / radius
            for position in (center - distance, center + distance):
                if 0 <= position < len(fuzzy) and weight > fuzzy[position]:
                    fuzzy[position] = weight
                    fuzzy_type[position] = event_type[center]
    return fuzzy, fuzzy_type


def _highest_in_each_run(candidates: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Retain the highest-scoring position in each consecutive candidate run."""
    result = np.zeros(len(candidates), dtype=bool)
    positions = np.flatnonzero(candidates)
    if positions.size == 0:
        return result
    for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
        run_score = score[run]
        result[run[np.flatnonzero(run_score == run_score.max())[0]]] = True
    return result


class HarEnsembleFuzzy(Harbinger):
    """Combine detector events into a normalized fuzzy consensus score.

    With ``threshold_type="percentile"``, the effective threshold is the
    maximum of the positive-score percentile and ``min_consensus``. This
    absolute floor permits an empty detection when consensus is weak.
    """

    def __init__(
        self,
        *models: Harbinger,
        names=None,
        threshold: float = 0.5,
        threshold_type: str = "fixed",
        min_consensus: float = 0.5,
        time_tolerance: int = 0,
        use_nms: bool = False,
    ):
        super().__init__()
        if not models:
            raise ValueError("At least one detector is required.")
        self.models = list(models)
        self.names = list(names) if names is not None else [f"model_{i + 1}" for i in range(len(models))]
        if len(self.names) != len(self.models):
            raise ValueError("names must have the same length as models.")
        self.threshold = self._unit_interval(threshold, "threshold")
        self.threshold_type = self._threshold_type(threshold_type)
        self.min_consensus = self._unit_interval(min_consensus, "min_consensus")
        self.time_tolerance = self._nonnegative_integer(time_tolerance, "time_tolerance")
        self.use_nms = bool(use_nms)

    @staticmethod
    def _unit_interval(value, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

    @staticmethod
    def _threshold_type(value: str) -> str:
        if value not in {"fixed", "percentile"}:
            raise ValueError("threshold_type must be 'fixed' or 'percentile'.")
        return value

    @staticmethod
    def _nonnegative_integer(value, name: str) -> int:
        integer = int(value)
        if integer != value or integer < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
        return integer

    def fit(self, serie, **kwargs):
        clean = as_series_array(serie)
        clean = clean[np.isfinite(clean)]
        self.models = [model.fit(clean, **kwargs) for model in self.models]
        return self

    def detect(
        self,
        serie,
        *,
        threshold=None,
        threshold_type=None,
        min_consensus=None,
        time_tolerance=None,
        use_nms=None,
        **kwargs,
    ) -> pd.DataFrame:
        threshold = self._unit_interval(self.threshold if threshold is None else threshold, "threshold")
        threshold_type = self._threshold_type(self.threshold_type if threshold_type is None else threshold_type)
        min_consensus = self._unit_interval(
            self.min_consensus if min_consensus is None else min_consensus, "min_consensus"
        )
        time_tolerance = self._nonnegative_integer(
            self.time_tolerance if time_tolerance is None else time_tolerance, "time_tolerance"
        )
        use_nms = self.use_nms if use_nms is None else bool(use_nms)

        self._store_refs(serie)
        score_columns = []
        type_columns = []
        model_events = {}
        for name, model in zip(self.names, self.models):
            detection = model.detect(self.serie, **kwargs)
            if len(detection) != len(self.serie):
                raise ValueError(f"Detector {name!r} returned an incompatible result length.")
            event = detection["event"].fillna(False).to_numpy(dtype=bool)
            event_type = (
                detection["type"].fillna("").to_numpy(dtype=object)
                if "type" in detection
                else np.full(len(event), "", dtype=object)
            )
            event_type[event & (event_type == "")] = "anomaly"
            event_type[~event] = ""
            fuzzy, fuzzy_type = _fuzzify_triangle(event, event_type, time_tolerance)
            score_columns.append(fuzzy)
            type_columns.append(fuzzy_type)
            model_events[name] = {"raw": event, "type": event_type}

        score = np.mean(np.column_stack(score_columns), axis=1).clip(0, 1)
        type_matrix = np.column_stack(type_columns)
        consensus_type = np.full(len(score), "", dtype=object)
        for index, row in enumerate(type_matrix):
            present = row[row != ""]
            if "changepoint" in present:
                consensus_type[index] = "changepoint"
            elif present.size:
                consensus_type[index] = "anomaly"

        if threshold_type == "percentile":
            positive = score[score > 0]
            threshold_value = (
                np.inf if positive.size == 0 else max(float(np.quantile(positive, threshold)), min_consensus)
            )
        else:
            threshold_value = threshold

        event = _highest_in_each_run(score >= threshold_value, score)
        if use_nms and event.sum() > 1 and time_tolerance > 0:
            positions = np.flatnonzero(event)
            order = positions[np.argsort(-score[positions], kind="stable")]
            selected = []
            for position in order:
                if all(abs(position - previous) > time_tolerance for previous in selected):
                    selected.append(int(position))
            event[:] = False
            event[selected] = True

        anomalies = event & (consensus_type == "anomaly")
        change_points = event & (consensus_type == "changepoint")
        detection = self._restore_refs(
            anomalies=anomalies, change_points=change_points, res=score, threshold=threshold_value
        )
        detection.attrs["model_events"] = model_events
        detection.attrs["score"] = detection.attrs["res"]
        detection.attrs["effective_threshold"] = threshold_value
        detection.attrs["min_consensus"] = min_consensus
        return detection
