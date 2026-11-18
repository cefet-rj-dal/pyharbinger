"""Evaluation of event detection.

Python port of the R ``har_eval.R`` (hard evaluation) and ``har_eval_soft.R``
(SoftED soft evaluation). Both produce a confusion matrix and the standard
detection metrics (accuracy, precision, recall, F1, ...).

* :class:`HarEval` -- hard matching: a detection counts only where it coincides
  exactly with a ground-truth event.
* :class:`HarEvalSoft` -- SoftED: detections within a tolerance window of an
  event receive partial credit, matched via the Hungarian algorithm
  (``scipy.optimize.linear_sum_assignment``, the counterpart of R's
  ``RcppHungarian::HungarianSolver``).

Both accept either a boolean array/Series or a detection ``DataFrame`` (its
``event`` column is used).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalBase
from scipy.optimize import linear_sum_assignment

__all__ = ["HarEval", "HarEvalSoft"]


def _as_bool(x) -> np.ndarray:
    """Coerce detection/event input to a boolean array (NaN -> False)."""
    if isinstance(x, pd.DataFrame):
        x = x["event"]
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.dtype == object or np.issubdtype(x.dtype, np.floating):
        x = np.where(pd.isna(x), False, x)
    return x.astype(bool)


def _safe_div(num: float, den: float) -> float:
    """Division returning NaN on a zero denominator (mirrors R's 0/0 = NaN)."""
    return float(num) / float(den) if den != 0 else float("nan")


def _confusion_matrix(tp, fp, fn, tn) -> pd.DataFrame:
    """A 2x2 confusion matrix (rows = detection, columns = event)."""
    return pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["detection_TRUE", "detection_FALSE"],
        columns=["event_TRUE", "event_FALSE"],
    )


def _metrics(tp, fp, fn, tn) -> dict:
    """Assemble the standard metric dictionary from confusion-matrix counts."""
    total = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, total)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, fp + tn)
    prevalence = _safe_div(tp + fn, total)
    ppv = _safe_div(
        sensitivity * prevalence,
        (sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)),
    )
    npv = _safe_div(
        specificity * (1 - prevalence),
        ((1 - sensitivity) * prevalence) + (specificity * (1 - prevalence)),
    )
    detection_rate = _safe_div(tp, total)
    detection_prevalence = _safe_div(tp + fp, total)
    balanced_accuracy = (sensitivity + specificity) / 2
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    beta = 1
    f1 = _safe_div((1 + beta**2) * precision * recall, (beta**2 * precision) + recall)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "confMatrix": _confusion_matrix(tp, fp, fn, tn),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "prevalence": prevalence,
        "PPV": ppv,
        "NPV": npv,
        "detection_rate": detection_rate,
        "detection_prevalence": detection_prevalence,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


class HarEval(DalBase):
    """Hard evaluation of event detection (confusion matrix + metrics)."""

    def evaluate(self, detection, event) -> dict:
        detection = _as_bool(detection)
        event = _as_bool(event)

        tp = int(np.sum(detection & event))
        fp = int(np.sum(detection & ~event))
        fn = int(np.sum(~detection & event))
        tn = int(np.sum(~detection & ~event))
        return _metrics(tp, fp, fn, tn)


class HarEvalSoft(HarEval):
    """SoftED soft evaluation: partial credit within a tolerance window.

    Parameters
    ----------
    sw_size:
        Tolerance window (``k``) for soft matching.
    """

    def __init__(self, sw_size: int = 15):
        self.sw_size = sw_size

    @staticmethod
    def _detection_score(d, e, k):
        """Triangular membership score of detection ``d`` for event ``e``."""
        return np.maximum(np.minimum((d - (e - k)) / k, ((e + k) - d) / k), 0)

    @staticmethod
    def _merge_intervals(intervals: np.ndarray) -> np.ndarray:
        """Merge overlapping [inf, sup] tolerance intervals (sorted by inf)."""
        merged = [intervals[0].copy()]
        for interval in intervals[1:]:
            if interval[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], interval[1])
            else:
                merged.append(interval.copy())
        return np.array(merged)

    def _soft_scores(self, detection: np.ndarray, event: np.ndarray, k: int) -> np.ndarray:
        d_pos = np.where(detection)[0]
        e_pos = np.where(event)[0]

        segments = np.column_stack([e_pos - k, e_pos + k]).astype(float)
        merged = self._merge_intervals(segments)

        scores: list[float] = []
        for inf, sup in merged:
            d_mini = d_pos[(d_pos >= inf) & (d_pos <= sup)]
            e_mini = e_pos[(e_pos >= inf) & (e_pos <= sup)]
            n, m = len(d_mini), len(e_mini)
            if n == 0:
                continue
            if n == 1 and m == 1:
                scores.append(float(self._detection_score(d_mini[0], e_mini[0], k)))
            elif n == 1 and m > 1:
                scores.append(float(np.max(self._detection_score(d_mini[0], e_mini, k))))
            elif n > 1 and m == 1:
                scores.append(float(np.max(self._detection_score(d_mini, e_mini[0], k))))
            elif n > 1 and m > 1:
                # Complex case: optimal detection<->event assignment maximizing score.
                mu = np.array(
                    [[self._detection_score(d, e, k) for e in e_mini] for d in d_mini]
                )
                rows, cols = linear_sum_assignment(-mu)
                scores.extend(mu[rows, cols].tolist())
        return np.array(scores, dtype=float)

    def evaluate(self, detection, event) -> dict:
        detection = _as_bool(detection)
        event = _as_bool(event)

        if detection.sum() == 0 or event.sum() == 0:
            return HarEval().evaluate(detection, event)

        scores = self._soft_scores(detection, event, self.sw_size)

        m = int(event.sum())
        t = len(event)
        n_det = int(detection.sum())

        # SoftED scores each detection in [0, 1]; unmatched detections score 0.
        # FPs is therefore (number of detections) - TPs over all detections,
        # matching R's sum(1 - S_d) with S_d pre-allocated to length(D).
        tps = float(np.sum(scores))
        fps = n_det - tps
        fns = m - tps
        tns = (t - m) - fps
        return _metrics(tps, fps, fns, tns)
