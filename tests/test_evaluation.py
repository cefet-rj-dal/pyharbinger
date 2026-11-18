"""Tests for HarEval and HarEvalSoft, verified against the R harbinger package."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pyharbinger import HarEval, HarEvalSoft

DETECTION = np.array(
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool
)
EVENT = np.array(
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=bool
)


def test_hard_eval_matches_r():
    m = HarEval().evaluate(DETECTION, EVENT)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (3, 3, 1, 8)
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.75) < 1e-9
    assert abs(m["F1"] - 0.6) < 1e-9
    assert abs(m["accuracy"] - 0.733333333) < 1e-6


def test_soft_eval_matches_r():
    m = HarEvalSoft(sw_size=3).evaluate(DETECTION, EVENT)
    assert abs(m["TP"] - 3.666667) < 1e-5
    assert abs(m["FP"] - 2.333333) < 1e-5
    assert abs(m["FN"] - 0.333333) < 1e-5
    assert abs(m["TN"] - 8.666667) < 1e-5
    assert abs(m["precision"] - 0.611111) < 1e-5
    assert abs(m["recall"] - 0.916667) < 1e-5
    assert abs(m["F1"] - 0.733333) < 1e-5


def test_eval_accepts_detection_dataframe():
    det = pd.DataFrame({"idx": np.arange(len(DETECTION)), "event": DETECTION, "type": ""})
    m = HarEval().evaluate(det, EVENT)
    assert m["TP"] == 3


def test_eval_handles_na_detection():
    det = np.array([True, None, False, True], dtype=object)
    ev = np.array([True, False, False, False], dtype=bool)
    m = HarEval().evaluate(det, ev)
    # NA -> False; detections at 0 (TP) and 3 (FP).
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 0, 2)


def test_soft_eval_falls_back_when_empty():
    det = np.zeros(10, dtype=bool)
    ev = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    m = HarEvalSoft().evaluate(det, ev)
    # No detections -> degenerate; precision is NaN (0/0), recall 0.
    assert m["TP"] == 0
    assert math.isnan(m["precision"])


def test_confusion_matrix_shape():
    m = HarEval().evaluate(DETECTION, EVENT)
    cm = m["confMatrix"]
    assert cm.shape == (2, 2)
    assert cm.loc["detection_TRUE", "event_TRUE"] == 3
