"""Tests for the anomaly detectors, verified against the R harbinger package.

Reference values were produced by running the equivalent R detectors on
``examples_anomalies$simple`` (the series carries one ground-truth anomaly at
position 50, 1-based / 49, 0-based).
"""
from __future__ import annotations

import numpy as np

from pyharbinger import HanrFbiad, HanrFft, HanrHistogram, HanrMl, datasets
from pyharbinger.anomaly import _nclass_sturges, r_pretty

SIMPLE = datasets.examples_anomalies()["simple"]["serie"].to_numpy()
GT_ANOMALY = 49  # 0-based position of the ground-truth anomaly


def _events(detection):
    return np.where(detection["event"].to_numpy())[0].tolist()


def test_hanr_fft_matches_r():
    det = HanrFft().detect(SIMPLE)
    assert _events(det) == [49]
    res = np.nan_to_num(det.attrs["res"])
    assert abs(res.sum() - 0.261971) < 1e-5
    assert abs(res.max() - 0.232140) < 1e-5


def test_hanr_histogram_matches_r():
    det = HanrHistogram().detect(SIMPLE)
    # R flags positions 1 and 50 (1-based) -> 0 and 49.
    assert _events(det) == [0, 49]


def test_r_pretty_matches_r_hist_breaks():
    breaks = r_pretty(float(SIMPLE.min()), float(SIMPLE.max()), _nclass_sturges(len(SIMPLE)))
    assert np.allclose(breaks, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


def test_hanr_fbiad_matches_r():
    det = HanrFbiad().detect(SIMPLE)
    assert _events(det) == [49]
    res = np.nan_to_num(det.attrs["res"])
    assert abs(res.sum() - 27.653656) < 1e-4
    assert abs(res.max() - 1.788156) < 1e-5


def test_hanr_ml_knn_matches_r():
    # Deterministic regressor -> exact R parity (events + residuals).
    from pytspredit import TsKnn, TsNormGminmax

    model = HanrMl(TsKnn(TsNormGminmax(), input_size=4, k=3), sw_size=15).fit(SIMPLE)
    det = model.detect(SIMPLE)
    assert _events(det) == [49]
    res = np.nan_to_num(det.attrs["res"])
    assert abs(res.sum() - 0.273302) < 1e-5
    assert abs(res.max() - 0.126520) < 1e-5


def test_hanr_ml_warmup_is_not_flagged():
    from pytspredit import TsKnn

    model = HanrMl(TsKnn(input_size=4, k=3), sw_size=15).fit(SIMPLE)
    det = model.detect(SIMPLE)
    # The first sw_size-1 positions are the warm-up window: never flagged.
    assert not det["event"].to_numpy()[:14].any()


def test_detection_shape_and_columns():
    det = HanrFft().detect(SIMPLE)
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == len(SIMPLE)
    assert set(det["type"].unique()) <= {"", "anomaly"}
