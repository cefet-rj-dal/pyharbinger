"""Tests for :class:`HanrFftSma` (adaptive FFT-selected moving-average detector).

Faithful port of R ``hanr_fft_sma``: the gate is exact parity of the detected
index set against the R reference on every ``examples_anomalies`` series, plus the
dominant-period window picked by the ``spec.pgram`` port.
"""
from __future__ import annotations

import pytest

from pyharbinger import HanrFftSma, datasets
from pyharbinger.anomaly import _fft_sma_window

ANOMALIES = datasets.examples_anomalies()


def _events(detection):
    return detection.loc[detection["event"], "idx"].tolist()


# R harbinger hanr_fft_sma on examples_anomalies (0-based detected indices),
# verified via Rscript sourcing harbinger base + hanr_fft_sma.R.
R_EVENTS = {
    "simple": [49],
    "contextual": [],
    "trend": [],
    "multiple": [49],
    "sequence": [49],
    "tt": [37, 49, 89, 100],
    "tt_warped": [37, 49],
    "increasing_amplitude": [910, 940, 970],
    "decreasing_amplitude": [205, 209, 233, 268, 291, 294, 300, 329, 391, 422, 428, 488, 492],
    "volatile": [708, 714, 850, 945, 957, 960],
}

# R find_best_moving_average (spec.pgram dominant period) per series.
R_WINDOW = {
    "simple": 27, "contextual": 27, "trend": 27, "multiple": 27, "sequence": 27,
    "tt": 27, "tt_warped": 27, "increasing_amplitude": 64,
    "decreasing_amplitude": 64, "volatile": 171,
}


@pytest.mark.parametrize("name", list(R_EVENTS))
def test_fft_sma_matches_r(name):
    serie = ANOMALIES[name]["serie"]
    det = HanrFftSma().detect(serie)
    assert _events(det) == R_EVENTS[name]


@pytest.mark.parametrize("name", list(R_WINDOW))
def test_fft_sma_window_matches_r(name):
    serie = ANOMALIES[name]["serie"].to_numpy(dtype=float)
    assert _fft_sma_window(serie) == R_WINDOW[name]


def test_fft_sma_detection_table_shape():
    serie = ANOMALIES["simple"]["serie"]
    det = HanrFftSma().detect(serie)
    assert set(det.columns) == {"idx", "event", "type"}
    assert len(det) == len(serie)
    # Warm-up window is never flagged.
    window = _fft_sma_window(serie.to_numpy(dtype=float))
    assert not det["event"].to_numpy()[: window - 1].any()


def test_fft_sma_none_raises():
    with pytest.raises(ValueError):
        HanrFftSma().detect(None)
