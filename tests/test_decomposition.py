"""Tests for decomposition-based detectors (wavelet / EMD).

Faithful port of the R harbinger decomposition detectors, reusing the native
``pytspredit`` decompositions (no external ``wavelets``/``hht`` dependency). The
gate is exact parity of the detected index set against R on every
``examples_anomalies`` series.
"""
from __future__ import annotations

import pytest

from pyharbinger import HanrWavelet, datasets

ANOMALIES = datasets.examples_anomalies()


def _events(detection):
    return detection.loc[detection["event"], "idx"].tolist()


# R harbinger hanr_wavelet(filter="haar") on examples_anomalies (0-based),
# verified via Rscript sourcing harbinger base + hanr_wavelet.R (wavelets::modwt).
R_WAVELET = {
    "simple": [49],
    "contextual": [],
    "trend": [],
    "multiple": [49],
    "sequence": [49],
    "tt": [11, 37, 49, 100],
    "tt_warped": [11, 23, 37, 49],
    "increasing_amplitude": [910, 940, 970],
    "decreasing_amplitude": [205, 209, 233, 268, 291, 294, 329, 391, 422, 428, 492],
    "volatile": [709, 835, 838, 842, 845, 863, 867],
}


@pytest.mark.parametrize("name", list(R_WAVELET))
def test_hanr_wavelet_matches_r(name):
    serie = ANOMALIES[name]["serie"].to_numpy()
    assert _events(HanrWavelet().fit(serie).detect(serie)) == R_WAVELET[name]


def test_hanr_wavelet_rejects_bad_filter():
    with pytest.raises(ValueError):
        HanrWavelet(filter="not_a_filter")


def test_hanr_wavelet_detect_none_raises():
    with pytest.raises(ValueError):
        HanrWavelet().detect(None)


def test_hanr_wavelet_other_filter_runs():
    serie = ANOMALIES["simple"]["serie"].to_numpy()
    det = HanrWavelet(filter="d4").fit(serie).detect(serie)
    assert list(det.columns[:3]) == ["idx", "event", "type"]
