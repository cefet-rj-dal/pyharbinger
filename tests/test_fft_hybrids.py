"""Tests for the FFT-hybrid anomaly detectors (spectral change-point cutoff)."""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import (
    HanrFftAmoc,
    HanrFftAmocCusum,
    HanrFftBinseg,
    HanrFftBinsegCusum,
    datasets,
)

SIMPLE = datasets.examples_anomalies()["simple"]["serie"].to_numpy(dtype=float)


def _events(detection):
    return (np.where(detection["event"].to_numpy())[0] + 1).tolist()


def _res_sum(detection):
    return float(np.nansum(np.asarray(detection.attrs["res"], dtype=float)))


# Reference values from R harbinger on examples_anomalies$simple (anomaly at 50).
R_REFERENCE = {
    HanrFftAmoc: 0.261971,
    HanrFftBinseg: 0.261971,
    HanrFftAmocCusum: 13.187635,
    HanrFftBinsegCusum: 0.059028,
}


@pytest.mark.parametrize("cls", list(R_REFERENCE))
def test_fft_hybrid_matches_r(cls):
    det = cls().detect(SIMPLE)
    assert _events(det) == [50]
    assert _res_sum(det) == pytest.approx(R_REFERENCE[cls], abs=1e-4)


@pytest.mark.parametrize("cls", list(R_REFERENCE))
def test_fft_hybrid_table_contract(cls):
    det = cls().detect(SIMPLE)
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == len(SIMPLE)
    assert (det["type"][det["event"]] == "anomaly").all()


@pytest.mark.parametrize("cls", list(R_REFERENCE))
def test_fft_hybrid_requires_data(cls):
    with pytest.raises(ValueError, match="No data"):
        cls().detect(None)
