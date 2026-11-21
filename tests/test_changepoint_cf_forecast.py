"""Tests for the ChangeFinder-forecast detectors (cf_arima, cf_ets).

These are native re-implementations that replace R's ``forecast`` dependency
(``auto.arima`` / ``ets``) -- the back-implementation initiative's P2
"auto-selection that breaks parity" case. Models are fitted with ``statsmodels``
but the order/model selection is ours, so parity is measured against the R
harbinger package on ``examples_changepoints`` and gated by the "small
difference" check:

* ``HcpCfEts`` matches R ``ets()`` EXACTLY on all 7 series.
* ``HcpCfArima`` matches R's integration order ``d`` on all 7 series and matches
  detections exactly on the well-conditioned series (``simple``, ``complex``);
  the ``(p, q)`` stepwise selection diverges from ``forecast`` elsewhere
  (statsmodels full-ML AICc vs ``forecast`` CSS/ML + stepwise path) -- the
  documented tolerance, consistent with PELT / joinpoint / ELM.
"""
from __future__ import annotations

import pytest

from pyharbinger import HcpCfArima, HcpCfEts, datasets
from pyharbinger.changepoint import _kpss_ndiffs

CP = datasets.examples_changepoints()


def _cps(detection):
    """0-based indices of detected change points (the detection ``idx`` column)."""
    return detection.loc[detection["event"], "idx"].tolist()


# --- HcpCfEts: EXACT R parity on all 7 series --------------------------------

ETS_REFERENCE = {
    "simple": [50],
    "sinusoidal": [],
    "incremental": [13, 50, 103, 113, 150],
    "abrupt": [13, 100, 109, 136, 194],
    "volatility": [136, 149, 160, 173, 192, 194],
    "increasing_amplitude": [199, 699],
    "complex": [200],
}


@pytest.mark.parametrize(("name", "expected"), list(ETS_REFERENCE.items()))
def test_cf_ets_matches_r(name, expected):
    det = HcpCfEts().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- HcpCfArima: integration order d matches R ndiffs on all 7 series --------

# R forecast::ndiffs(test="kpss") on each example series.
D_REFERENCE = {
    "simple": 2,
    "sinusoidal": 2,
    "incremental": 1,
    "abrupt": 1,
    "volatility": 0,
    "increasing_amplitude": 0,
    "complex": 1,
}


@pytest.mark.parametrize(("name", "expected"), list(D_REFERENCE.items()))
def test_cf_arima_ndiffs_matches_r(name, expected):
    assert _kpss_ndiffs(CP[name]["serie"].to_numpy(dtype=float)) == expected


# --- HcpCfArima: EXACT R parity on the well-conditioned series ----------------

ARIMA_REFERENCE = {
    "simple": ([50], (0, 2, 0, False)),
    "complex": ([200], (2, 1, 2, False)),
}


@pytest.mark.parametrize(("name", "expected"), list(ARIMA_REFERENCE.items()))
def test_cf_arima_matches_r_well_conditioned(name, expected):
    expected_cps, expected_order = expected
    model = HcpCfArima()
    model.fit(CP[name]["serie"].to_numpy(dtype=float))
    det = model.detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected_cps
    assert model.order == expected_order


# --- structure / API ---------------------------------------------------------


def test_cf_arima_detect_without_fit_autoselects():
    # detect() should auto-fit when fit() was not called first.
    model = HcpCfArima()
    det = model.detect(CP["simple"]["serie"].to_numpy(dtype=float))
    assert model.order is not None
    assert model.sw_size is not None
    assert list(det.columns) == ["idx", "event", "type"]
    assert det["event"].dtype == bool


def test_cf_arima_sw_size_from_order():
    model = HcpCfArima()
    model.fit(CP["simple"]["serie"].to_numpy(dtype=float))
    p, d, q, _ = model.order
    assert model.sw_size == max(p, d + 1, q)


def test_cf_ets_detect_structure():
    det = HcpCfEts().detect(CP["sinusoidal"]["serie"].to_numpy(dtype=float))
    n = len(CP["sinusoidal"])
    assert len(det) == n
    assert set(det["type"].unique()) <= {"", "changepoint"}


def test_cf_none_input_raises():
    with pytest.raises(ValueError, match="No data"):
        HcpCfArima().detect(None)
    with pytest.raises(ValueError, match="No data"):
        HcpCfEts().detect(None)
    with pytest.raises(ValueError, match="No data"):
        HcpCfArima().fit(None)
