"""Tests for the ChangeFinder detectors (cf_arima, cf_ets) -- the "++" re-engineering.

``HcpCfArima`` / ``HcpCfEts`` were faithful native reimplementations that scored
``statsmodels`` ARIMA/ETS residuals (replacing R's ``forecast`` auto-selection).
They were then **re-engineered into "++" versions**: a native, pure-numpy **robust
ChangeFinder** that fuses the canonical two-stage online **SDAR** surprise score
(Takeuchi & Yamanishi 2006) with a Gaussian **mean+variance structural break**
(shared with chow/gft). ``HcpCfArima`` uses an autoregressive SDAR predictor;
``HcpCfEts`` a discounted Holt predictor. No ``forecast`` and no ``statsmodels``.

Because this deliberately diverges, the gate is detection quality vs the ground
truth (SoftED-F1), which must beat the R reference. See
``develop/benchmark_backimpl.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HarEvalSoft, HcpCfArima, HcpCfEts, datasets
from pyharbinger.changepoint import _sdar_ar_scores, _sdar_holt_scores

CP = datasets.examples_changepoints()


def _cps(detection):
    return [int(i) for i in np.where(detection["event"].to_numpy())[0]]


def _softed_total(cls) -> float:
    total = 0.0
    for df in CP.values():
        gt = df["event"].to_numpy(dtype=bool)
        det = cls().detect(df["serie"].to_numpy(dtype=float))["event"].to_numpy(dtype=bool)
        f1 = HarEvalSoft(sw_size=15).evaluate(det, gt)["F1"]
        total += 0.0 if (f1 is None or np.isnan(f1)) else float(f1)
    return total


# --- Action-1++ gate: the ++ version must beat the R reference ----------------

# Summed SoftED-F1 of the reference R ``forecast`` detections over the 7 series.
R_REF_TOTAL = {"cf_arima": 2.918, "cf_ets": 2.662}


def test_cf_arima_beats_r_reference():
    assert _softed_total(HcpCfArima) > R_REF_TOTAL["cf_arima"]


def test_cf_ets_beats_r_reference():
    assert _softed_total(HcpCfEts) > R_REF_TOTAL["cf_ets"]


# --- Regression anchors (current ++ detections, 0-based) ---------------------

ARIMA_EXPECTED = {
    "simple": [48],
    "sinusoidal": [49],
    "incremental": [85, 145],
    "abrupt": [99, 100, 102, 195],
    "volatility": [103, 122, 126, 195],
    "increasing_amplitude": [199, 204, 501, 504, 699, 704],
    "complex": [101, 199, 200],
}
ETS_EXPECTED = {
    "simple": [48],
    "sinusoidal": [49],
    "incremental": [85],
    "abrupt": [99, 106, 195],
    "volatility": [103, 122, 195],
    "increasing_amplitude": [199, 201, 204, 501, 504, 507, 699, 700, 704],
    "complex": [109, 114, 199, 200, 205],
}


@pytest.mark.parametrize(("name", "expected"), list(ARIMA_EXPECTED.items()))
def test_cf_arima_detections(name, expected):
    assert _cps(HcpCfArima().detect(CP[name]["serie"].to_numpy(dtype=float))) == expected


@pytest.mark.parametrize(("name", "expected"), list(ETS_EXPECTED.items()))
def test_cf_ets_detections(name, expected):
    assert _cps(HcpCfEts().detect(CP[name]["serie"].to_numpy(dtype=float))) == expected


# --- SDAR engine properties ---------------------------------------------------


def test_sdar_ar_spikes_at_abrupt_change():
    # The AR log-loss score should surge right after an abrupt mean shift.
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)])
    sc = _sdar_ar_scores(x, r=0.1, order=2)
    assert np.argmax(sc) >= 80  # peak is at/after the change, not before
    assert sc[:2].sum() == 0.0  # the first ``order`` (warm-up) positions score zero


def test_sdar_holt_scores_length_and_warmup():
    x = CP["abrupt"]["serie"].to_numpy(dtype=float)
    sc = _sdar_holt_scores(x, alpha=0.8, beta=0.1, r=0.05)
    assert len(sc) == len(x)
    assert np.all(np.isfinite(sc))


def test_cf_arima_order_param_wired():
    s = CP["volatility"]["serie"].to_numpy(dtype=float)
    model = HcpCfArima(order=3)
    assert model.order == 3
    # a higher AR order changes the surprise scores -> generally a different result
    assert _cps(model.detect(s)) != _cps(HcpCfArima(order=1).detect(s))


# --- structure / API ---------------------------------------------------------


def test_cf_detect_structure():
    for cls in (HcpCfArima, HcpCfEts):
        det = cls().detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
        assert list(det.columns) == ["idx", "event", "type"]
        assert det["event"].dtype == bool
        assert set(det["type"].unique()) <= {"", "anomaly", "changepoint"}


def test_cf_none_input_raises():
    with pytest.raises(ValueError, match="No data"):
        HcpCfArima().detect(None)
    with pytest.raises(ValueError, match="No data"):
        HcpCfEts().detect(None)


def test_cf_fit_is_chainable():
    # fit is a no-op that returns self (detect is self-contained).
    model = HcpCfArima()
    assert model.fit(CP["simple"]["serie"].to_numpy(dtype=float)) is model
