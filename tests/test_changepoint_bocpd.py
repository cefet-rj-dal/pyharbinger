"""Tests for the BOCPD detector (bocpd) -- the "++" re-engineering.

``HcpBocpd`` was a faithful native port of R's ``ocp`` package (i.i.d. Gaussian
UPM + three-list voting). It was then **re-engineered into a "++" version**: a
genuine re-reading of Bayesian Online Change Point Detection (Adams & MacKay
2007) with (1) a conjugate **Bayesian autoregressive UPM** (sensitive to changes
in dynamics/slope, not only level/variance), (2) a **log-space, eps-pruned
streaming** run-length recursion, and (3) change points read from the **Bayesian
predictive surprise** via a robust local z-score, replacing the noisy ``ocp``
voting.

Because this deliberately diverges from ``ocp``, the gate is detection quality
vs the ground truth (SoftED-F1), which must beat the R reference. See
``develop/benchmark_backimpl.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HarEvalSoft, HcpBocpd, datasets
from pyharbinger.changepoint import (
    _bocpd_ar_surprise,
    _bocpd_local_z,
    _bocpd_peaks,
)

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

# Summed SoftED-F1 of the reference R ``ocp`` detections over the 7 series.
R_REF_TOTAL = 1.116


def test_bocpd_beats_r_reference():
    assert _softed_total(HcpBocpd) > R_REF_TOTAL


def test_bocpd_beats_r_reference_by_margin():
    # The AR UPM recovers the trend/slope change points ocp is blind to.
    assert _softed_total(HcpBocpd) > 3.0


# --- Regression anchors (current ++ detections, 0-based) ---------------------

BOCPD_EXPECTED = {
    "simple": [50],
    "sinusoidal": [58],
    "incremental": [50, 103, 150, 195],
    "abrupt": [50, 100, 130, 194],
    "volatility": [50, 102, 123],
    "increasing_amplitude": [199, 501, 699, 744],
    "complex": [200],
}


@pytest.mark.parametrize(("name", "expected"), list(BOCPD_EXPECTED.items()))
def test_bocpd_detections(name, expected):
    assert _cps(HcpBocpd().detect(CP[name]["serie"].to_numpy(dtype=float))) == expected


# --- AR-UPM engine properties -------------------------------------------------

_AR_KW = {"order": 1, "hazard": 250.0, "lam0": 0.1, "a0": 1.0,
          "b0": 1e-2, "prune_eps": 1e-6}


def test_ar_surprise_spikes_at_slope_change():
    # A pure slope change (triangular kink, no mean/variance shift) is invisible
    # to an i.i.d. UPM but breaks the segment regression -> surprise peaks there.
    x = np.concatenate([np.arange(60) * 0.5, 30 - np.arange(1, 61) * 0.5]).astype(float)
    sur = _bocpd_ar_surprise(x, **_AR_KW)
    assert abs(int(np.argmax(sur)) - 60) <= 2


def test_ar_surprise_spikes_at_mean_shift():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)])
    sur = _bocpd_ar_surprise(y, **_AR_KW)
    assert int(np.argmax(sur)) >= 80


def test_ar_surprise_length_and_finite():
    x = CP["abrupt"]["serie"].to_numpy(dtype=float)
    sur = _bocpd_ar_surprise(x, **_AR_KW)
    assert len(sur) == len(x)
    assert np.all(np.isfinite(sur))


def test_eps_pruning_is_stable():
    # More aggressive pruning keeps the same leading change points (the dropped
    # run-length mass is negligible by construction).
    x = CP["abrupt"]["serie"].to_numpy(dtype=float)
    fine = _bocpd_ar_surprise(x, order=1, hazard=250.0, lam0=0.1, a0=1.0,
                              b0=1e-2, prune_eps=1e-8)
    coarse = _bocpd_ar_surprise(x, order=1, hazard=250.0, lam0=0.1, a0=1.0,
                                b0=1e-2, prune_eps=1e-4)
    assert int(np.argmax(fine)) == int(np.argmax(coarse))


def test_local_z_is_causal():
    # The robust local z-score uses only the trailing window -> truncating the
    # tail does not change earlier z-scores (streaming-compatible).
    s = np.abs(np.sin(np.arange(60) / 3.0)) + 0.1
    full = _bocpd_local_z(s, 20)
    partial = _bocpd_local_z(s[:40], 20)
    np.testing.assert_allclose(full[:40], partial, rtol=1e-12)


def test_peaks_thresholds_and_thins():
    s = np.zeros(50)
    s[10] = 8.0
    s[12] = 7.0        # within min_distance of the stronger peak at 10
    s[30] = 6.5
    ev = _bocpd_peaks(s, threshold=6.0, min_distance=5, burn_in=5)
    assert list(np.where(ev)[0]) == [10, 30]


# --- parameter wiring ---------------------------------------------------------


def test_bocpd_order_param_wired():
    s = CP["sinusoidal"]["serie"].to_numpy(dtype=float)
    assert HcpBocpd(order=2).order == 2
    assert _cps(HcpBocpd(order=2).detect(s)) != _cps(HcpBocpd(order=1).detect(s))


def test_bocpd_burn_in_zeroes_prefix():
    det = HcpBocpd(burn_in=60).detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
    assert all(i >= 60 for i in _cps(det))


def test_bocpd_min_distance_thins():
    s = CP["incremental"]["serie"].to_numpy(dtype=float)
    close = _cps(HcpBocpd(min_distance=2).detect(s))
    spread = _cps(HcpBocpd(min_distance=30).detect(s))
    assert len(spread) <= len(close)
    assert all(d >= 30 for d in np.diff(spread))


def test_bocpd_higher_threshold_fewer():
    s = CP["abrupt"]["serie"].to_numpy(dtype=float)
    base = set(_cps(HcpBocpd().detect(s)))
    strict = set(_cps(HcpBocpd(threshold=12.0).detect(s)))
    assert strict <= base


# --- structure / API ---------------------------------------------------------


def test_bocpd_detect_structure():
    det = HcpBocpd().detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
    n = len(CP["abrupt"])
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == n
    assert det["event"].dtype == bool
    assert set(det["type"].unique()) <= {"", "changepoint"}
    res = det.attrs["res"]
    assert np.all(np.isfinite(res[~np.isnan(res)]))


def test_bocpd_handles_nan():
    s = CP["abrupt"]["serie"].to_numpy(dtype=float).copy()
    s[5] = np.nan
    det = HcpBocpd().detect(s)
    assert len(det) == len(s)
    assert det["event"].dtype == bool


def test_bocpd_none_input_raises():
    with pytest.raises(ValueError, match="No data"):
        HcpBocpd().detect(None)


def test_bocpd_fit_is_chainable():
    model = HcpBocpd()
    assert model.fit(CP["simple"]["serie"].to_numpy(dtype=float)) is model
