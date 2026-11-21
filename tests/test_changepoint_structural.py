"""Tests for the structural-change detectors (chow, gft) -- the "++" re-engineering.

``HcpChow`` / ``HcpGft`` started as faithful native ports of R's ``strucchange``
backend. They were then **re-engineered into "++" versions**: the least-squares
(mean-only) criterion is generalised to a Gaussian **mean + variance** cost, so
the detectors also catch scale/variance-regime changes; ``HcpGft`` uses the
conservative LWZ penalty for the break count. Because this deliberately diverges
from ``strucchange``, the acceptance criterion is no longer R parity but
**detection quality vs the ground truth** (SoftED-F1), which must beat the R
reference. See ``develop/benchmark_backimpl.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HarEvalSoft, HcpChow, HcpGft, datasets
from pyharbinger.changepoint import _meanvar_breakpoints, _meanvar_single_break

CP = datasets.examples_changepoints()


def _cps(detection):
    return [int(i) for i in np.where(detection["event"].to_numpy())[0]]


def _softed_total(cls) -> float:
    """Summed SoftED-F1 (window 15) vs ground truth over the 7 series."""
    total = 0.0
    for df in CP.values():
        gt = df["event"].to_numpy(dtype=bool)
        det = cls().detect(df["serie"].to_numpy(dtype=float))["event"].to_numpy(dtype=bool)
        f1 = HarEvalSoft(sw_size=15).evaluate(det, gt)["F1"]
        total += 0.0 if (f1 is None or np.isnan(f1)) else float(f1)
    return total


# --- Action-1++ gate: the ++ version must beat the R reference ----------------

# Summed SoftED-F1 of the reference R ``strucchange`` detections (from the
# faithful-port milestone) over the 7 series -- the bar the ++ must clear.
R_REF_TOTAL = {"chow": 3.307, "gft": 2.200}


def test_chow_beats_r_reference():
    assert _softed_total(HcpChow) > R_REF_TOTAL["chow"]


def test_gft_beats_r_reference():
    assert _softed_total(HcpGft) > R_REF_TOTAL["gft"]


# --- Regression anchors (current ++ detections, 0-based) ---------------------

CHOW_EXPECTED = {
    "simple": [48], "sinusoidal": [49], "incremental": [85], "abrupt": [99],
    "volatility": [122], "increasing_amplitude": [504], "complex": [199],
}
GFT_EXPECTED = {
    "simple": [48], "sinusoidal": [25, 41, 57, 73], "incremental": [85],
    "abrupt": [99], "volatility": [122], "increasing_amplitude": [504],
    "complex": [199, 297, 390],
}


@pytest.mark.parametrize(("name", "expected"), list(CHOW_EXPECTED.items()))
def test_chow_detections(name, expected):
    assert _cps(HcpChow().detect(CP[name]["serie"].to_numpy(dtype=float))) == expected


@pytest.mark.parametrize(("name", "expected"), list(GFT_EXPECTED.items()))
def test_gft_detections(name, expected):
    assert _cps(HcpGft().detect(CP[name]["serie"].to_numpy(dtype=float))) == expected


# --- The mean+variance innovation actually catches a variance break ----------


def test_meanvar_catches_pure_variance_break():
    # A constant-mean series whose variance jumps at t=100: a mean-only Chow
    # would be blind to it, but the mean+variance LR locates it.
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 0.2, 100), rng.normal(0, 2.0, 100)])
    bp = _meanvar_single_break(y, 0.15)
    assert bp is not None
    assert abs((bp - 1) - 100) <= 15  # within the SoftED tolerance window


def test_gft_lwz_is_more_conservative_than_bic():
    # The conservative LWZ penalty should never select more breaks than BIC.
    for df in CP.values():
        y = df["serie"].to_numpy(dtype=float)
        n_lwz = len(_meanvar_breakpoints(y, 0.15, "lwz"))
        n_bic = len(_meanvar_breakpoints(y, 0.15, "bic"))
        assert n_lwz <= n_bic


def test_gft_penalty_param_wired():
    y = CP["increasing_amplitude"]["serie"].to_numpy(dtype=float)
    assert _cps(HcpGft(penalty="lwz").detect(y)) == _cps(
        HcpGft().detect(y)  # lwz is the default
    )


# --- Contract ----------------------------------------------------------------


def test_structural_cp_table_contract():
    for cls in (HcpChow, HcpGft):
        det = cls().detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
        assert list(det.columns) == ["idx", "event", "type"]
        assert (det["type"][det["event"]] == "changepoint").all()


def test_structural_cp_require_data():
    for cls in (HcpChow, HcpGft):
        with pytest.raises(ValueError, match="No data"):
            cls().detect(None)
