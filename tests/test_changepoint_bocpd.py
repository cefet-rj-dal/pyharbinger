"""Tests for the Bayesian Online Change Point Detection detector (bocpd).

``HcpBocpd`` is a native (pure numpy/scipy) back-implementation of R's ``ocp``
package -- the back-implementation initiative's P1 case ("no maintained Python
equivalent -> native is the only faithful path"). ``ocp`` is only an R
``Suggests`` dependency (Adams & MacKay 2007, gaussian UPM).

The "small difference" (Action-1) gate is measured against the *real* ``ocp``
backend (installed from the CRAN archive, ``ocp`` 0.1.1) driven through the R
``hcp_bocpd`` on ``examples_changepoints``: parity is **EXACT** (identical
detected indices) on all seven series, including the long
``increasing_amplitude`` (n=1002).
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HcpBocpd, datasets
from pyharbinger.changepoint import _bocpd_studentpdf

CP = datasets.examples_changepoints()


def _cps(detection):
    """0-based indices of detected change points (the detection ``idx`` column)."""
    return detection.loc[detection["event"], "idx"].tolist()


# --- EXACT R (ocp) parity on all 7 series ------------------------------------

# R harbinger ``hcp_bocpd()`` (defaults) with the real ``ocp`` backend,
# 0-based detected indices on examples_changepoints.
BOCPD_REFERENCE = {
    "simple": [8, 13, 18, 23, 28, 33, 38, 59, 69, 74, 79, 84, 89],
    "sinusoidal": [7, 23, 32, 37, 60, 69, 74, 89],
    "incremental": [128, 175],
    "abrupt": [100, 149],
    "volatility": [123, 149],
    "increasing_amplitude": [
        8, 25, 38, 58, 70, 89, 101, 121, 133, 152, 165, 182, 192, 200, 218,
        228, 245, 258, 278, 290, 309, 321, 353, 371, 384, 403, 415, 435, 448,
        465, 476, 501, 509, 526, 539, 559, 571, 590, 602, 622, 634, 653, 664,
        700, 713, 727, 760, 778, 791, 810, 822, 854, 872, 885, 904, 916, 936,
        949, 966, 976,
    ],
    "complex": [9, 101, 107, 200, 311, 329, 351, 366, 372, 385, 390, 410],
}


@pytest.mark.parametrize(("name", "expected"), list(BOCPD_REFERENCE.items()))
def test_bocpd_matches_r_exactly(name, expected):
    det = HcpBocpd().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- structure / API ----------------------------------------------------------


def test_bocpd_detect_structure():
    det = HcpBocpd().detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
    n = len(CP["abrupt"])
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == n
    assert det["event"].dtype == bool
    assert set(det["type"].unique()) <= {"", "changepoint"}
    # the res score (vote proportion) is attached and in [0, 1]
    res = det.attrs["res"]
    finite = res[~np.isnan(res)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_bocpd_burn_in_zeroes_prefix():
    # With a large burn-in no early change point can be flagged.
    det = HcpBocpd(burn_in=60).detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
    assert all(i >= 60 for i in _cps(det))


def test_bocpd_min_distance_thins():
    s = CP["simple"]["serie"].to_numpy(dtype=float)
    close = _cps(HcpBocpd(min_distance=2).detect(s))
    spread = _cps(HcpBocpd(min_distance=20).detect(s))
    # a larger minimum distance can only keep fewer (or equal) detections
    assert len(spread) <= len(close)
    diffs = np.diff(spread)
    assert all(d >= 20 for d in diffs)


def test_bocpd_higher_threshold_fewer():
    s = CP["abrupt"]["serie"].to_numpy(dtype=float)
    base = _cps(HcpBocpd().detect(s))              # threshold 0.5
    strict = _cps(HcpBocpd(threshold=1.0).detect(s))
    assert set(strict) <= set(base)


def test_bocpd_handles_nan():
    # NaNs are dropped by _store_refs and detections map back to the full index.
    s = CP["abrupt"]["serie"].to_numpy(dtype=float).copy()
    s[5] = np.nan
    det = HcpBocpd().detect(s)
    assert len(det) == len(s)
    assert det["event"].dtype == bool


def test_bocpd_none_input_raises():
    with pytest.raises(ValueError, match="No data"):
        HcpBocpd().detect(None)


def test_bocpd_studentpdf_matches_scipy():
    # The Student-t predictive must agree with scipy's t distribution.
    from scipy import stats

    x, mu, var, nu = 1.3, 0.2, 2.5, 7.0
    scale = np.sqrt(var)
    expected = stats.t.pdf((x - mu) / scale, df=nu) / scale
    got = float(_bocpd_studentpdf(np.array([x]), np.array([mu]),
                                  np.array([var]), np.array([nu]))[0])
    assert got == pytest.approx(expected, rel=1e-10)
