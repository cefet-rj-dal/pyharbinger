"""Tests for the extended change-point detectors (cf_lr, scp, joinpoint).

Reference change points verified against the R harbinger package on
``examples_changepoints``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HcpCfLr, HcpJoinpoint, HcpScp, datasets

CP = datasets.examples_changepoints()


def _cps(detection):
    return [int(i) + 1 for i in np.where(detection["event"].to_numpy())[0]]


# --- HcpCfLr: exact R parity on every example series -------------------------

CF_LR_REFERENCE = {
    "simple": [],
    "sinusoidal": [],
    "incremental": [104, 185, 200],
    "abrupt": [89, 93, 95, 103],
    "volatility": [131, 140, 145, 156, 161, 168, 193],
    "increasing_amplitude": [700],
    "complex": [],
}


@pytest.mark.parametrize(("name", "expected"), list(CF_LR_REFERENCE.items()))
def test_cf_lr_matches_r(name, expected):
    det = HcpCfLr().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- HcpScp: exact R parity on every example series --------------------------

SCP_REFERENCE = {
    "simple": [50],
    "sinusoidal": [],
    "incremental": [88, 97, 105, 186],
    "abrupt": [90],
    "volatility": [151, 159, 172, 180, 183],
    "increasing_amplitude": [701],
    "complex": [112, 121, 131, 140, 150, 160, 170, 180, 193],
}


@pytest.mark.parametrize(("name", "expected"), list(SCP_REFERENCE.items()))
def test_scp_matches_r(name, expected):
    det = HcpScp().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- HcpJoinpoint: exact R parity on every example series --------------------
#
# The joinpoint DP is identical in R (hcp_joinpoint) and Python (releitura++).
# A tolerant tie-break (``_JOINPOINT_TOL``) plus an RSS floor
# (``_JOINPOINT_RSS_FLOOR``) make the segmentation backend-independent, so the
# detected indices now match R bit-for-bit on all seven series -- including the
# previously tie-degenerate ``sinusoidal`` and the near-perfectly piecewise-linear
# ``simple`` ramp (which now cleanly recovers its single true change point).

# Only the series up to length 200 are exercised here: the joinpoint DP is
# O(k_max * n^2) least-squares solves, so ``complex`` (n=500) and
# ``increasing_amplitude`` (n=1002) take tens of seconds. R parity was verified
# offline on those two as well (complex -> [161,179,201,308,392];
# increasing_amplitude -> [697,752,817,879,943], same _cps convention).
JOINPOINT_REFERENCE = {
    "simple": [50],
    "sinusoidal": [10, 26, 44, 57, 80],
    "incremental": [64, 87, 103, 164, 187],
    "abrupt": [64, 87, 101, 150, 174],
    "volatility": [96, 137, 150, 192, 196],
}


@pytest.mark.parametrize(("name", "expected"), list(JOINPOINT_REFERENCE.items()))
def test_joinpoint_matches_r(name, expected):
    det = HcpJoinpoint().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


def test_joinpoint_table_contract():
    det = HcpJoinpoint().detect(CP["sinusoidal"]["serie"].to_numpy(dtype=float))
    assert list(det.columns) == ["idx", "event", "type"]
    assert (det["type"][det["event"]] == "changepoint").all()


def test_extended_cp_require_data():
    for cls in (HcpCfLr, HcpScp, HcpJoinpoint):
        with pytest.raises(ValueError, match="No data"):
            cls().detect(None)
