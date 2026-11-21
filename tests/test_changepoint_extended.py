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


# --- HcpJoinpoint -----------------------------------------------------------


def test_joinpoint_matches_r_on_well_conditioned_series():
    # Sinusoidal is well-conditioned: exact R parity.
    det = HcpJoinpoint().detect(CP["sinusoidal"]["serie"].to_numpy(dtype=float))
    assert _cps(det) == [10, 26, 44, 57, 80]


def test_joinpoint_degenerate_piecewise_linear_series():
    # 'simple' is a (near-)perfectly piecewise-linear ramp: the segment RSS
    # collapses to round-off, so log(RSS) dominates the BIC and k_max is chosen.
    # The exact break positions are numerically ill-conditioned (differ from R by
    # round-off), but the true change point (50) is always recovered.
    det = HcpJoinpoint().detect(CP["simple"]["serie"].to_numpy(dtype=float))
    cps = _cps(det)
    assert len(cps) == 5
    assert 50 in cps


def test_joinpoint_table_contract():
    det = HcpJoinpoint().detect(CP["sinusoidal"]["serie"].to_numpy(dtype=float))
    assert list(det.columns) == ["idx", "event", "type"]
    assert (det["type"][det["event"]] == "changepoint").all()


def test_extended_cp_require_data():
    for cls in (HcpCfLr, HcpScp, HcpJoinpoint):
        with pytest.raises(ValueError, match="No data"):
            cls().detect(None)
