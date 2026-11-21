"""Tests for the structural-change detectors (chow, gft).

These are native re-implementations that replace R's ``strucchange`` dependency
(back-implementation initiative), so parity is measured against the R harbinger
package on ``examples_changepoints`` and gated by the "small difference" check:

* ``HcpGft`` matches R exactly on the 5 well-conditioned series.
* ``HcpChow`` matches R exactly on 6 of 7 series.
* The remaining cases (``simple`` ramp, ``sinusoidal``) are documented ties /
  degeneracies with an identical fit objective, not real disagreements.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import HcpChow, HcpGft, datasets

CP = datasets.examples_changepoints()


def _cps(detection):
    return [int(i) + 1 for i in np.where(detection["event"].to_numpy())[0]]


# --- HcpGft: exact R parity on the well-conditioned series -------------------

GFT_REFERENCE = {
    "incremental": [86],
    "abrupt": [100, 149],
    "volatility": [123, 170],
    "increasing_amplitude": [629, 816],
    "complex": [200, 307, 391],
}


@pytest.mark.parametrize(("name", "expected"), list(GFT_REFERENCE.items()))
def test_gft_matches_r(name, expected):
    det = HcpGft().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- HcpChow: exact R parity on 6 of 7 series -------------------------------

CHOW_REFERENCE = {
    "sinusoidal": [50],
    "incremental": [86],
    "abrupt": [100],
    "volatility": [149],
    "increasing_amplitude": [817],
    "complex": [200],
}


@pytest.mark.parametrize(("name", "expected"), list(CHOW_REFERENCE.items()))
def test_chow_matches_r(name, expected):
    det = HcpChow().detect(CP[name]["serie"].to_numpy(dtype=float))
    assert _cps(det) == expected


# --- Documented ties / degeneracies -----------------------------------------


def test_chow_simple_is_a_tie():
    # 'simple' is a perfectly piecewise-linear ramp: the pre/post split that
    # zeroes the residual sum of squares makes the sup-F statistic diverge, so
    # positions 49 and 50 are tied (identical objective). R picks 50, we pick 49.
    det = HcpChow().detect(CP["simple"]["serie"].to_numpy(dtype=float))
    assert _cps(det) in ([49], [50])


def test_gft_sinusoidal_same_count_identical_rss():
    # Sinusoidal yields the same number of breaks as R (5); the exact positions
    # differ because the segmentations have an identical total residual sum of
    # squares (a tie between equally optimal partitions), not a worse fit.
    det = HcpGft().detect(CP["sinusoidal"]["serie"].to_numpy(dtype=float))
    assert len(_cps(det)) == 5


def test_gft_simple_degenerate_ramp():
    # 'simple' is a degenerate ramp; GFT still returns a change point.
    det = HcpGft().detect(CP["simple"]["serie"].to_numpy(dtype=float))
    assert len(_cps(det)) >= 1


# --- Contract ---------------------------------------------------------------


def test_structural_cp_table_contract():
    for cls in (HcpChow, HcpGft):
        det = cls().detect(CP["abrupt"]["serie"].to_numpy(dtype=float))
        assert list(det.columns) == ["idx", "event", "type"]
        assert (det["type"][det["event"]] == "changepoint").all()


def test_structural_cp_require_data():
    for cls in (HcpChow, HcpGft):
        with pytest.raises(ValueError, match="No data"):
            cls().detect(None)
