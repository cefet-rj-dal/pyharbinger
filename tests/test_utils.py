"""Tests for the detection utilities (deviation / filter / candidate selection).

Reference values verified against the R harbinger ``harutils()`` on the same
deterministic residual vector.
"""
from __future__ import annotations

import numpy as np

from pyharbinger import (
    har_candidate_selection_firstgroup,
    har_candidate_selection_highgroup,
    har_deviation_huber,
    har_deviation_l1,
    har_deviation_l2,
    har_filter_boxplot,
    har_filter_gaussian,
    har_filter_mad,
    har_filter_none,
    har_filter_ratio,
)

RES = np.array(
    [0.1, 0.2, 5.0, 0.15, 0.05, -4.2, 0.3, 0.25, 0.1, 6.1, 5.9, 0.2, 0.1, -0.3, 0.2]
)


def test_deviation_measures_match_r():
    assert np.allclose(har_deviation_l1(RES), np.abs(RES))
    assert np.allclose(har_deviation_l2(RES), RES**2)
    # Huber (k=1.345): spot-check against R values.
    hub = har_deviation_huber(RES)
    assert abs(hub[2] - 5.820487) < 1e-5
    assert abs(hub[9] - 7.299987) < 1e-5


def test_deviation_rowsums_for_matrix():
    m = np.array([[3.0, 4.0], [1.0, 1.0]])
    assert np.allclose(har_deviation_l2(m), [25.0, 2.0])
    assert np.allclose(har_deviation_l1(m), [7.0, 2.0])


def test_filters_match_r_indices_and_thresholds():
    dev = har_deviation_l2(RES)
    # Gaussian: nothing beyond 3 sigma (the spikes inflate sd).
    fg = har_filter_gaussian(dev)
    assert fg.index.tolist() == []
    assert abs(fg.threshold[0] - (-33.599863)) < 1e-4
    assert abs(fg.threshold[1] - 48.94353) < 1e-4
    # Boxplot (0-based indices 2, 9, 10).
    fb = har_filter_boxplot(dev)
    assert fb.index.tolist() == [2, 9, 10]
    assert abs(fb.threshold[1] - 22.138125) < 1e-4
    # MAD (0-based 2, 5, 9, 10).
    fm = har_filter_mad(dev)
    assert fm.index.tolist() == [2, 5, 9, 10]
    # Ratio (0-based 9, 10).
    fr = har_filter_ratio(dev)
    assert fr.index.tolist() == [9, 10]


def test_filter_none():
    f = har_filter_none(har_deviation_l2(RES))
    assert f.index.tolist() == []
    assert f.threshold == (-np.inf, np.inf)


def test_candidate_selection_groups():
    dev = har_deviation_l2(RES)
    fb = har_filter_boxplot(dev)  # indices 2, 9, 10 -> runs {2}, {9,10}
    first = har_candidate_selection_firstgroup(fb, dev, RES)
    assert np.where(first.mask)[0].tolist() == [2, 9]  # first of each run
    high = har_candidate_selection_highgroup(fb, dev, RES)
    # In run {9,10}, position 9 has the larger residual (37.21 vs 34.81).
    assert np.where(high.mask)[0].tolist() == [2, 9]


def test_candidate_selection_highgroup_picks_max():
    # run {1,2,3}; residual largest at position 2
    dev = np.array([0.0, 1.0, 9.0, 2.0, 0.0])
    from pyharbinger.utils import FilterResult

    flt = FilterResult(np.array([1, 2, 3]), (0.0, 0.0))
    high = har_candidate_selection_highgroup(flt, dev, dev)
    assert np.where(high.mask)[0].tolist() == [2]


def test_mas_moving_average():
    from pyharbinger import mas

    x = np.array([1.0, 2, 3, 4, 5])
    # Window means: (1+2+3)/3, (2+3+4)/3, (3+4+5)/3 -> length n-order+1.
    np.testing.assert_allclose(mas(x, 3), [2.0, 3.0, 4.0])
    assert len(mas(x, 1)) == 5
    assert mas(x, 5)[0] == 3.0
