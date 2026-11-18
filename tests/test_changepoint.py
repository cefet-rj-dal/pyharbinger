"""Tests for the change-point detectors.

Page-Hinkley is deterministic and matches R exactly. The ruptures-backed
detectors (AMOC/BinSeg/PELT) use a different cost/penalty than R's
changepoint::cpt.meanvar, so these tests assert sensible behavior and the
documented approximate agreement, not exact equality. KSWIN uses a random
subsample, so it is tested with a fixed seed.
"""
from __future__ import annotations

import numpy as np

from pyharbinger import (
    HcpAmoc,
    HcpBinseg,
    HcpKswin,
    HcpPageHinkley,
    HcpPelt,
    datasets,
)

CP = datasets.examples_changepoints()


def _events(detection):
    return (np.where(detection["event"].to_numpy())[0] + 1).tolist()  # 1-based


def test_page_hinkley_matches_r_exactly():
    # Deterministic: verified against R hcp_page_hinkley.
    assert _events(HcpPageHinkley().detect(CP["simple"]["serie"].to_numpy())) == [29]
    assert _events(HcpPageHinkley().detect(CP["sinusoidal"]["serie"].to_numpy())) == [36]
    assert _events(HcpPageHinkley().detect(CP["abrupt"]["serie"].to_numpy())) == [110]
    assert _events(HcpPageHinkley().detect(CP["volatility"]["serie"].to_numpy())) == []


def test_amoc_single_changepoint():
    det = HcpAmoc().detect(CP["abrupt"]["serie"].to_numpy())
    ev = _events(det)
    assert len(ev) == 1
    # R found the change at 100; ruptures agrees here.
    assert ev == [100]
    assert (det["type"][det["event"]] == "changepoint").all()


def test_binseg_returns_Q_changepoints():
    det = HcpBinseg(Q=2).detect(CP["abrupt"]["serie"].to_numpy())
    ev = _events(det)
    assert len(ev) == 2
    assert ev == [100, 149]  # exact match with R on this series


def test_pelt_segments_series():
    det = HcpPelt().detect(CP["simple"]["serie"].to_numpy())
    ev = _events(det)
    # Default 3*log(n) penalty -> a sensible number of changepoints (not over-split).
    assert 1 <= len(ev) <= 20


def test_pelt_penalty_controls_granularity():
    serie = CP["simple"]["serie"].to_numpy()
    few = len(_events(HcpPelt(penalty=50).detect(serie)))
    many = len(_events(HcpPelt(penalty=2).detect(serie)))
    assert many > few


def test_kswin_reproducible_with_seed():
    serie = CP["abrupt"]["serie"].to_numpy()
    a = _events(HcpKswin(random_state=0).detect(serie))
    b = _events(HcpKswin(random_state=0).detect(serie))
    assert a == b
    assert all(e >= 100 for e in a)  # drift detected after the regime change


def test_kswin_validates_params():
    import pytest

    with pytest.raises(ValueError):
        HcpKswin(window_size=10, stat_size=30)


def test_changepoint_table_shape():
    serie = CP["simple"]["serie"].to_numpy()
    det = HcpPelt().detect(serie)
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == len(serie)
