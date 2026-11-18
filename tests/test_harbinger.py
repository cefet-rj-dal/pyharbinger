"""Tests for the Harbinger base class contract and index-restoration helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyharbinger import Harbinger


def test_base_detect_flags_nothing():
    base = Harbinger()
    det = base.detect(pd.Series(np.arange(10.0)))
    assert list(det.columns) == ["idx", "event", "type"]
    assert det.shape == (10, 3)
    assert not det["event"].any()
    assert (det["type"] == "").all()


def test_fit_is_chainable():
    base = Harbinger()
    assert base.fit(pd.Series([1.0, 2.0, 3.0])) is base


def test_store_refs_omits_na():
    base = Harbinger()
    base._store_refs(np.array([1.0, np.nan, 3.0, 4.0]))
    assert base._n == 4
    assert base.non_na.tolist() == [0, 2, 3]
    assert base.serie.tolist() == [1.0, 3.0, 4.0]


def test_restore_refs_maps_anomalies_back_over_na():
    base = Harbinger()
    base._store_refs(np.array([1.0, np.nan, 3.0, 4.0]))
    # anomalies computed on the 3 non-NA positions
    anomalies = np.array([True, False, True])
    det = base._restore_refs(anomalies=anomalies)
    # positions 0 and 3 are anomalies in the original indexing; 1 (NA) stays False
    assert det["event"].tolist() == [True, False, False, True]
    assert det["type"].tolist() == ["anomaly", "", "", "anomaly"]


def test_restore_refs_changepoints():
    base = Harbinger()
    base._store_refs(np.arange(5.0))
    cps = np.array([False, False, True, False, True])
    det = base._restore_refs(change_points=cps)
    assert det["event"].tolist() == [False, False, True, False, True]
    assert det.loc[2, "type"] == "changepoint"


def test_restore_refs_attaches_res_and_threshold():
    base = Harbinger()
    base._store_refs(np.arange(4.0))
    det = base._restore_refs(
        anomalies=np.array([False, True, False, False]),
        res=np.array([0.0, 9.0, 0.0, 0.0]),
        threshold=(-1.0, 1.0),
    )
    assert det.attrs["threshold"] == (-1.0, 1.0)
    assert det.attrs["res"].tolist() == [0.0, 9.0, 0.0, 0.0]


def test_base_evaluate_uses_hard_eval():
    base = Harbinger()
    det = np.array([False, True, True], dtype=bool)
    ev = np.array([False, True, False], dtype=bool)
    m = base.evaluate(det, ev)
    assert m["TP"] == 1 and m["FP"] == 1
