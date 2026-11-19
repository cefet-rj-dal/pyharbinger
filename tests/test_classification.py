"""Tests for the classification-based anomaly detector (HancMl)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyharbinger import HancMl, datasets

ClaKnn = pytest.importorskip("pydaltoolbox").ClaKnn


def _tt_train_test():
    """Reproduce R's hanc_ml example setup on examples_anomalies$tt."""
    tt = datasets.examples_anomalies()["tt"]
    df = pd.DataFrame(
        {
            "serie": tt["serie"].to_numpy(dtype=float),
            "event": np.where(tt["event"].to_numpy().astype(bool), "TRUE", "FALSE"),
        }
    )
    train = df.iloc[:80].reset_index(drop=True)
    test = df.iloc[80:].reset_index(drop=True)
    # R's minmax() normalizes the numeric feature by train min/max (factor left alone).
    lo, hi = train["serie"].min(), train["serie"].max()
    train = train.assign(serie=(train["serie"] - lo) / (hi - lo))
    test = test.assign(serie=(test["serie"] - lo) / (hi - lo))
    return train, test


def _events(detection):
    return (np.where(detection["event"].to_numpy())[0] + 1).tolist()


def test_hancml_matches_r_knn():
    # R hanc_ml(cla_knn("event", slevels, k=3)) flags events at 10 and 21 on the test set.
    train, test = _tt_train_test()
    model = HancMl(ClaKnn("event", ["FALSE", "TRUE"], k=3)).fit(train)
    det = model.detect(test)
    assert _events(det) == [10, 21]


def test_hancml_detection_table_contract():
    train, test = _tt_train_test()
    det = HancMl(ClaKnn("event", ["FALSE", "TRUE"], k=3)).fit(train).detect(test)
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == len(test)
    # Flagged rows are anomalies; the threshold is carried on the table.
    assert (det["type"][det["event"]] == "anomaly").all()
    assert det.attrs["threshold"] == 0.5


def test_hancml_high_threshold_flags_nothing():
    train, test = _tt_train_test()
    det = HancMl(ClaKnn("event", ["FALSE", "TRUE"], k=3), threshold=1.01).fit(train).detect(test)
    assert not det["event"].any()


def test_hancml_requires_data():
    train, _ = _tt_train_test()
    model = HancMl(ClaKnn("event", ["FALSE", "TRUE"], k=3)).fit(train)
    with pytest.raises(ValueError, match="No data"):
        model.detect(None)
