"""Tests for the multivariate PCA anomaly detector (HmuPca)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyharbinger import HmuPca, datasets


def _multivariate_fixture():
    """A deterministic 2-column series from shipped univariate examples.

    Column ``a`` carries the ``simple`` anomaly (ground truth at index 49).
    """
    an = datasets.examples_anomalies()
    return pd.DataFrame(
        {
            "a": an["simple"]["serie"].to_numpy(dtype=float),
            "b": an["contextual"]["serie"].to_numpy(dtype=float),
        }
    )


def _events(detection):
    return (np.where(detection["event"].to_numpy())[0] + 1).tolist()


def test_hmupca_matches_fixed_r_default():
    # Fixed R hmu_pca (variance=0.9, auto k=1 here) flags the anomaly at idx 50.
    mv = _multivariate_fixture()
    det = HmuPca().detect(mv)
    res = np.asarray(det.attrs["res"], dtype=float)
    assert _events(det) == [50]
    assert np.nanmax(res) == pytest.approx(1.054, abs=1e-3)


def test_hmupca_num_components_override():
    mv = _multivariate_fixture()
    det = HmuPca(num_components=1).detect(mv)
    assert _events(det) == [50]


def test_hmupca_multivariate_na_handling():
    # A row with any NA column is dropped during computation but restored in the table.
    mv = _multivariate_fixture()
    mv.loc[5, "b"] = np.nan
    det = HmuPca().detect(mv)
    assert len(det) == len(mv)
    assert not det.loc[5, "event"]
    assert det.loc[5, "type"] == ""


def test_hmupca_detection_table_contract():
    mv = _multivariate_fixture()
    det = HmuPca().detect(mv)
    assert list(det.columns) == ["idx", "event", "type"]
    assert det["idx"].tolist() == list(range(len(mv)))
    assert det["event"].dtype == bool
    assert (det["type"][det["event"]] == "anomaly").all()
