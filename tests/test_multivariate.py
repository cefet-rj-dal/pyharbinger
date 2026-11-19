"""Tests for the multivariate PCA anomaly detector (HmuPca)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyharbinger import HmuPca, datasets


def _multivariate_fixture():
    """A deterministic 2-column series from shipped univariate examples."""
    an = datasets.examples_anomalies()
    return pd.DataFrame(
        {
            "a": an["simple"]["serie"].to_numpy(dtype=float),
            "b": an["contextual"]["serie"].to_numpy(dtype=float),
        }
    )


def test_hmupca_faithful_default_is_degenerate():
    # Faithful to R's hmu_pca: keeping all components => residual is float round-off.
    # R reports max|res| ~ 9.86e-32; Python matches that order of magnitude.
    mv = _multivariate_fixture()
    det = HmuPca().detect(mv)
    res = np.asarray(det.attrs["res"], dtype=float)
    assert len(det) == len(mv)
    assert np.nanmax(np.abs(res)) < 1e-6


def test_hmupca_n_components_extension_detects():
    # pyharbinger extension: dropping a component makes it a real detector.
    mv = _multivariate_fixture()
    det = HmuPca(n_components=1).detect(mv)
    res = np.asarray(det.attrs["res"], dtype=float)
    assert np.nanmax(np.abs(res)) > 0.1
    # Any flagged row is an anomaly.
    assert (det["type"][det["event"]] == "anomaly").all()


def test_hmupca_multivariate_na_handling():
    # A row with any NA column is dropped during computation but restored in the table.
    mv = _multivariate_fixture()
    mv.loc[5, "b"] = np.nan
    det = HmuPca(n_components=1).detect(mv)
    assert len(det) == len(mv)
    # The NA row cannot be flagged and keeps an empty type.
    assert not det.loc[5, "event"]
    assert det.loc[5, "type"] == ""


def test_hmupca_detection_table_contract():
    mv = _multivariate_fixture()
    det = HmuPca(n_components=1).detect(mv)
    assert list(det.columns) == ["idx", "event", "type"]
    assert det["idx"].tolist() == list(range(len(mv)))
    assert det["event"].dtype == bool
