"""Tests for the reference datasets (ported from the R ``data.R``)."""
from __future__ import annotations

import pandas as pd

from pyharbinger import datasets


def test_examples_anomalies_structure():
    data = datasets.examples_anomalies()
    expected = [
        "simple",
        "contextual",
        "trend",
        "multiple",
        "sequence",
        "tt",
        "tt_warped",
        "increasing_amplitude",
        "decreasing_amplitude",
        "volatile",
    ]
    assert list(data) == expected
    for df in data.values():
        assert list(df.columns) == ["serie", "event"]
        assert pd.api.types.is_numeric_dtype(df["serie"])
        assert df["event"].dtype == bool


def test_examples_changepoints_structure():
    data = datasets.examples_changepoints()
    expected = [
        "simple",
        "sinusoidal",
        "incremental",
        "abrupt",
        "volatility",
        "increasing_amplitude",
        "complex",
    ]
    assert list(data) == expected
    for df in data.values():
        assert list(df.columns) == ["serie", "event"]


def test_anomalies_simple_matches_r():
    # Values extracted from the original R examples_anomalies$simple.
    simple = datasets.examples_anomalies()["simple"]
    assert simple.shape == (101, 2)
    assert simple["serie"].iloc[0] == 1.0
    assert abs(simple["serie"].iloc[1] - 0.9689124217106447) < 1e-12
    # The simple series carries a single injected anomaly.
    assert int(simple["event"].sum()) == 1


def test_changepoints_simple_matches_r():
    simple = datasets.examples_changepoints()["simple"]
    assert simple.shape == (101, 2)
    assert simple["serie"].iloc[0] == 0.0
    assert simple["serie"].iloc[1] == 0.25
