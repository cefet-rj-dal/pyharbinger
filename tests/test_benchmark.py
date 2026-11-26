"""Tests for the benchmark dataset collection (Eidolon miniatures + loadfulldata).

The miniatures ship with the package; ``loadfulldata`` resolves the full artifact
from a remote URL. The mini accessors are checked against the shipped data, and
``loadfulldata`` is checked against a local gzip served over a ``file://`` URL
(so the test needs no network and no sibling repo).
"""
from __future__ import annotations

import gzip

import pandas as pd
import pytest

from pyharbinger import datasets

MANIFEST = datasets._united_manifest()


def test_list_benchmarks_covers_all_families():
    names = datasets.list_benchmarks()
    assert len(names) == 26
    assert "gecco" in names
    assert any(n.startswith("A1") for n in names)
    assert any(n.startswith("mit_bih") for n in names)
    assert any(n.startswith("nab_") for n in names)
    assert any(n.startswith("oil_3w") for n in names)
    assert any(n.startswith("ucr_") for n in names)


def test_benchmark_miniature_is_first_series_30_rows():
    mini = datasets.benchmark("gecco")
    assert list(mini) == ["tp"]
    df = mini["tp"]
    assert len(df) == 30
    assert list(df.columns) == ["idx", "value", "event", "type"]
    assert df["event"].dtype == bool


def test_benchmark_multivariate_schema():
    # oil_3w is multivariate: several value columns instead of a single `value`.
    mini = datasets.benchmark("oil_3w_Type_1")
    df = next(iter(mini.values()))
    assert "event" in df.columns
    assert "p_pdg" in df.columns  # a process variable column


def test_all_miniatures_load():
    for name in datasets.list_benchmarks():
        mini = datasets.benchmark(name)
        assert len(mini) >= 1
        first = next(iter(mini.values()))
        assert len(first) == 30
        assert first["event"].dtype == bool


def test_benchmark_url_and_unknown():
    url = datasets.benchmark_url("gecco")
    assert url.endswith("/python/gecco.csv.gz")
    assert "cefet-rj-dal/united" in url
    with pytest.raises(KeyError):
        datasets.benchmark("does_not_exist")


def test_examples_harbinger_loads_with_per_series_columns():
    h = datasets.examples_harbinger()
    assert set(h) == {
        "nonstationarity", "global_temperature_yearly", "global_temperature_monthly",
        "multidimensional", "seattle_week", "seattle_daily",
    }
    assert h["nonstationarity"]["event"].dtype == bool
    # Extra columns are kept only where a series defines them.
    assert list(h["nonstationarity"].columns) == ["serie", "event"]
    assert "min" in h["seattle_daily"].columns
    assert "x" in h["multidimensional"].columns


def test_loadfulldata_reads_gzip_via_file_url(tmp_path, monkeypatch):
    # Build a tidy long frame with two series, gzip it, and serve via file://.
    frame = pd.DataFrame({
        "series": ["a", "a", "b"],
        "idx": [1, 2, 1],
        "value": [1.0, 2.0, 3.0],
        "event": [False, True, False],
        "type": ["", "anomaly", ""],
    })
    gz_path = tmp_path / "toy.csv.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8", newline="") as fh:
        frame.to_csv(fh, index=False)

    monkeypatch.setattr(
        datasets, "_united_manifest",
        lambda: {"toy": {"kind": "frame_list", "full_url": gz_path.as_uri()}},
    )
    full = datasets.loadfulldata("toy")
    assert set(full) == {"a", "b"}
    assert len(full["a"]) == 2
    assert full["a"]["event"].tolist() == [False, True]
    assert "series" not in full["a"].columns
