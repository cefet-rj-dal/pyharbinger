"""Reference datasets for examples and tutorials.

Python counterpart of the R ``data.R``. Each collection is the port of an R
``data(...)`` list of labeled time series, shipped as a tidy CSV extracted from
the original ``.RData`` so the Python port is numerically identical to the R
package.

Every series is a :class:`~pandas.DataFrame` with two columns:

* ``serie`` -- the numeric time series values.
* ``event`` -- the boolean ground-truth events (anomaly / change point).

A collection loader returns a ``dict`` mapping each series name to its
DataFrame, mirroring the R list access ``examples_anomalies$simple``::

    from pyharbinger import datasets
    data = datasets.examples_anomalies()
    simple = data["simple"]
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

__all__ = [
    "benchmark",
    "benchmark_url",
    "examples_anomalies",
    "examples_changepoints",
    "examples_harbinger",
    "examples_motifs",
    "list_benchmarks",
    "loadfulldata",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_UNITED_DIR = _DATA_DIR / "united"


def _load_collection(fname: str) -> dict[str, pd.DataFrame]:
    """Load a tidy collection CSV into a dict of per-series DataFrames.

    Columns beyond the ``name``/``idx`` keys and the standard ``serie``/``event``
    pair (e.g. the ``symbol`` ground-truth label on the ECG motif series) are
    preserved, dropping values that are missing for every row of a series.
    """
    raw = pd.read_csv(_DATA_DIR / fname)
    extra_cols = [c for c in raw.columns if c not in ("name", "idx", "serie", "event")]
    collection: dict[str, pd.DataFrame] = {}
    # Preserve the original series order as stored in the CSV.
    for name in raw["name"].drop_duplicates():
        sub = raw[raw["name"] == name]
        cols = {
            "serie": sub["serie"].to_numpy(),
            "event": sub["event"].astype(bool).to_numpy(),
        }
        for col in extra_cols:
            if not sub[col].isna().all():
                cols[col] = sub[col].to_numpy()
        collection[name] = pd.DataFrame(cols).reset_index(drop=True)
    return collection


def examples_anomalies() -> dict[str, pd.DataFrame]:
    """Return the anomaly-detection example series.

    Keys: ``simple``, ``contextual``, ``trend``, ``multiple``, ``sequence``,
    ``tt``, ``tt_warped``, ``increasing_amplitude``, ``decreasing_amplitude``,
    ``volatile``.
    """
    return _load_collection("examples_anomalies.csv")


def examples_changepoints() -> dict[str, pd.DataFrame]:
    """Return the change-point example series.

    Keys: ``simple``, ``sinusoidal``, ``incremental``, ``abrupt``,
    ``volatility``, ``increasing_amplitude``, ``complex``.
    """
    return _load_collection("examples_changepoints.csv")


def examples_motifs() -> dict[str, pd.DataFrame]:
    """Return the motif/discord example series.

    Keys: ``simple`` and the ECG series ``mitdb100`` / ``mitdb102`` (which carry
    an extra ``symbol`` column of beat-type ground-truth labels).
    """
    return _load_collection("examples_motifs.csv")


def examples_harbinger() -> dict[str, pd.DataFrame]:
    """Return the general Harbinger example series.

    Keys: ``nonstationarity``, ``global_temperature_yearly`` /
    ``global_temperature_monthly``, ``multidimensional``, ``seattle_week`` /
    ``seattle_daily``. Series carry ``serie``/``event`` plus any extra columns
    they define (e.g. ``min``/``max``, an auxiliary dimension ``x``).
    """
    return _load_collection("examples_harbinger.csv")


# --- benchmark collection (Eidolon miniatures + on-demand full data) ----------
#
# The ``united`` benchmark datasets (Yahoo A1-A4, gecco, MIT-BIH, NAB, oil_3w,
# UCR) are shipped as compact *miniatures* -- a tidy CSV with the first series
# truncated to 30 rows -- plus a ``manifest.json`` recording each dataset's
# ``full_url``. The complete artifact (every series, full length) is a gzipped
# CSV published in the ``cefet-rj-dal/united`` repository and resolved on demand
# by :func:`loadfulldata` with the standard library only (no extra dependency).
# This mirrors the Eidolon architecture used by ``pytspredit`` / ``tspredit`` /
# ``tspredbench``; the serialization is Python-native (CSV + manifest) rather
# than ``.RData``, with values extracted from the R datasets so they are
# numerically identical.


def _united_manifest() -> dict:
    with open(_UNITED_DIR / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def _frame_list_from_csv(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a tidy long benchmark frame (``series`` + columns) into a dict of frames."""
    collection: dict[str, pd.DataFrame] = {}
    for name in frame["series"].drop_duplicates():
        sub = frame[frame["series"] == name].drop(columns="series").reset_index(drop=True)
        if "event" in sub:
            sub["event"] = sub["event"].fillna(False).astype(bool)
        collection[str(name)] = sub
    return collection


def list_benchmarks() -> list[str]:
    """Return the names of the available benchmark datasets."""
    return sorted(_united_manifest())


def benchmark_url(name: str) -> str:
    """Return the remote URL of a benchmark's full artifact."""
    return _united_manifest()[name]["full_url"]


def benchmark(name: str) -> dict[str, pd.DataFrame]:
    """Return the *miniature* of a benchmark dataset (first series, 30 rows).

    The miniature is a dict mapping each series name to its
    :class:`~pandas.DataFrame`, mirroring the R list access ``gecco[[1]]``. Use
    :func:`loadfulldata` for the complete artifact.
    """
    if name not in _united_manifest():
        raise KeyError(f"unknown benchmark dataset: {name!r}")
    frame = pd.read_csv(_UNITED_DIR / f"{name}.csv")
    return _frame_list_from_csv(frame)


def loadfulldata(name: str) -> dict[str, pd.DataFrame]:
    """Download and return the *full* benchmark dataset referenced by its miniature.

    The full artifact is a gzip-compressed, Python-native CSV published in the
    ``cefet-rj-dal/united`` repository; its URL is recorded as ``full_url`` in the
    manifest. Read with the standard library (``urllib`` + ``gzip``) plus pandas
    -- no extra dependency. The returned structure matches the miniature's.
    """
    entry = _united_manifest()[name]
    with urllib.request.urlopen(entry["full_url"]) as resp:
        data = resp.read()
    frame = pd.read_csv(io.BytesIO(gzip.decompress(data)))
    return _frame_list_from_csv(frame)
