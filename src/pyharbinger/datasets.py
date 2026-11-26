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

from pathlib import Path

import pandas as pd

__all__ = ["examples_anomalies", "examples_changepoints", "examples_motifs"]

_DATA_DIR = Path(__file__).resolve().parent / "data"


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
