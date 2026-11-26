"""Tests for streaming evaluation (HarStreamEval) and the experiment runner.

Faithful port of R ``har_stream_eval`` / ``har_stream_experiment``: the gate is
exact parity of the trace summary (DP/DL), the hard metrics, the threshold
detection, and the experiment summary table against R on a frozen series.
Reference events are 0-based here (R 1-based 4,9,16 -> 3,8,15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyharbinger import (
    HarMemoryFull,
    HarOnlineSession,
    HarSourceSimulated,
    HarStreamEval,
    HcpPageHinkley,
    har_stream_experiment,
)

SERIE = [10, 11, 12, 20, 12, 11, 10, 9, 30, 31, 32, 10, 11, 12, 13, 40, 12, 11, 10, 9]


def _reference():
    ref = np.zeros(len(SERIE), dtype=bool)
    ref[[3, 8, 15]] = True  # R 1-based 4,9,16
    return ref


def _trace():
    session = HarOnlineSession(
        source=HarSourceSimulated(SERIE), detector=HcpPageHinkley(min_instances=3, threshold=1),
        warmup_size=3, batch_size=2,
    )
    session.fit()
    session.run()
    return session.collect_trace()


def test_stream_eval_summary_matches_r():
    ev = HarStreamEval().evaluate(_trace())
    s = ev["summary"]
    assert s["n_observations"] == 20
    assert s["n_detected"] == 3
    assert s["mean_detection_probability"] == pytest.approx(0.15)
    assert s["median_detection_probability"] == pytest.approx(0.0)
    assert s["mean_detection_lag_batches"] == pytest.approx(0.0)
    assert s["mean_detection_lag_observations"] == pytest.approx(2.0)
    assert s["zero_lag_rate"] == pytest.approx(1.0)
    assert ev["by_observation"] is not None


def test_stream_eval_hard_metrics_match_r():
    ev = HarStreamEval().evaluate(_trace(), reference=_reference())
    h = ev["hard_metrics"]
    assert (h["TP"], h["FP"], h["FN"]) == (2, 1, 1)
    assert h["accuracy"] == pytest.approx(0.9)
    assert h["F1"] == pytest.approx(0.6666667, abs=1e-6)


def test_stream_eval_threshold_detection():
    ev = HarStreamEval().evaluate(_trace(), reference=_reference(), probability_threshold=0.5)
    assert ev["threshold"] == 0.5
    assert int(ev["threshold_detection"].sum()) == 3
    assert ev["threshold_hard_metrics"]["F1"] == pytest.approx(0.6666667, abs=1e-6)


def test_stream_eval_validates_trace():
    with pytest.raises(TypeError):
        HarStreamEval().evaluate([1, 2, 3])
    with pytest.raises(ValueError):
        HarStreamEval().evaluate(pd.DataFrame({"idx": [0, 1]}))


def test_stream_experiment_summary_matches_r():
    exp = har_stream_experiment(
        detectors={"page": HcpPageHinkley(min_instances=3, threshold=1)},
        source_factory=lambda: HarSourceSimulated(SERIE),
        warmup_grid=3, batch_grid=[1, 2], memory_grid=[HarMemoryFull()],
        reference=_reference(),
    )
    summary = exp["summary"]
    assert list(summary["batch_size"]) == [1, 2]
    assert list(summary["n_batches"]) == [17, 9]
    assert summary["mean_detection_probability"].tolist() == pytest.approx([0.15, 0.15])
    assert summary["F1"].tolist() == pytest.approx([0.6666667, 0.6666667], abs=1e-6)
    assert len(exp["runs"]) == 2
    assert (summary["memory"] == "full").all()


def test_stream_experiment_unnamed_detectors_labeled_by_class():
    exp = har_stream_experiment(
        detectors=[HcpPageHinkley(min_instances=3, threshold=1)],
        source_factory=lambda: HarSourceSimulated(SERIE),
        warmup_grid=3, batch_grid=[2], memory_grid=[HarMemoryFull()],
    )
    assert exp["summary"]["detector"].iloc[0] == "HcpPageHinkley"


def test_stream_experiment_validates_inputs():
    with pytest.raises(TypeError):
        har_stream_experiment({"a": HcpPageHinkley()}, source_factory=123,
                              warmup_grid=3, batch_grid=[2], memory_grid=[HarMemoryFull()])
    with pytest.raises(ValueError):
        har_stream_experiment({"a": HcpPageHinkley()}, source_factory=lambda: None,
                              warmup_grid=3, batch_grid=[2], memory_grid=[])
