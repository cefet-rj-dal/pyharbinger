"""Tests for the ARIMA++ anomaly detector (``HanrArima``).

``HanrArima`` is the releitura of R ``hanr_arima``: it reuses the DAL ARIMA
auto-selector (``pytspredit.fit_arima``) instead of ``forecast::auto.arima`` and
flags large in-sample residuals. The acceptance criterion is the Action-1++ gate
-- SoftED-F1 against the ground-truth ``event`` column, at least as good as the
reference R detector -- not bit parity with ``forecast``.

Only the length-101 series are exercised here; the three ~1000-point series are
left to the offline benchmark (``develop/benchmark_backimpl.py`` style) for speed.
The compact ``auto="aicc"`` mode is used so the tests stay fast and deterministic.
"""
from __future__ import annotations

import numpy as np

from pyharbinger import HanrArima, HarEvalSoft, datasets

AN = datasets.examples_anomalies()
K = 15  # SoftED tolerance window, as in the benchmark


def _idx(detection):
    return [int(i) for i in np.where(detection["event"].to_numpy())[0]]


def _softed_f1(detection, event_bool):
    f1 = HarEvalSoft(sw_size=K).evaluate(
        detection["event"].to_numpy(dtype=bool), np.asarray(event_bool, dtype=bool)
    )["F1"]
    return 0.0 if (f1 is None or np.isnan(f1)) else float(f1)


def test_hanr_arima_detects_the_simple_anomaly():
    # The single injected anomaly on ``simple`` is at index 49; HanrArima recovers
    # exactly it (the warm-up neutralisation removes statsmodels' init transient).
    det = HanrArima(auto="aicc").detect(AN["simple"]["serie"].to_numpy(dtype=float))
    assert _idx(det) == [49]


def test_hanr_arima_meets_action1_gate_on_sequence():
    # Action-1++ gate: on ``sequence`` R's forecast::auto.arima detector scores a
    # SoftED-F1 of 0.667 against the ground truth; the ARIMA++ releitura must match
    # or beat it (it reaches 1.0).
    df = AN["sequence"]
    det = HanrArima(auto="aicc").detect(df["serie"].to_numpy(dtype=float))
    assert _softed_f1(det, df["event"].to_numpy(dtype=bool)) >= 0.667


def test_hanr_arima_order_and_warmup_are_set_after_fit():
    det = HanrArima(auto="aicc")
    det.fit(AN["simple"]["serie"].to_numpy(dtype=float))
    assert det.p >= 0 and det.d >= 0 and det.q >= 0
    assert det.sw_size == max(det.p, det.d + 1, det.q)
