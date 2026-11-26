"""Streaming evaluation and experiment runner for online detection.

Python port of the R ``har_stream_eval`` / ``har_stream_experiment``. These sit
on top of the online layer (:mod:`pyharbinger.online`): :class:`HarStreamEval`
scores a trace produced by ``HarOnlineSession.collect_trace`` with the online
metrics Detection Probability (DP) and Detection Lag (DL) -- and, when labels
are given, with the standard hard evaluation; :func:`har_stream_experiment` runs
a reproducible grid of online configurations and returns a comparison table.

References
----------
- Ogasawara E, Salles R, Porto F, Pacitti E (2025). Event Detection in Time
  Series. Springer. doi:10.1007/978-3-031-75941-3

Porting
-------
Mirrors ``har_stream_eval.R`` / ``har_stream_experiment.R``. Fidelity: exact on
the trace summaries and the hard metrics (the trace itself has exact parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import HarEval
from .online import HarOnlineRefitFull, HarOnlineSession, har_memory_label

__all__ = ["HarStreamEval", "har_stream_experiment"]

_REQUIRED_TRACE_COLUMNS = (
    "idx",
    "batch_frequency",
    "detection_frequency",
    "detection_probability",
    "detection_lag_batches",
    "detection_lag_observations",
)


class HarStreamEval:
    """Evaluate a streaming trace (DP / DL summaries, plus optional hard metrics).

    Complements :class:`~pyharbinger.HarEval` by summarizing the online trace
    produced by :meth:`~pyharbinger.HarOnlineSession.collect_trace`. Mirrors R
    ``har_stream_eval``.
    """

    def evaluate(self, detection, reference=None, probability_threshold=None) -> dict:
        """Summarize a trace; add hard metrics when a ``reference`` is given.

        Parameters
        ----------
        detection:
            The trace DataFrame returned by ``collect_trace()``.
        reference:
            Optional ground-truth event vector aligned with the observation index.
        probability_threshold:
            Optional DP threshold used to derive a filtered detection vector.
        """
        if not isinstance(detection, pd.DataFrame):
            raise TypeError("detection must be a trace DataFrame returned by collect_trace().")
        if not set(_REQUIRED_TRACE_COLUMNS).issubset(detection.columns):
            raise ValueError("detection does not contain the required trace columns.")

        n = len(detection)
        observed = (detection["detection_frequency"] > 0).to_numpy()
        detected_rows = detection[observed]
        prob = pd.to_numeric(detection["detection_probability"], errors="coerce")
        lag_b = pd.to_numeric(detected_rows["detection_lag_batches"], errors="coerce")
        lag_o = pd.to_numeric(detected_rows["detection_lag_observations"], errors="coerce")

        empty_trace = np.nan if n == 0 else None
        empty_det = np.nan if len(detected_rows) == 0 else None
        summary = {
            "n_observations": n,
            "n_detected": int(observed.sum()),
            "mean_detection_probability": empty_trace if n == 0 else float(prob.mean()),
            "median_detection_probability": empty_trace if n == 0 else float(prob.median()),
            "mean_detection_lag_batches": empty_det if empty_det is not None else float(lag_b.mean()),
            "median_detection_lag_batches": empty_det if empty_det is not None else float(lag_b.median()),
            "mean_detection_lag_observations": empty_det if empty_det is not None else float(lag_o.mean()),
            "median_detection_lag_observations": empty_det if empty_det is not None else float(lag_o.median()),
            "zero_lag_rate": empty_det if empty_det is not None else float((lag_b == 0).mean()),
        }

        result = {"summary": summary, "by_observation": detection}

        if probability_threshold is not None:
            result["threshold"] = probability_threshold
            result["threshold_detection"] = (prob >= probability_threshold).to_numpy()

        if reference is not None:
            reference = np.asarray(reference, dtype=bool)
            hard = HarEval()

            final_detection = np.zeros(len(reference), dtype=bool)
            idx = detection.loc[observed, "idx"].to_numpy()
            idx = idx[(idx >= 0) & (idx < len(reference))].astype(int)
            final_detection[idx] = True
            result["hard_metrics"] = hard.evaluate(final_detection, reference)

            if probability_threshold is not None:
                dp_detection = np.zeros(len(reference), dtype=bool)
                dp_idx = detection.loc[
                    pd.to_numeric(detection["detection_probability"], errors="coerce") >= probability_threshold,
                    "idx",
                ].to_numpy()
                dp_idx = dp_idx[(dp_idx >= 0) & (dp_idx < len(reference))].astype(int)
                dp_detection[dp_idx] = True
                result["threshold_hard_metrics"] = hard.evaluate(dp_detection, reference)

        return result


def _detector_names(detectors):
    if isinstance(detectors, dict):
        if not detectors:
            raise ValueError("detectors must be a non-empty collection.")
        return list(detectors.keys()), list(detectors.values())
    detectors = list(detectors)
    if not detectors:
        raise ValueError("detectors must be a non-empty collection.")
    return [type(d).__name__ for d in detectors], detectors


def har_stream_experiment(
    detectors,
    source_factory,
    warmup_grid,
    batch_grid,
    memory_grid,
    executor=None,
    reference=None,
    mode: str = "auto",
    **kwargs,
) -> dict:
    """Run a grid of online streaming configurations and summarize them.

    Combines each detector with every (warm-up, batch, memory) setting, running a
    fresh session per cell (``source_factory`` must return a new source each
    time). Mirrors R ``har_stream_experiment``.

    Returns
    -------
    dict
        ``{"runs": [...], "summary": DataFrame}`` -- the raw runs (session, trace,
        evaluation) and a compact comparison table.
    """
    if not callable(source_factory):
        raise TypeError("source_factory must be callable.")
    memory_grid = list(memory_grid)
    if not memory_grid:
        raise ValueError("memory_grid must be a non-empty list.")
    names, detector_list = _detector_names(detectors)
    executor = executor if executor is not None else HarOnlineRefitFull()
    evaluator = HarStreamEval()

    runs = []
    summary_rows = []
    for name, detector in zip(names, detector_list):
        for warmup_size in np.atleast_1d(warmup_grid):
            for batch_size in np.atleast_1d(batch_grid):
                for memory in memory_grid:
                    session = HarOnlineSession(
                        source=source_factory(),
                        detector=detector,
                        executor=executor,
                        warmup_size=int(warmup_size),
                        batch_size=int(batch_size),
                        memory=memory,
                        mode=mode,
                    )
                    session.fit()
                    session.run(**kwargs)
                    trace = session.collect_trace()
                    evaluation = evaluator.evaluate(trace, reference=reference)
                    runs.append({
                        "detector_name": name,
                        "warmup_size": int(warmup_size),
                        "batch_size": int(batch_size),
                        "memory": memory,
                        "session": session,
                        "trace": trace,
                        "evaluation": evaluation,
                    })
                    summary_rows.append(
                        _experiment_row(name, int(warmup_size), int(batch_size), memory,
                                        evaluation, session.collect_batch_log())
                    )

    summary = pd.DataFrame(summary_rows)
    return {"runs": runs, "summary": summary}


def _experiment_row(detector_name, warmup_size, batch_size, memory, evaluation, batch_log) -> dict:
    hard = evaluation.get("hard_metrics")
    s = evaluation["summary"]
    return {
        "detector": detector_name,
        "warmup_size": warmup_size,
        "batch_size": batch_size,
        "memory": har_memory_label(memory),
        "n_batches": len(batch_log),
        "mean_detection_probability": s["mean_detection_probability"],
        "median_detection_probability": s["median_detection_probability"],
        "mean_detection_lag_batches": s["mean_detection_lag_batches"],
        "median_detection_lag_batches": s["median_detection_lag_batches"],
        "mean_batch_time_sec": np.nan if len(batch_log) == 0 else float(batch_log["total_time_sec"].mean()),
        "accuracy": np.nan if hard is None else hard["accuracy"],
        "precision": np.nan if hard is None else hard["precision"],
        "recall": np.nan if hard is None else hard["recall"],
        "F1": np.nan if hard is None else hard["F1"],
    }
