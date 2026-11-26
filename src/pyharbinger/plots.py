"""Plotting helpers for Harbinger detections (matplotlib).

Python port of the R ``har_plot`` / ``har_ensemble_plot`` /
``har_ensemble_plot_models`` helpers. The R originals build ``ggplot2`` /
``patchwork`` figures; this port uses ``matplotlib`` (a core dependency).

These are visualization helpers with no numeric-parity gate: they render the
same information (true/false positives, false negatives, change points, motif
segments, ensemble scores and per-model votes) using the same colour coding.

References
----------
- Ogasawara E, Salles R, Porto F, Pacitti E (2025). Event Detection in Time
  Series. Springer. doi:10.1007/978-3-031-75941-3

Porting
-------
Mirrors ``har_plot.R`` / ``har_plot_ensemble.R``. Fidelity: adapted (matplotlib
rather than ggplot2; same marks and colour semantics).
"""
from __future__ import annotations

import numpy as np

from .harbinger import as_series_array

__all__ = ["har_ensemble_plot", "har_ensemble_plot_models", "har_plot"]

_DEFAULT_COLORS = ("green", "blue", "red", "purple")  # TP, FN, FP, motif


def _detection_attr(detection, name):
    """Read a value stored on the detection's ``attrs`` (score / threshold / ...)."""
    if detection is None:
        return None
    attrs = getattr(detection, "attrs", {}) or {}
    return attrs.get(name)


def _motif_mask(detection, n: int):
    """Expand motif/discord ``seq``/``seqlen`` into a per-observation boolean mask.

    Mirrors the R ``motifs_seqs`` helper: every position carrying a non-missing
    ``seq`` starts a run of ``seqlen`` consecutive marked positions.
    """
    if detection is None or "seq" not in detection or "seqlen" not in detection:
        return None
    seq = detection["seq"].to_numpy(dtype=object)
    seqlen = detection["seqlen"].to_numpy()
    present = np.array([s is not None and not (isinstance(s, float) and np.isnan(s)) for s in seq])
    starts = np.flatnonzero(present)
    if starts.size == 0:
        return np.zeros(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for start in starts:
        length = seqlen[start]
        if not np.isfinite(length):
            continue
        end = min(n, start + int(length))
        mask[start:end] = True
    return mask


def har_plot(
    obj,
    serie,
    detection=None,
    event=None,
    *,
    mark_cp: bool = True,
    ylim=None,
    idx=None,
    pointsize: float = 0.5,
    colors=_DEFAULT_COLORS,
    yline=None,
    ax=None,
):
    """Plot a series with detected events highlighted.

    Points are coloured as true positives (``colors[0]``), false negatives
    (``colors[1]``), false positives (``colors[2]``) and motif/discord segments
    (``colors[3]``). Detected change points are drawn as dashed vertical lines
    when ``mark_cp`` is set; ``yline`` draws dotted horizontal reference lines.

    Parameters mirror R ``har_plot``. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    serie = as_series_array(serie)
    n = len(serie)
    if idx is None:
        idx = np.arange(n)
    idx = np.asarray(idx)

    if detection is not None:
        det_event = detection["event"].fillna(False).to_numpy(dtype=bool)
        cp = (detection["type"] == "changepoint").to_numpy(dtype=bool)
    else:
        det_event = np.zeros(n, dtype=bool)
        cp = np.zeros(n, dtype=bool)

    tp = np.zeros(n, dtype=bool)
    fn = np.zeros(n, dtype=bool)
    fp = det_event.copy()
    if event is not None:
        event = np.asarray(event, dtype=bool)
        tp = det_event & (event == det_event)
        fp = det_event & (event != det_event)
        fn = event & (event != det_event)

    color = np.full(n, "black", dtype=object)
    size = np.full(n, float(pointsize))

    motif_mask = _motif_mask(detection, n)
    if motif_mask is not None:
        size[motif_mask] = 0.75
        color[motif_mask] = colors[3]
    color[tp] = colors[0]
    color[fn] = colors[1]
    color[fp] = colors[2]
    size[fn | tp | fp] = 1.5

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(idx, serie, color="black", linewidth=0.7, zorder=1)
    # matplotlib marker area scales with size**2; amplify the R linewidth-like sizes.
    ax.scatter(idx, serie, c=list(color), s=(size * 6) ** 2, zorder=2)

    min_data, max_data = float(np.min(serie)), float(np.max(serie))
    if ylim is not None:
        lo, hi = ylim
        min_data = min_data if lo is None else lo
        max_data = max_data if hi is None else hi
        span = max_data - min_data
        ax.set_ylim(min_data - span * 0.05, max_data + span * 0.05)
    span = max_data - min_data
    top_1 = max_data + span * 0.02
    bottom_1 = min_data - span * 0.02

    if mark_cp and cp.any():
        for x in idx[cp]:
            ax.plot([x, x], [bottom_1, top_1], color="grey", linestyle="--", linewidth=1, zorder=0)

    if yline is not None:
        for y in np.atleast_1d(yline):
            ax.axhline(y=y, color="lightgrey", linestyle=":", linewidth=1, zorder=0)

    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax


def har_ensemble_plot(detection, serie, threshold=None, time_idx=None, fig=None):
    """Two-panel view of an ensemble detection: consensus score + detected events.

    Uses the ``score`` and ``threshold`` stored on the detection (as produced by
    :class:`~pyharbinger.HarEnsembleFuzzy`). Mirrors R ``har_ensemble_plot``;
    returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    serie = as_series_array(serie)
    n = len(serie)
    if time_idx is None:
        time_idx = np.arange(n)
    time_idx = np.asarray(time_idx)

    event = (
        detection["event"].fillna(False).to_numpy(dtype=bool)
        if detection is not None and len(detection) == n
        else np.zeros(n, dtype=bool)
    )
    score = _detection_attr(detection, "score")
    score = np.zeros(n) if score is None or len(score) != n else np.clip(np.nan_to_num(score), 0, 1)
    if threshold is None:
        threshold = _detection_attr(detection, "threshold")
    if not np.isscalar(threshold) or threshold is None or not np.isfinite(threshold):
        threshold = 0.5

    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    else:
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

    ax1.plot(time_idx, score, color="black", linewidth=0.5)
    ax1.axhline(y=threshold, color="red", linestyle="--", linewidth=0.5)
    ax1.set_title("Ensemble Score")

    ax2.plot(time_idx, serie, color="black", linewidth=0.5)
    ax2.scatter(time_idx, serie, color="black", s=4)
    if event.any():
        ax2.scatter(time_idx[event], serie[event], color="red", s=36, zorder=3)
        for x in time_idx[event]:
            ax2.axvline(x=x, color="red", linestyle="--", linewidth=0.5)
    ax2.set_title("Detected Events")
    fig.tight_layout()
    return fig


def har_ensemble_plot_models(detection, serie, time_idx=None, fig=None):
    """Stacked view of the original series and each member detector's votes.

    Reads the ``model_events`` stored on the detection (as produced by
    :class:`~pyharbinger.HarEnsembleFuzzy`). Mirrors R
    ``har_ensemble_plot_models``; returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    serie = as_series_array(serie)
    n = len(serie)
    if time_idx is None:
        time_idx = np.arange(n)
    time_idx = np.asarray(time_idx)

    model_events = _detection_attr(detection, "model_events") or {}
    names = list(model_events)
    nrows = 1 + len(names)
    heights = [6] + [1] * len(names)

    if fig is None:
        fig, axes = plt.subplots(nrows, 1, sharex=True, gridspec_kw={"height_ratios": heights})
    else:
        axes = fig.subplots(nrows, 1, sharex=True, gridspec_kw={"height_ratios": heights})
    axes = np.atleast_1d(axes)

    axes[0].plot(time_idx, serie, color="black", linewidth=0.5)
    axes[0].scatter(time_idx, serie, color="black", s=4)
    axes[0].set_title("Original Series")

    for i, name in enumerate(names):
        ax = axes[i + 1]
        raw = model_events[name].get("raw")
        event = (
            np.asarray(raw, dtype=bool)
            if raw is not None and len(raw) == n
            else np.zeros(n, dtype=bool)
        )
        ax.plot(time_idx, np.zeros(n), color="black", linewidth=0.7)
        if event.any():
            ax.scatter(time_idx[event], np.zeros(event.sum()), color="red", s=36, zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)

    fig.tight_layout()
    return fig
