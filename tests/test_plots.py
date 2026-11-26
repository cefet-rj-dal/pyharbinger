"""Smoke tests for the matplotlib plotting helpers.

These are visualization helpers with no numeric-parity gate; the tests check that
the figures build, carry the expected structure (panels, marked points, change
point lines, motif segments), and never require a display.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyharbinger import (
    HanrFbiad,
    HanrFft,
    HanrHistogram,
    HarEnsembleFuzzy,
    HcpBinseg,
    HmoSax,
    datasets,
    har_ensemble_plot,
    har_ensemble_plot_models,
    har_plot,
)

ANOM = datasets.examples_anomalies()
MOTIF = datasets.examples_motifs()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_har_plot_returns_axes_with_series():
    serie = ANOM["simple"]["serie"].to_numpy()
    det = HanrFft().fit(serie).detect(serie)
    ax = har_plot(HanrFft(), serie, det)
    assert isinstance(ax, plt.Axes)
    # one line (the series) plus a scatter collection.
    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 1


def test_har_plot_colors_tp_fp_fn():
    serie = ANOM["simple"]["serie"].to_numpy()
    event = ANOM["simple"]["event"].to_numpy()
    det = HanrFft().fit(serie).detect(serie)
    ax = har_plot(HanrFft(), serie, det, event)
    colors = {tuple(c) for c in ax.collections[0].get_facecolors()}
    # Black plus at least one highlight color (TP/FP/FN) must be present.
    assert len(colors) >= 2


def test_har_plot_marks_change_points():
    serie = ANOM["simple"]["serie"].to_numpy()
    det = HcpBinseg().fit(serie).detect(serie)
    n_cp = int((det["type"] == "changepoint").sum())
    ax = har_plot(HcpBinseg(), serie, det, mark_cp=True)
    # series line + one dashed segment per change point.
    dashed = [ln for ln in ax.lines if ln.get_linestyle() in ("--", "dashed")]
    assert len(dashed) == n_cp


def test_har_plot_motif_segments():
    serie = MOTIF["simple"]["serie"].to_numpy()
    det = HmoSax(26, 3, 3).fit(serie).detect(serie)
    ax = har_plot(HmoSax(26, 3, 3), serie, det)
    assert isinstance(ax, plt.Axes)


def test_har_plot_yline():
    serie = ANOM["simple"]["serie"].to_numpy()
    ax = har_plot(HanrFft(), serie, yline=[0.0, 0.5])
    dotted = [ln for ln in ax.lines if ln.get_linestyle() in (":", "dotted")]
    assert len(dotted) == 2


def test_har_ensemble_plot_two_panels():
    serie = ANOM["simple"]["serie"].to_numpy()
    det = HarEnsembleFuzzy(HanrFft(), HanrFbiad(), HanrHistogram()).fit(serie).detect(serie)
    fig = har_ensemble_plot(det, serie)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_har_ensemble_plot_models_one_row_per_member():
    serie = ANOM["simple"]["serie"].to_numpy()
    det = HarEnsembleFuzzy(HanrFft(), HanrFbiad(), HanrHistogram()).fit(serie).detect(serie)
    fig = har_ensemble_plot_models(det, serie)
    # original series + one row per member detector.
    assert len(fig.axes) == 1 + 3


def test_har_ensemble_plot_defaults_without_attrs():
    # A plain detection (no score attr) should still render with the 0.5 default.
    serie = np.arange(10.0)
    det = HanrFft().fit(serie).detect(serie)
    fig = har_ensemble_plot(det, serie)
    assert isinstance(fig, plt.Figure)
