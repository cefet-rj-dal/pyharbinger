"""Tests for the k-means clustering-based detector."""
from __future__ import annotations

import numpy as np

from pyharbinger import HanctKmeans, datasets

SIMPLE = datasets.examples_anomalies()["simple"]["serie"].to_numpy()


def _events(detection):
    return (np.where(detection["event"].to_numpy())[0] + 1).tolist()


def test_kmeans_default_centers():
    model = HanctKmeans(random_state=0).fit(SIMPLE)
    # ceil(log10(101)) = 3
    assert model.centers == 3


def test_kmeans_flags_anomaly_like_r():
    # Matches R hanct_kmeans on examples_anomalies$simple (anomaly at 50).
    det = HanctKmeans(random_state=0).fit(SIMPLE).detect(SIMPLE)
    assert _events(det) == [50]


def test_kmeans_reproducible_with_seed():
    a = _events(HanctKmeans(random_state=1).fit(SIMPLE).detect(SIMPLE))
    b = _events(HanctKmeans(random_state=1).fit(SIMPLE).detect(SIMPLE))
    assert a == b


def test_kmeans_discord_columns_for_seq_gt_1():
    det = HanctKmeans(seq=4, random_state=0).fit(SIMPLE).detect(SIMPLE)
    assert "seq" in det.columns and "seqlen" in det.columns
    # Any flagged subsequence is a discord, not an anomaly.
    assert (det["type"][det["event"]] == "discord").all()


def test_kmeans_warmup_not_flagged_for_seq_gt_1():
    det = HanctKmeans(seq=5, random_state=0).fit(SIMPLE).detect(SIMPLE)
    assert not det["event"].to_numpy()[:4].any()
