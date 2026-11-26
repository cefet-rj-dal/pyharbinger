import numpy as np
import pandas as pd
import pytest

from pyharbinger import (
    HanrFbiad,
    HanrFft,
    HanrHistogram,
    Harbinger,
    HarEnsemble,
    HarEnsembleFuzzy,
    datasets,
)


class StaticDetector(Harbinger):
    def __init__(self, positions, event_type="changepoint"):
        super().__init__()
        self.positions = positions
        self.event_type = event_type

    def detect(self, serie, **kwargs):
        event = np.zeros(len(serie), dtype=bool)
        event[self.positions] = True
        types = np.where(event, self.event_type, "")
        return pd.DataFrame({"idx": np.arange(len(event)), "event": event, "type": types})


def test_percentile_floor_allows_empty_detection_under_weak_consensus():
    model = HarEnsembleFuzzy(
        StaticDetector([2]),
        StaticDetector([]),
        StaticDetector([]),
        threshold_type="percentile",
        threshold=0.9,
        min_consensus=0.5,
    )
    detection = model.detect(np.arange(5.0))
    assert not detection["event"].any()
    assert detection.attrs["effective_threshold"] == 0.5


def test_percentile_floor_retains_sufficient_consensus():
    model = HarEnsembleFuzzy(
        StaticDetector([2]),
        StaticDetector([2]),
        StaticDetector([]),
        threshold_type="percentile",
        threshold=0.9,
        min_consensus=0.5,
    )
    detection = model.detect(np.arange(5.0))
    assert detection.loc[2, "event"]
    assert np.isclose(detection.attrs["effective_threshold"], 2 / 3)


def test_fixed_threshold_is_unchanged_by_consensus_floor():
    model = HarEnsembleFuzzy(
        StaticDetector([2]),
        StaticDetector([]),
        StaticDetector([]),
        threshold_type="fixed",
        threshold=0.3,
        min_consensus=0.9,
    )
    detection = model.detect(np.arange(5.0))
    assert detection.loc[2, "event"]
    assert detection.attrs["effective_threshold"] == 0.3


# --- HarEnsemble (majority vote), faithful port of R har_ensemble -------------

def _events(detection):
    return detection.loc[detection["event"], "idx"].tolist()


# R har_ensemble(hanr_fft(), hanr_fbiad(), hanr_histogram()) on examples_anomalies
# (0-based detected indices), verified via Rscript sourcing harbinger base + members.
R_ENSEMBLE = {
    "simple": [49],
    "contextual": [],
    "tt": [11, 23, 37, 49, 89, 100],
    "volatile": [709, 749, 952],
}


@pytest.mark.parametrize("name", list(R_ENSEMBLE))
def test_har_ensemble_majority_vote_matches_r(name):
    serie = datasets.examples_anomalies()[name]["serie"].to_numpy()
    model = HarEnsemble(HanrFft(), HanrFbiad(), HanrHistogram()).fit(serie)
    detection = model.detect(serie)
    assert _events(detection) == R_ENSEMBLE[name]


def test_har_ensemble_type_priority_and_threshold():
    # 3 detectors: a changepoint vote takes priority; a lone vote loses the majority.
    model = HarEnsemble(
        StaticDetector([2], event_type="changepoint"),
        StaticDetector([2], event_type="anomaly"),
        StaticDetector([4], event_type="anomaly"),
    )
    det = model.detect(np.arange(6.0))
    assert _events(det) == [2]  # position 4 has a single vote -> below majority
    assert det.loc[2, "type"] == "changepoint"


def test_har_ensemble_requires_a_detector():
    with pytest.raises(ValueError):
        HarEnsemble()


def test_har_ensemble_detect_none_raises():
    with pytest.raises(ValueError):
        HarEnsemble(StaticDetector([1])).detect(None)
