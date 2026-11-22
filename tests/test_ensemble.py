import numpy as np
import pandas as pd

from pyharbinger import Harbinger, HarEnsembleFuzzy


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
