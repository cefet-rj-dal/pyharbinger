"""Tests for the online / streaming layer.

Faithful port of the R online framework: the gate is exact parity of the
collected detection (events, DP, DL) and trace (batch/detection frequency)
against the R ``har_online_session`` on a frozen series, across the memory
policies, execution strategies, and both pull and push modes. Positions are
0-based here vs R's 1-based (R events 3,9,16 -> 2,8,15).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pyharbinger import (
    HarMemoryFull,
    HarMemoryLastObservations,
    HarMemorySliding,
    HarOnlineDetectOnly,
    HarOnlineRefitFull,
    HarOnlineSession,
    HarSimulatedCollector,
    HarSourceCallback,
    HarSourceDataframe,
    HarSourceKafka,
    HarSourceSimulated,
    HarStreamCollector,
    HcpPageHinkley,
    har_memory_label,
)

SERIE = [10, 11, 12, 20, 12, 11, 10, 9, 30, 31, 32, 10, 11, 12, 13, 40, 12, 11, 10, 9]


def _detector():
    return HcpPageHinkley(min_instances=3, threshold=1)


def _run(memory, executor, push=False):
    source = None if push else HarSourceSimulated(SERIE)
    mode = "push" if push else "pull"
    session = HarOnlineSession(
        source=source, detector=_detector(), warmup_size=3, batch_size=2,
        memory=memory, executor=executor, mode=mode,
    )
    session.fit()
    if push:
        for value in SERIE:
            session.ingest(value)
    session.run()
    return session


def _events(det):
    return det.loc[det["event"], "idx"].tolist()


# --- parity vs R -------------------------------------------------------------

def test_full_refit_matches_r():
    session = _run(HarMemoryFull(), HarOnlineRefitFull())
    det = session.collect_detection()
    assert _events(det) == [2, 8, 15]  # R 1-based 3,9,16
    assert det.loc[det["event"], "detection_probability"].tolist() == [1.0, 1.0, 1.0]
    assert det.loc[det["event"], "detection_lag_batches"].tolist() == [0, 0, 0]
    assert det.loc[det["event"], "detection_lag_observations"].tolist() == [2, 2, 2]
    trace = session.collect_trace()
    assert trace["batch_frequency"].tolist() == [9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1]
    assert trace["detection_frequency"].tolist() == [0, 0, 9, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    assert len(session.collect_batch_log()) == 9


def test_sliding_memory_matches_r():
    det = _run(HarMemorySliding(2), HarOnlineRefitFull()).collect_detection()
    assert _events(det) == [2, 8, 13, 15]
    probs = [round(p, 3) for p in det.loc[det["event"], "detection_probability"].tolist()]
    assert probs == [1.0, 1.0, 0.333, 0.667]


def test_last_observations_memory_matches_r():
    det = _run(HarMemoryLastObservations(4), HarOnlineRefitFull()).collect_detection()
    assert _events(det) == [2, 3, 8, 13, 15]
    probs = [round(p, 3) for p in det.loc[det["event"], "detection_probability"].tolist()]
    assert probs == [0.5, 0.333, 1.0, 0.333, 0.667]


def test_detect_only_matches_r():
    det = _run(HarMemoryFull(), HarOnlineDetectOnly(fit_on_warmup=True)).collect_detection()
    assert _events(det) == [2, 8, 15]


def test_push_mode_matches_r():
    det = _run(HarMemoryFull(), HarOnlineRefitFull(), push=True).collect_detection()
    assert _events(det) == [2, 8, 15]
    trace = _run(HarMemoryFull(), HarOnlineRefitFull(), push=True).collect_trace()
    assert trace["batch_frequency"].tolist()[-3:] == [1, 1, 0]


# --- sources & misc ----------------------------------------------------------

def test_source_simulated_exhausts():
    src = HarSourceSimulated([1.0, 2.0])
    assert src.next_observation()["value"] == 1.0
    assert src.next_observation()["idx"] == 1
    assert src.next_observation() is None


def test_source_dataframe_selects_columns():
    df = pd.DataFrame({"ts": [1, 2, 3], "value": [10.0, 11.0, 9.0]})
    src = HarSourceDataframe(df, timestamp_col="ts", value_cols="value")
    obs = src.next_observation()
    assert obs["value"] == 10.0
    assert obs["timestamp"] == 1
    assert src.info()["value_cols"] == ["value"]


def test_source_callback_until_none():
    seq = iter([5.0, 6.0])
    src = HarSourceCallback(lambda: next(seq, None))
    assert src.next_observation()["value"] == 5.0
    assert src.next_observation()["value"] == 6.0
    assert src.next_observation() is None


def test_max_batches_limits_cycles():
    session = HarOnlineSession(
        source=HarSourceSimulated(SERIE), detector=_detector(),
        warmup_size=3, batch_size=2, executor=HarOnlineRefitFull(),
    )
    session.fit()
    session.run(max_batches=2)
    assert session.run_count == 2


def test_memory_labels():
    assert har_memory_label(HarMemoryFull()) == "full"
    assert har_memory_label(HarMemorySliding(3)) == "sliding_batches_3"
    assert har_memory_label(HarMemoryLastObservations(100)) == "last_observations_100"


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        HarMemorySliding(0)
    with pytest.raises(ValueError):
        HarOnlineSession(source=None, detector=None)
    with pytest.raises(ValueError):
        HarOnlineSession(source=None, detector=_detector(), batch_size=0)


def test_collect_detection_empty_before_run():
    session = HarOnlineSession(source=HarSourceSimulated(SERIE), detector=_detector())
    det = session.collect_detection()
    assert list(det.columns) == ["idx", "event", "type"]
    assert len(det) == 0


# --- Kafka source (dependency-free abstraction + collector seam) --------------

def test_kafka_source_with_simulated_collector_matches_simulated_source():
    # A Kafka source fed by a simulated collector must behave exactly like the
    # simulated replay source (same gradual consumption -> same detections).
    source = HarSourceKafka("topic", ["broker:9092"], "grp", collector=HarSimulatedCollector(SERIE))
    session = HarOnlineSession(source=source, detector=_detector(), warmup_size=3, batch_size=2)
    session.fit()
    session.run()
    assert _events(session.collect_detection()) == [2, 8, 15]


def test_kafka_source_without_collector_raises_actionable_error():
    source = HarSourceKafka("topic", ["broker:9092"], "grp")
    with pytest.raises(RuntimeError, match="collector"):
        source.next_observation()


def test_kafka_source_accepts_custom_collector_subclass():
    class MyCollector(HarStreamCollector):
        def __init__(self, data):
            self._it = iter(data)

        def get_next(self):
            return next(self._it, None)

    source = HarSourceKafka("topic", ["broker:9092"], "grp", collector=MyCollector(SERIE))
    session = HarOnlineSession(source=source, detector=_detector(), warmup_size=3, batch_size=2)
    session.fit()
    session.run()
    assert _events(session.collect_detection()) == [2, 8, 15]


def test_kafka_source_validates_config_and_collector():
    with pytest.raises(ValueError):
        HarSourceKafka("", ["b"], "g")
    with pytest.raises(ValueError):
        HarSourceKafka("t", [], "g")
    with pytest.raises(TypeError):
        HarSourceKafka("t", ["b"], "g", collector=object())


def test_simulated_collector_exhausts():
    collector = HarSimulatedCollector([1.0, 2.0])
    assert collector.get_next() == 1.0
    assert collector.get_next() == 2.0
    assert collector.get_next() is None


def test_stream_collector_base_is_abstract():
    with pytest.raises(NotImplementedError):
        HarStreamCollector().get_next()


def test_real_collector_over_kafka_like_consumer_matches_stub():
    # A realistic collector reading one message at a time from a kafka-python-like
    # consumer (iterator of messages with a `.value`) must match the stub.
    class _Message:
        def __init__(self, value):
            self.value = value

    class _FakeConsumer:
        def __init__(self, values):
            self._messages = [_Message(v) for v in values]
            self.closed = False

        def __iter__(self):
            return iter(self._messages)

        def close(self):
            self.closed = True

    class KafkaCollector(HarStreamCollector):
        def __init__(self, consumer):
            self._consumer = consumer
            self._messages = iter(consumer)

        def get_next(self):
            message = next(self._messages, None)
            return None if message is None else float(message.value)

        def close(self):
            self._consumer.close()

    consumer = _FakeConsumer(SERIE)
    source = HarSourceKafka("t", ["b:9092"], "g", collector=KafkaCollector(consumer))
    session = HarOnlineSession(source=source, detector=_detector(), warmup_size=3, batch_size=2)
    session.fit()
    session.run()
    assert _events(session.collect_detection()) == [2, 8, 15]
    # Lifecycle: closing the source closes the underlying consumer.
    source.close()
    assert consumer.closed
