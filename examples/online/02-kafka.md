# Streaming from Kafka without a broker dependency

`HarSourceKafka` holds the broker configuration but delegates the *actual*
consumption to a **collector** (`get_next()`). `pyharbinger` therefore declares
**no Kafka dependency**: you attach a collector, and the Kafka client (if any)
lives in your code. This notebook shows two collectors:

1. a **stub** collector (`HarSimulatedCollector`) that replays data in-memory;
2. a **real** collector, built exactly as you would for a live broker, run
   standalone here against a drop-in consumer with the real `kafka-python` API.


```python
from pyharbinger import (
    HarOnlineSession,
    HarSourceKafka,
    HarStreamCollector,
    HarSimulatedCollector,
    HcpPageHinkley,
    HarOnlineRefitFull,
)

serie = [10, 11, 12, 20, 12, 11, 10, 9, 30, 31, 32, 10, 11, 12, 13, 40, 12, 11, 10, 9]


def run(source):
    session = HarOnlineSession(
        source=source,
        detector=HcpPageHinkley(min_instances=3, threshold=1),
        warmup_size=3, batch_size=2, executor=HarOnlineRefitFull(),
    )
    session.fit()
    session.run()
    det = session.collect_detection()
    return det.index[det["event"]].tolist()
```

## 1. Stub example (simulated collector, no broker)

The stub consumes the stream one observation at a time, exactly like a live
consumer, but from memory -- so the whole pipeline is testable anywhere.


```python
stub_source = HarSourceKafka(
    topic="sensor-events",
    bootstrap_servers=["broker:9092"],
    group_id="harbinger",
    collector=HarSimulatedCollector(serie),
)
print("stub events:", run(stub_source))
```

    stub events: [2, 8, 15]


## 2. Real example (a genuine Kafka collector, run standalone)

This is the collector you would ship to production: it reads one record at a
time from a `kafka-python` `KafkaConsumer` (which is an iterator of messages
whose payload is `message.value`). The code below is the real thing.


```python
class KafkaCollector(HarStreamCollector):
    """Pulls one record at a time from a kafka-python KafkaConsumer.

    `consumer` is any iterator of messages exposing a `.value` payload -- the
    exact contract of kafka-python's KafkaConsumer.
    """

    def __init__(self, consumer, decode=float):
        self._messages = iter(consumer)
        self._decode = decode

    def get_next(self):
        message = next(self._messages, None)
        if message is None:
            return None
        return self._decode(message.value)

    def close(self):
        close = getattr(self._messages, "close", None)
        if callable(close):
            close()
```

To run it **standalone**, we feed `KafkaCollector` a tiny in-process consumer
that mimics the `kafka-python` message/iterator API byte-for-byte, so the real
collector code path is exercised without a broker:


```python
class _Message:
    def __init__(self, value):
        self.value = value


class FakeKafkaConsumer:
    """Minimal stand-in with the kafka-python consumer interface (iterable of messages)."""

    def __init__(self, values):
        self._messages = [_Message(v) for v in values]

    def __iter__(self):
        return iter(self._messages)

    def close(self):
        pass


real_source = HarSourceKafka(
    topic="sensor-events",
    bootstrap_servers=["broker:9092"],
    group_id="harbinger",
    collector=KafkaCollector(FakeKafkaConsumer(serie)),
)
print("real-collector events:", run(real_source))
```

    real-collector events: [2, 8, 15]


Both collectors drive the same detections, proving the abstraction is faithful:


```python
assert run(HarSourceKafka("t", ["b:9092"], "g", collector=HarSimulatedCollector(serie))) == \
       run(HarSourceKafka("t", ["b:9092"], "g", collector=KafkaCollector(FakeKafkaConsumer(serie))))
print("stub == real-collector: OK")
```

    stub == real-collector: OK


### Pointing the same collector at a live broker

In production you build a real `KafkaConsumer` and hand it to the *same*
`KafkaCollector` -- nothing else changes. The import stays in your code, never
in `pyharbinger`:


```python
# from kafka import KafkaConsumer  # pip install kafka-python
#
# consumer = KafkaConsumer(
#     "sensor-events",
#     bootstrap_servers=["broker:9092"],
#     group_id="harbinger",
#     value_deserializer=lambda b: float(b.decode()),
#     consumer_timeout_ms=1000,  # so iteration ends instead of blocking forever
# )
# source = HarSourceKafka("sensor-events", ["broker:9092"], "harbinger",
#                         collector=KafkaCollector(consumer, decode=lambda v: v))
# session = HarOnlineSession(source=source, detector=HcpPageHinkley(), batch_size=50)
# session.fit(); session.run()
```
