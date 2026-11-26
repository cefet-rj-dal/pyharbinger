---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Online / streaming detection

The online layer drives any offline Harbinger detector over a stream. It handles
ingestion, batching, a memory policy, and an execution strategy, and records a
trace that yields Detection Probability (DP) and Detection Lag (DL).

```python
from pyharbinger import (
    HarOnlineSession,
    HarSourceSimulated,
    HarMemoryFull,
    HarOnlineRefitFull,
    HcpPageHinkley,
)

serie = [10, 11, 12, 20, 12, 11, 10, 9, 30, 31, 32, 10, 11, 12, 13, 40, 12, 11, 10, 9]
```

## Pull mode (a replay source)

```python
session = HarOnlineSession(
    source=HarSourceSimulated(serie),
    detector=HcpPageHinkley(min_instances=3, threshold=1),
    warmup_size=3,
    batch_size=2,
    memory=HarMemoryFull(),
    executor=HarOnlineRefitFull(),
)
session.fit()
session.run()

det = session.collect_detection()
print("events at:", det.index[det["event"]].tolist())
det[det["event"]]
```

## The online trace (DP / DL)

```python
trace = session.collect_trace()
trace.head()
```

## Push mode (ingest observations)

```python
from pyharbinger import HarOnlineSession

push = HarOnlineSession(
    source=None,
    detector=HcpPageHinkley(min_instances=3, threshold=1),
    warmup_size=3, batch_size=2, mode="push",
)
push.fit()
for value in serie:
    push.ingest(value)
push.run()
push.collect_detection().index[push.collect_detection()["event"]].tolist()
```

## Live streams (Kafka) without a broker dependency

`HarSourceKafka` holds the broker configuration but delegates the actual
consumption to a **collector** (`get_next()`), so `pyharbinger` ships *no* Kafka
dependency. Attach a `HarSimulatedCollector` to test end-to-end, or subclass
`HarStreamCollector` with your own Kafka client to read a live topic.

```python
from pyharbinger import HarSourceKafka, HarSimulatedCollector

source = HarSourceKafka(
    topic="sensor-events",
    bootstrap_servers=["broker:9092"],
    group_id="harbinger",
    collector=HarSimulatedCollector(serie),  # swap for a real-broker collector
)
session = HarOnlineSession(source=source, detector=HcpPageHinkley(min_instances=3, threshold=1),
                           warmup_size=3, batch_size=2)
session.fit()
session.run()
session.collect_detection().index[session.collect_detection()["event"]].tolist()
```

To consume a real topic, write a tiny collector and plug it in (the Kafka client
lives in *your* code, never in `pyharbinger`):

```python
from pyharbinger import HarStreamCollector

class KafkaCollector(HarStreamCollector):
    def __init__(self, consumer):
        self._consumer = consumer       # e.g. a kafka-python KafkaConsumer

    def get_next(self):
        for message in self._consumer:  # one message at a time
            return float(message.value)
        return None                     # stream exhausted
```

> **Memory policies:** `HarMemoryFull`, `HarMemorySliding(batches)`,
> `HarMemoryLastObservations(n)` bound the in-memory context.
> **Execution strategies:** `HarOnlineRefitFull`, `HarOnlineDetectOnly`,
> `HarOnlineIncremental`. Positions are 0-based (the R original is 1-based).
