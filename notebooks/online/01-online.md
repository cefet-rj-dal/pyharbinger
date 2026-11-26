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

> **Memory policies:** `HarMemoryFull`, `HarMemorySliding(batches)`,
> `HarMemoryLastObservations(n)` bound the in-memory context.
> **Execution strategies:** `HarOnlineRefitFull`, `HarOnlineDetectOnly`,
> `HarOnlineIncremental`. Positions are 0-based (the R original is 1-based).
