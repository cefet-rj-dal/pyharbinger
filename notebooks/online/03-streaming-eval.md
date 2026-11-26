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

# Streaming evaluation and experiments

`HarStreamEval` scores the online trace with Detection Probability (DP) and
Detection Lag (DL), and adds the standard hard metrics when ground-truth labels
are available. `har_stream_experiment` sweeps a grid of online configurations
and returns a comparison table.

```python
import numpy as np
from pyharbinger import (
    HarOnlineSession, HarSourceSimulated, HcpPageHinkley,
    HarStreamEval, HarMemoryFull, har_stream_experiment,
)

serie = [10, 11, 12, 20, 12, 11, 10, 9, 30, 31, 32, 10, 11, 12, 13, 40, 12, 11, 10, 9]
reference = np.zeros(len(serie), dtype=bool)
reference[[3, 8, 15]] = True
```

## Evaluate a trace

```python
session = HarOnlineSession(
    source=HarSourceSimulated(serie),
    detector=HcpPageHinkley(min_instances=3, threshold=1),
    warmup_size=3, batch_size=2,
)
session.fit()
session.run()

ev = HarStreamEval().evaluate(session.collect_trace(), reference=reference,
                              probability_threshold=0.5)
ev["summary"]
```

```python
ev["hard_metrics"]["F1"], ev["threshold_hard_metrics"]["F1"]
```

## Sweep a grid of configurations

```python
experiment = har_stream_experiment(
    detectors={"page": HcpPageHinkley(min_instances=3, threshold=1)},
    source_factory=lambda: HarSourceSimulated(serie),  # a fresh source per run
    warmup_grid=3,
    batch_grid=[1, 2],
    memory_grid=[HarMemoryFull()],
    reference=reference,
)
experiment["summary"]
```

> **Parity:** `HarStreamEval` and `har_stream_experiment` reproduce the R
> `har_stream_eval` / `har_stream_experiment` exactly (trace summaries, hard
> metrics, and the experiment table), since the underlying trace and `HarEval`
> already match R.
