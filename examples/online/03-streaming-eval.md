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




    {'n_observations': 20,
     'n_detected': 3,
     'mean_detection_probability': 0.15,
     'median_detection_probability': 0.0,
     'mean_detection_lag_batches': 0.0,
     'median_detection_lag_batches': 0.0,
     'mean_detection_lag_observations': 2.0,
     'median_detection_lag_observations': 2.0,
     'zero_lag_rate': 1.0}




```python
ev["hard_metrics"]["F1"], ev["threshold_hard_metrics"]["F1"]
```




    (0.6666666666666666, 0.6666666666666666)



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




      detector  warmup_size  batch_size memory  n_batches  mean_detection_probability  median_detection_probability  mean_detection_lag_batches  median_detection_lag_batches  mean_batch_time_sec  accuracy  precision    recall        F1
    0     page            3           1   full         17                        0.15                           0.0                         0.0                           0.0             0.000158       0.9   0.666667  0.666667  0.666667
    1     page            3           2   full          9                        0.15                           0.0                         0.0                           0.0             0.000112       0.9   0.666667  0.666667  0.666667



> **Parity:** `HarStreamEval` and `har_stream_experiment` reproduce the R
> `har_stream_eval` / `har_stream_experiment` exactly (trace summaries, hard
> metrics, and the experiment table), since the underlying trace and `HarEval`
> already match R.
