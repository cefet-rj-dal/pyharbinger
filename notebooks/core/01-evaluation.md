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

# The Harbinger contract and event evaluation

Every detector in `pyharbinger` extends the `Harbinger` base class and shares one
interface: `fit` → `detect` → `evaluate`. A `detect` call returns a table with
one row per observation and the columns `idx` (0-based position), `event`
(a detected event) and `type` (`"anomaly"`, `"changepoint"`, ... or `""`).

Detections are scored against a ground-truth boolean vector with a common
evaluation protocol:

* `HarEval` — **hard** evaluation: a detection counts only where it coincides
  exactly with an event.
* `HarEvalSoft` — **SoftED**: detections within a tolerance window of an event
  get partial credit (matched with the Hungarian algorithm).

**References:** Salles et al. *Harbinger* (SBBD 2020); Salles et al. *SoftED:
Metrics for soft evaluation of time series event detection* (CIE 2024),
doi:10.1016/j.cie.2024.110728.

```python
import numpy as np
from pyharbinger import Harbinger, HarEval, HarEvalSoft
```

## The base contract

The base `Harbinger` is a valid (trivial) detector: it flags no events. Concrete
detectors override `detect` while keeping the same output shape.

```python
base = Harbinger()
detection = base.detect(np.sin(np.linspace(0, 10, 12)))
detection
```

## Hard evaluation

We compare a detection vector against the ground-truth events.

```python
detection = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
event     = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=bool)

hard = HarEval().evaluate(detection, event)
print("confusion matrix:")
print(hard["confMatrix"])
print(f"\nprecision={hard['precision']:.3f}  recall={hard['recall']:.3f}  F1={hard['F1']:.3f}")
```

## Soft evaluation (SoftED)

With a tolerance window, detections *near* an event earn partial credit, so a
detection one step off the true event is no longer a complete miss.

```python
soft = HarEvalSoft(sw_size=3).evaluate(detection, event)
print(f"TPs={soft['TP']:.3f}  FPs={soft['FP']:.3f}  FNs={soft['FN']:.3f}")
print(f"precision={soft['precision']:.3f}  recall={soft['recall']:.3f}  F1={soft['F1']:.3f}")
```
