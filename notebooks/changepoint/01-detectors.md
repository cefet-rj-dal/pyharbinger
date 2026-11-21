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

# Change-point detectors

Change-point detectors flag the positions where the statistical regime of the
series changes (a shift in mean and/or variance). `pyharbinger` offers two
families:

* **offline, cost-based** (`HcpAmoc`, `HcpBinseg`, `HcpPelt`) — wrap the
  `ruptures` library with a Gaussian cost;
* **online, streaming** (`HcpKswin`, `HcpPageHinkley`) — scan the series once.

> **Note on parity:** the R package wraps `changepoint::cpt.meanvar` with the
> MBIC penalty, which `ruptures` does not reproduce exactly, so the offline
> detectors agree with R only approximately (compare breakpoint positions).
> `HcpPageHinkley` is deterministic and matches R exactly.

```python
import numpy as np
from pyharbinger import (
    HcpAmoc, HcpBinseg, HcpPelt, HcpKswin, HcpPageHinkley, datasets,
)

dataset = datasets.examples_changepoints()["abrupt"]
serie, event = dataset["serie"], dataset["event"]
print("ground-truth change points at:", dataset.index[event].tolist())
```

## HcpAmoc — at most one change

```python
det = HcpAmoc().detect(serie)
print("change point at:", det.index[det["event"]].tolist())
```

## HcpBinseg — binary segmentation

```python
det = HcpBinseg(Q=2).detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

## HcpPelt — PELT

The `penalty` argument controls granularity (higher = fewer change points).

```python
det = HcpPelt().detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

## HcpPageHinkley — online cumulative statistic

```python
det = HcpPageHinkley().detect(serie)
print("change point at:", det.index[det["event"]].tolist())
```

## HcpKswin — Kolmogorov-Smirnov windowing

Uses a random subsample, so fix `random_state` for reproducibility.

```python
det = HcpKswin(random_state=0).detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

## ChangeFinder and regression-based detectors

These additional detectors are **native** (no `ruptures`): they are faithful
ports that match R exactly on the example series.

* `HcpCfLr` — ChangeFinder with a two-stage linear-regression residual model.
* `HcpScp` — the "seminal" change point: compares a whole-window regression
  against two half-window regressions split around each window's centre.
* `HcpJoinpoint` — Joinpoint Regression++ (dynamic programming + weighted BIC).

```python
from pyharbinger import HcpCfLr, HcpScp, HcpJoinpoint

for name, det_model in [
    ("cf_lr", HcpCfLr()),
    ("scp", HcpScp()),
    ("joinpoint", HcpJoinpoint()),
]:
    det = det_model.detect(serie)
    print(f"{name:10s} change points at: {det.index[det['event']].tolist()}")
```

> **Parity note.** `HcpCfLr` and `HcpScp` match R exactly on every example
> series. `HcpJoinpoint` matches R on well-conditioned series; on a perfectly
> piecewise-linear series the segment residuals collapse to round-off, so the
> model selection is numerically ill-conditioned and the exact break positions
> may differ from R (the dominant change point is still recovered).
