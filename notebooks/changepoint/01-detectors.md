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

## Structural-change detectors (strucchange replacement, re-engineered "++")

`HcpChow` and `HcpGft` detect **structural breaks**. They began as native ports
of R's `strucchange` backend and were then **re-engineered into "++" versions**:
the original least-squares (mean-only) criterion is generalised to a Gaussian
**mean + variance** segmentation cost, so the detectors also catch
scale/variance-regime changes (e.g. volatility or amplitude shifts) that a
mean-shift model is blind to.

* `HcpChow` — the single break that maximises the mean+variance Gaussian
  log-likelihood ratio (a scale-aware generalisation of Chow's homoskedastic
  sup-F; the break is scanned over the unknown change date à la Andrews 1993).
* `HcpGft` — multiple breaks via a vectorised Bai & Perron dynamic program under
  the mean+variance cost, with the break count chosen by the conservative LWZ
  criterion (Liu, Wu & Zidek 1997) instead of plain BIC.

```python
from pyharbinger import HcpChow, HcpGft

for name, det_model in [("chow", HcpChow()), ("gft", HcpGft())]:
    det = det_model.detect(serie)
    print(f"{name:5s} change points at: {det.index[det['event']].tolist()}")
```

> **Quality note (Action-1++ gate).** Because the "++" versions deliberately
> diverge from `strucchange`, they are no longer judged by R parity but by
> **detection quality vs the ground truth** (`event` column). On the seven
> `examples_changepoints` series the summed SoftED-F1 rises from the faithful
> ports' 3.40 / 2.44 (R `strucchange` 3.31 / 2.20) to **3.73 (chow) / 3.17 (gft)**
> — the mean+variance cost recovers variance/amplitude regimes the original
> mean-shift model misses. See `develop/benchmark_backimpl.py`.

## ChangeFinder with ARIMA / ETS (native forecast replacement)

`HcpCfArima` and `HcpCfEts` are the ChangeFinder detectors that model the
series with a one-step-ahead forecaster and score the residuals. They are
**native re-implementations** that replace R's dependency on the `forecast`
package (`auto.arima` / `ets`) — part of the back-implementation initiative. The
models are fitted with `statsmodels`, but the order/model **selection** (the
part of `forecast` that drives the result) is ours, so parity is under our
control.

* `HcpCfArima` — ChangeFinder with a natively order-selected ARIMA model
  (Hyndman–Khandakar stepwise AICc search; KPSS for the differencing order).
* `HcpCfEts` — ChangeFinder with a natively selected ETS model (AICc over the
  non-seasonal ETS family).

```python
from pyharbinger import HcpCfArima, HcpCfEts

for name, det_model in [("cf_arima", HcpCfArima()), ("cf_ets", HcpCfEts())]:
    det = det_model.detect(serie)
    print(f"{name:9s} change points at: {det.index[det['event']].tolist()}")
```

> **Parity note (Action-1 gate).** `HcpCfEts` matches R `ets()` **exactly** on
> all seven example series. `HcpCfArima` matches R's integration order `d` on all
> seven series and matches detections exactly on the well-conditioned ones
> (`simple`, `complex`); on the others the `(p, q)` stepwise order diverges from
> `forecast` (statsmodels full-ML AICc vs `forecast`'s CSS/ML approximation and
> stepwise path), so detections overlap substantially but are not identical. This
> is the documented tolerance for `forecast`'s auto-selection — the same
> acceptance criterion used for PELT / joinpoint / ELM.

## Bayesian Online Change Point Detection (native ocp replacement)

`HcpBocpd` is the Python port of R's `hcp_bocpd`. It implements **Bayesian
Online Change Point Detection** (Adams & MacKay, 2007): it tracks, online, the
posterior over the *run length* (time since the last change), which at each step
either **grows** (no change, weight `1 − 1/hazard`) or **resets** to zero (a
change, weight `1/hazard`), each hypothesis scored by its one-step predictive
probability under a conjugate **Normal-Inverse-Gamma** model (unknown mean and
variance → Student-t predictive).

It is a **native** (pure numpy/scipy) back-implementation of the R `ocp`
package, which R uses only as a `Suggests` dependency and which has no maintained
Python port — the back-implementation initiative's P1 case. Change points are
extracted exactly as `ocp`/`hcp_bocpd` do: three lists (`colmaxes`, `threshcps`,
`maxCPs`) vote per position, the votes are normalised, an initial `burn_in` is
zeroed, positions scoring `≥ threshold` are flagged, and `min_distance` thins
neighbours.

```python
from pyharbinger import HcpBocpd

det = HcpBocpd().detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

> **Parity note (Action-1 gate).** `HcpBocpd` matches the real `ocp` backend
> (CRAN-archive `ocp` 0.1.1, driven through R `hcp_bocpd`) **exactly** —
> identical detected indices — on all seven example series, including the long
> `increasing_amplitude` (n = 1002). Only the gaussian UPM (the R default) is
> ported.
