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

    ground-truth change points at: [99]


## HcpAmoc — at most one change


```python
det = HcpAmoc().detect(serie)
print("change point at:", det.index[det["event"]].tolist())
```

    change point at: [99]


## HcpBinseg — binary segmentation


```python
det = HcpBinseg(Q=2).detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

    change points at: [99, 148]


## HcpPelt — PELT

The `penalty` argument controls granularity (higher = fewer change points).


```python
det = HcpPelt().detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

    change points at:

     [77, 79, 85, 94, 99, 117, 119, 148]


## HcpPageHinkley — online cumulative statistic


```python
det = HcpPageHinkley().detect(serie)
print("change point at:", det.index[det["event"]].tolist())
```

    change point at: [109]


## HcpKswin — Kolmogorov-Smirnov windowing

Uses a random subsample, so fix `random_state` for reproducibility.


```python
det = HcpKswin(random_state=0).detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

    change points at: [105, 182]


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

    cf_lr      change points at: [88, 92, 94, 102]
    scp        change points at: [89]


    joinpoint  change points at: [63, 86, 100, 149, 173]


> **Parity note.** `HcpCfLr`, `HcpScp` and `HcpJoinpoint` all match R exactly on
> every example series. The joinpoint dynamic program is tie-degenerate on
> (near-)symmetric or perfectly piecewise-linear series — several break sets share
> the same residual sum of squares to within round-off — so a tolerant tie-break
> plus an RSS floor keep the segmentation deterministic and backend-independent,
> making the Python releitura and the R backport agree bit-for-bit.

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

    chow  change points at: [99]
    gft   change points at: [99]


> **Quality note (Action-1++ gate).** Because the "++" versions deliberately
> diverge from `strucchange`, they are no longer judged by R parity but by
> **detection quality vs the ground truth** (`event` column). On the seven
> `examples_changepoints` series the summed SoftED-F1 rises from the faithful
> ports' 3.40 / 2.44 (R `strucchange` 3.31 / 2.20) to **3.73 (chow) / 3.17 (gft)**
> — the mean+variance cost recovers variance/amplitude regimes the original
> mean-shift model misses. See `develop/benchmark_backimpl.py`.

## ChangeFinder (forecast replacement, re-engineered "++")

`HcpCfArima` and `HcpCfEts` began as native reimplementations that scored
`statsmodels` ARIMA/ETS residuals. They were then **re-engineered into "++"
versions** — a native, pure-numpy **robust ChangeFinder** that fuses two
complementary signals and takes their union:

1. the **canonical two-stage SDAR** ChangeFinder of Takeuchi & Yamanishi (2006):
   an online Sequentially Discounting AutoRegressive model with a forgetting
   factor, scored by the predictive log-loss — strong on abrupt, variance and
   multi-change regimes;
2. a Gaussian **mean+variance structural break** (shared with `HcpChow`/`HcpGft`),
   which recovers the smooth trend/slope changes the AR-surprise score is blind to.

`HcpCfArima` uses an autoregressive SDAR predictor; `HcpCfEts` a discounted Holt
(level+trend) predictor. Both are pure numpy — no `forecast`, no `statsmodels`.


```python
from pyharbinger import HcpCfArima, HcpCfEts

for name, det_model in [("cf_arima", HcpCfArima()), ("cf_ets", HcpCfEts())]:
    det = det_model.detect(serie)
    print(f"{name:9s} change points at: {det.index[det['event']].tolist()}")
```

    cf_arima  change points at: [99, 100, 102, 195]
    cf_ets    change points at: [99, 106, 195]


> **Quality note (Action-1++ gate).** Because the "++" versions deliberately
> diverge from `forecast`, they are judged by detection quality vs the ground
> truth (`event` column), not R parity. On the seven `examples_changepoints`
> series the summed SoftED-F1 rises from the faithful ports' 2.78 / 2.66 (R
> `forecast` 2.92 / 2.66) to **3.84 (cf_arima) / 3.64 (cf_ets)** — the SDAR
> surprise catches abrupt/variance regimes while the mean+variance break catches
> smooth trend changes. See `develop/benchmark_backimpl.py`.

## Bayesian Online Change Point Detection ("++" ocp re-engineering)

`HcpBocpd` implements **Bayesian Online Change Point Detection** (Adams & MacKay,
2007): it tracks, online, the posterior over the *run length* (time since the last
change), which at each step either **grows** (no change, weight `1 − 1/hazard`) or
**resets** to zero (a change, weight `1/hazard`), each hypothesis scored by its
one-step predictive probability under an *underlying predictive model* (UPM).

It is a **re-engineered "++" version** of BOCPD — a genuine re-reading, not a port
of R's `ocp` package. Three advances over `ocp`:

1. **Autoregressive UPM.** `ocp` uses an i.i.d. Gaussian UPM, which only sees
   shifts in *level* and *variance*. Here the per-run-length model is a conjugate
   **Bayesian linear autoregression** (a regression of `x_t` on `[1, x_{t−1}, …]`
   with a Normal-Inverse-Gamma prior and a Student-t predictive). A change of
   slope or oscillation — the kink of a triangular ramp, the turning point of a
   sinusoid — breaks the segment regression and spikes the evidence, recovering
   change points the mean/variance UPM never detects.
2. **Log-space + eps-pruning.** The whole recursion runs in log-space (no
   underflow clamping), and the run-length distribution is pruned to the entries
   holding all but a fraction `prune_eps` of the peak mass — a bounded-state,
   genuinely **streaming** `O(n·R̄)` recursion instead of the `O(n²)` batch one.
3. **Surprise-based extraction.** Instead of `ocp`'s noisy three-list voting
   (which over-fires), change points are read from the **Bayesian predictive
   surprise** `−log p(xₜ | x₁:ₜ₋₁)` via a causal robust local z-score, thinned by
   `burn_in` and `min_distance`.


```python
from pyharbinger import HcpBocpd

det = HcpBocpd().detect(serie)
print("change points at:", det.index[det["event"]].tolist())
```

    change points at: [50, 100, 130, 194]


> **Gate (Action-1++).** This deliberately diverges from `ocp`, so acceptance is
> detection quality vs the ground truth (SoftED-F1), not R parity. Summed over the
> seven example series the "++" detector scores **3.59** (of 7.0), versus **1.12**
> for the faithful `ocp` reference — the AR UPM recovers the trend/slope change
> points the mean/variance model misses. See `develop/benchmark_backimpl.py`.
