# pyharbinger

[![PyPI](https://img.shields.io/pypi/v/pyharbinger.svg)](https://pypi.org/project/pyharbinger/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyharbinger.svg)](https://pypi.org/project/pyharbinger/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python port of the R [harbinger](https://github.com/cefet-rj-dal/harbinger) — a **framework
for time series event detection** (anomalies, change points, and motifs/discords), built on
the DAL Toolbox *Experiment Line*.

`pyharbinger` unifies many detection methods under one interface (`fit` → `detect` →
`evaluate`) and a common evaluation protocol, so different detectors can be compared fairly —
reusing classic libraries (scikit-learn, numpy, pandas, scipy, ruptures) through thin
adapters.

It is the capstone of the DAL Python family:

- Builds on **[pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox)** (foundation) and
  **[pytspredit](https://github.com/cefet-rj-dal/pytspredit)** (time series prediction) — its
  regressors power the forecasting-based anomaly detectors.

## Installation

```bash
pip install pyharbinger
```

> On **Python 3.14**, `ruptures` 1.1.x has no prebuilt wheel yet; install with
> `pip install --prefer-binary pyharbinger` so pip uses the pure-Python
> `ruptures` 1.0.6 wheel. Python 3.9–3.13 install directly.

## Quickstart

```python
from pyharbinger import HanrFft, HarEval, datasets

# a labeled example series (one injected anomaly)
dataset = datasets.examples_anomalies()["simple"]
serie, event = dataset["serie"], dataset["event"]

# fit -> detect -> evaluate
model = HanrFft().fit(serie)
detection = model.detect(serie)
print(detection[detection["event"]])

metrics = model.evaluate(detection, event)
print(metrics["confMatrix"])
print("F1:", metrics["F1"])
```

## What's included

Every detector extends `Harbinger` and returns a detection table with columns
`idx` / `event` / `type` (`"anomaly"`, `"changepoint"`, `"discord"` or `""`).

| Group | Classes |
|---|---|
| **Core** | `Harbinger` (base contract), deviation / filter / candidate-selection utilities |
| **Evaluation** | `HarEval` (hard), `HarEvalSoft` (SoftED, soft matching via the Hungarian algorithm) |
| **Anomaly** | `HanrMl` (forecasting residuals, wraps a pytspredit regressor), `HanrArima` (ARIMA residuals, reuses the DAL ARIMA), `HanrFft`, `HanrFftAmoc`, `HanrFftBinseg`, `HanrFftAmocCusum`, `HanrFftBinsegCusum`, `HanrFftSma` (adaptive FFT-selected moving average), `HanrHistogram`, `HanrFbiad` |
| **Change point** | `HcpAmoc`, `HcpBinseg`, `HcpPelt` (via `ruptures`), `HcpKswin`, `HcpPageHinkley`, `HcpCfLr`, `HcpScp`, `HcpJoinpoint`, `HcpChow`, `HcpGft` (native structural change), `HcpCfArima`, `HcpCfEts` (ChangeFinder, native forecast replacement), `HcpBocpd` (Bayesian online, "++" ocp re-engineering) |
| **Clustering** | `HanctKmeans` (anomalies / discords) |
| **Classification** | `HancMl` (supervised; wraps a pydaltoolbox classifier) |
| **Multivariate** | `HmuPca` (PCA reconstruction error) |
| **Symbolic** | `TransSax` / `TransXsax` (SAX / XSAX encoding), `HmoSax` / `HmoXsax` (motifs), `HdisSax` (discords) |
| **Ensemble** | `HarEnsemble` (majority vote), `HarEnsembleFuzzy` (fuzzy consensus) |
| **Plotting** | `har_plot`, `har_ensemble_plot`, `har_ensemble_plot_models` (matplotlib) |
| **Online / streaming** | `HarOnlineSession` + sources (`HarSourceSimulated`/`HarSourceDataframe`/`HarSourceCallback`/`HarSourceKafka` + collector seam), memory policies, execution strategies; `HarStreamEval` + `har_stream_experiment` |
| **Datasets** | `datasets.examples_anomalies()`, `datasets.examples_changepoints()`, `datasets.examples_motifs()` |

Detectors accept a univariate series or, for the multivariate ones, a
`DataFrame` with one column per variable.

### Roadmap

Later releases expand coverage toward feature parity with R in two cycles. The
**low-dependency** cycle adds the FFT-hybrid anomaly detectors, the extended
change-point family (ChangeFinder, Chow, GFT, join-point, BOCPD, ...), native
SAX motif/discord, ensembles, plotting, and the streaming runners. The
**heavy-dependency** cycle then adds — each behind an optional extra — matrix
profile (`stumpy`), GARCH (`arch`), EMD (`EMD-signal`), wavelets (`PyWavelets`),
DTW clustering, and the autoencoder detectors (deep-learning backend). See
`develop/planejamento.md` for the full plan.

## Faithfulness to R

The port is validated against the R package on the same data at every step.
Deterministic components match R to numerical precision (deviation measures,
filters, candidate selection, hard/soft evaluation, FFT/histogram/FBIAD
detectors, `HanrMl` with a deterministic regressor, `HanctKmeans` centroids
count, and `HcpPageHinkley`). Where the backend or RNG differs, the tolerance is
documented:

- `HanrMl` with a rank-reduced ELM (`nhid < input_size`, `purelin`) depends on
  the hidden-weight RNG.
- `HanctKmeans` and `HcpKswin` depend on initialization / random-subsample RNG.
- `HcpAmoc`/`HcpBinseg`/`HcpPelt` use `ruptures` (Gaussian cost); R uses
  `changepoint::cpt.meanvar` with the MBIC penalty, so agreement is approximate.
- `HancMl` detections match R (e.g. the `hanc_ml` k-NN example flags the same
  positions), but class *probabilities* can differ because scikit-learn and R's
  `class::knn` break ties differently.
- `HmuPca` matches R exactly (same events and residuals) after fixing a latent
  degeneracy in R's `hmu_pca`, which kept *all* principal components; both now
  reconstruct from a reduced set (90% variance by default, capped at
  `n_features - 1`).
- `HcpCfLr`, `HcpScp` and the FFT-hybrid anomaly detectors match R exactly on the
  example series. `HcpJoinpoint` matches R on well-conditioned series; on a
  perfectly piecewise-linear series its segment residuals collapse to round-off,
  so the exact break positions are numerically ill-conditioned (the dominant
  change point is still recovered).
- `HcpChow` and `HcpGft` are **re-engineered "++" versions** of R's `strucchange`
  backend: the least-squares (mean-only) Chow sup-F / Bai & Perron breakpoints are
  generalised to a Gaussian **mean + variance** segmentation cost (with the
  conservative LWZ penalty for the break count in `HcpGft`), so they also catch
  scale/variance-regime changes the mean-shift model misses. They deliberately
  diverge from `strucchange`, so the gate is detection quality vs ground truth,
  not R parity: on the example series the summed SoftED-F1 rises from the faithful
  ports' 3.40 / 2.44 (R 3.31 / 2.20) to **3.73 / 3.17** (`develop/benchmark_backimpl.py`).
- `HcpCfArima` and `HcpCfEts` are **re-engineered "++" versions** of R's `forecast`
  backend. They are a native, pure-numpy **robust ChangeFinder**: the canonical
  two-stage online **SDAR** surprise score (Takeuchi & Yamanishi, 2006 — strong on
  abrupt/variance/multi-change regimes) fused (union) with a Gaussian mean+variance
  structural break (which recovers smooth trend/slope changes). `HcpCfArima` uses an
  autoregressive SDAR predictor, `HcpCfEts` a discounted Holt predictor — no
  `forecast`, no `statsmodels`. The gate is detection quality vs ground truth, not R
  parity: on the example series the summed SoftED-F1 rises from the faithful ports'
  2.78 / 2.66 (R `forecast` 2.92 / 2.66) to **3.84 / 3.64**
  (`develop/benchmark_backimpl.py`).
- `HcpBocpd` is a **re-engineered "++" version** of Bayesian Online Change Point
  Detection (Adams & MacKay, 2007), not a port of R's `ocp` backend. Three
  advances over `ocp`: (1) a conjugate **Bayesian autoregressive UPM** (a linear
  regression of `x_t` on its lags per run length) in place of `ocp`'s i.i.d.
  Gaussian UPM, so it detects changes in *dynamics* (slope, trend,
  autocorrelation), not only level/variance; (2) a **log-space, eps-pruned
  streaming** run-length recursion (bounded state, `O(n·R̄)` instead of `O(n²)`);
  and (3) change points read from the **Bayesian predictive surprise** via a
  robust local z-score, replacing `ocp`'s noisy three-list voting. It deliberately
  diverges from `ocp`, so the gate is detection quality vs ground truth, not R
  parity: on the example series the summed SoftED-F1 rises from the faithful port's
  **1.12** (= R `ocp`) to **3.59** (`develop/benchmark_backimpl.py`) — the AR UPM
  recovers the trend/slope change points the mean/variance model misses.

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- [pytspredit](https://github.com/cefet-rj-dal/pytspredit) — time series prediction (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
