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
| **Anomaly** | `HanrMl` (forecasting residuals, wraps a pytspredit regressor), `HanrFft`, `HanrHistogram`, `HanrFbiad` |
| **Change point** | `HcpAmoc`, `HcpBinseg`, `HcpPelt` (via `ruptures`), `HcpKswin`, `HcpPageHinkley` |
| **Clustering** | `HanctKmeans` (anomalies / discords) |
| **Classification** | `HancMl` (supervised; wraps a pydaltoolbox classifier) |
| **Multivariate** | `HmuPca` (PCA reconstruction error) |
| **Datasets** | `datasets.examples_anomalies()`, `datasets.examples_changepoints()` |

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

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- [pytspredit](https://github.com/cefet-rj-dal/pytspredit) — time series prediction (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
