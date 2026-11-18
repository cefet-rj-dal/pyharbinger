# pyharbinger

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

## What's included (v1)

Every detector extends `Harbinger` and returns a detection table with columns
`idx` / `event` / `type` (`"anomaly"`, `"changepoint"`, `"discord"` or `""`).

| Group | Classes |
|---|---|
| **Core** | `Harbinger` (base contract), deviation / filter / candidate-selection utilities |
| **Evaluation** | `HarEval` (hard), `HarEvalSoft` (SoftED, soft matching via the Hungarian algorithm) |
| **Anomaly** | `HanrMl` (forecasting residuals, wraps a pytspredit regressor), `HanrFft`, `HanrHistogram`, `HanrFbiad` |
| **Change point** | `HcpAmoc`, `HcpBinseg`, `HcpPelt` (via `ruptures`), `HcpKswin`, `HcpPageHinkley` |
| **Clustering** | `HanctKmeans` (anomalies / discords) |
| **Datasets** | `datasets.examples_anomalies()`, `datasets.examples_changepoints()` |

### Deferred to a future release (0.2.0 / Block B)

ML-classification detector (`hanc_ml`), motif/discord via matrix profile
(`hdis_mp`/`hmo_mp`), ARIMA/GARCH/EMD/wavelet anomaly detectors, DTW clustering,
SAX representations, ensembles, streaming/online runners, and plotting helpers.

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

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- [pytspredit](https://github.com/cefet-rj-dal/pytspredit) — time series prediction (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
