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

# Ensemble detection

An ensemble combines several Harbinger detectors into one. `HarEnsemble` uses a
**majority vote** (a position fires when more than half of the members agree);
`HarEnsembleFuzzy` instead builds a normalized fuzzy-consensus score.

```python
from pyharbinger import HarEnsemble, HanrFft, HanrFbiad, HanrHistogram, datasets

serie = datasets.examples_anomalies()["tt"]["serie"]
```

## Majority vote

```python
model = HarEnsemble(HanrFft(), HanrFbiad(), HanrHistogram()).fit(serie)
det = model.detect(serie)
print("ensemble events:", det.index[det["event"]].tolist())
det[det["event"]].head()
```

The per-member vote count is kept on the detection for inspection:

```python
import numpy as np
print("max votes:", int(np.nanmax(det.attrs["score"])))
```

> **Parity:** `HarEnsemble` reproduces the R `har_ensemble` majority vote exactly
> when its member detectors match R (e.g. `HanrFft` / `HanrFbiad` /
> `HanrHistogram`, all exact on `examples_anomalies`).
