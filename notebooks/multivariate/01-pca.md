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

# Multivariate detection (HmuPca)

`HmuPca` standardizes a **multivariate** series, projects it onto principal
components, reconstructs it, and flags rows with a large reconstruction error.
It wraps scikit-learn's PCA, so it adds **no new dependency**. This notebook also
shows the new **multivariate detection path**: the input is a `DataFrame` with
one column per variable.

```python
import numpy as np
import pandas as pd

from pyharbinger import HmuPca, datasets
```

## A multivariate series

We build a deterministic 2-variable series from two shipped example series.

```python
an = datasets.examples_anomalies()
mv = pd.DataFrame({
    "a": an["simple"]["serie"].to_numpy(dtype=float),
    "b": an["contextual"]["serie"].to_numpy(dtype=float),
})
mv.head()
```

## Faithful R behaviour (keep all components)

By default `HmuPca()` keeps **all** principal components, faithfully reproducing
R's `hmu_pca`. With all components retained the reconstruction is exact, so the
residual collapses to floating-point round-off — the detector is effectively
degenerate (this matches R, where `max|res|` is ~1e-31).

```python
det = HmuPca().detect(mv)
res = np.asarray(det.attrs["res"], dtype=float)
print("max |residual| (all components):", f"{np.nanmax(np.abs(res)):.3e}")
```

## A genuine detector (pyharbinger extension)

Keeping fewer components than the input dimensionality turns it into a real
reconstruction-error detector. `n_components` is a **pyharbinger extension** (not
present in R).

```python
det = HmuPca(n_components=1).detect(mv)
res = np.asarray(det.attrs["res"], dtype=float)
print("max |residual| (1 component):", f"{np.nanmax(np.abs(res)):.3f}")
print("detected (idx):", det.index[det["event"]].tolist())
```

> **Note.** A row is considered present only when *all* its columns are
> non-missing (mirroring R's `har_store_refs`). Missing rows are restored to the
> original indexing in the detection table and never flagged.
