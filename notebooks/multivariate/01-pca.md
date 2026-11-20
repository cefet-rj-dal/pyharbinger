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

## Detect

By default `HmuPca()` keeps the leading components that explain 90% of the
variance (capped at `n_features - 1`, so at least one component is always
dropped). The discarded components form the reconstruction error used as the
anomaly score. Column `a` carries the injected anomaly, which the detector flags.

```python
det = HmuPca().detect(mv)
res = np.asarray(det.attrs["res"], dtype=float)
print("max |residual|:", f"{np.nanmax(res):.3f}")
print("detected (idx):", det.index[det["event"]].tolist())
```

You can also fix the number of retained components explicitly:

```python
det = HmuPca(num_components=1).detect(mv)
print("detected (idx):", det.index[det["event"]].tolist())
```

> **Note.** A row is considered present only when *all* its columns are
> non-missing (mirroring R's `har_store_refs`). Missing rows are restored to the
> original indexing in the detection table and never flagged.
