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




              a         b
    0  1.000000  1.000000
    1  0.968912  0.968912
    2  0.877583  0.877583
    3  0.731689  0.731689
    4  0.540302  0.540302



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

    max |residual|: 1.054
    detected (idx): [49]
    

You can also fix the number of retained components explicitly:


```python
det = HmuPca(num_components=1).detect(mv)
print("detected (idx):", det.index[det["event"]].tolist())
```

    detected (idx): [49]
    

> **Note.** A row is considered present only when *all* its columns are
> non-missing (mirroring R's `har_store_refs`). Missing rows are restored to the
> original indexing in the detection table and never flagged.
