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

# Classification-based detection (HancMl)

`HancMl` is a **supervised** anomaly detector: it wraps a
[pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) classifier trained
on labeled observations. At detection time the positive-class probability is
thresholded and consecutive candidates are collapsed by the candidate-selection
hook. Because the classifier family wraps scikit-learn, this adds **no new
dependency**.

```python
import numpy as np
import pandas as pd

from pyharbinger import HancMl, datasets
from pydaltoolbox import ClaKnn
```

## A labeled train/test series

We use the `tt` (train-test) example, whose `event` column is the supervision
signal. The feature (`serie`) is min-max normalized with the training range.

```python
tt = datasets.examples_anomalies()["tt"]
df = pd.DataFrame({
    "serie": tt["serie"].to_numpy(dtype=float),
    "event": np.where(tt["event"].to_numpy().astype(bool), "TRUE", "FALSE"),
})
train, test = df.iloc[:80].copy(), df.iloc[80:].reset_index(drop=True).copy()

lo, hi = train["serie"].min(), train["serie"].max()
train["serie"] = (train["serie"] - lo) / (hi - lo)
test["serie"] = (test["serie"] - lo) / (hi - lo)
print("ground-truth events in test:", np.where(test["event"].to_numpy() == "TRUE")[0].tolist())
```

## Fit a classifier and detect

`ClaKnn("event", ["FALSE", "TRUE"])` names the target column and the two class
levels; the second level is the positive (event) class.

```python
model = HancMl(ClaKnn("event", ["FALSE", "TRUE"], k=3)).fit(train)
detection = model.detect(test)
print("detected (idx):", detection.index[detection["event"]].tolist())
print(detection[detection["event"]])
```

## Evaluate

```python
metrics = model.evaluate(detection, test["event"].to_numpy() == "TRUE")
print(metrics["confMatrix"])
print("F1:", metrics["F1"])
```

> **Parity note.** On this example the detections match R's `hanc_ml` exactly
> (events at positions 10 and 21). The class *probabilities* themselves can
> differ slightly because scikit-learn's k-NN and R's `class::knn` break
> distance ties differently — the same backend tolerance documented for `HanrMl`.
