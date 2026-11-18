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

# Example series for event detection

`pyharbinger` ships the labeled example series from the R `harbinger` package,
extracted from the original `.RData` so they are numerically identical. Each
collection is a dict mapping a series name to a `DataFrame` with two columns:

* `serie` — the numeric time series values.
* `event` — the boolean ground-truth events (the anomalies or change points to
  recover).

These are the didactic datasets used throughout the detector examples.

**Reference:** Ogasawara, E., Salles, R., Porto, F., Pacitti, E. *Event Detection
in Time Series.* Springer, 2025. doi:10.1007/978-3-031-75941-3

```python
from pyharbinger import datasets
```

## Anomaly series

```python
anomalies = datasets.examples_anomalies()
list(anomalies)
```

```python
# the "simple" series: a smooth signal with a single injected anomaly
simple = anomalies["simple"]
print("shape:", simple.shape)
print("number of events:", int(simple["event"].sum()))
simple.head()
```

```python
# the index of the ground-truth anomaly
simple.index[simple["event"]].tolist()
```

## Change-point series

```python
changepoints = datasets.examples_changepoints()
list(changepoints)
```

```python
cp = changepoints["simple"]
print("shape:", cp.shape)
print("change-point indices:", cp.index[cp["event"]].tolist())
cp.head()
```
