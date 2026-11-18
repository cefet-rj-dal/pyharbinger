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




    ['simple',
     'contextual',
     'trend',
     'multiple',
     'sequence',
     'tt',
     'tt_warped',
     'increasing_amplitude',
     'decreasing_amplitude',
     'volatile']




```python
# the "simple" series: a smooth signal with a single injected anomaly
simple = anomalies["simple"]
print("shape:", simple.shape)
print("number of events:", int(simple["event"].sum()))
simple.head()
```

    shape: (101, 2)
    number of events: 1
    




          serie  event
    0  1.000000  False
    1  0.968912  False
    2  0.877583  False
    3  0.731689  False
    4  0.540302  False




```python
# the index of the ground-truth anomaly
simple.index[simple["event"]].tolist()
```




    [49]



## Change-point series


```python
changepoints = datasets.examples_changepoints()
list(changepoints)
```




    ['simple',
     'sinusoidal',
     'incremental',
     'abrupt',
     'volatility',
     'increasing_amplitude',
     'complex']




```python
cp = changepoints["simple"]
print("shape:", cp.shape)
print("change-point indices:", cp.index[cp["event"]].tolist())
cp.head()
```

    shape: (101, 2)
    change-point indices: [49]
    




       serie  event
    0   0.00  False
    1   0.25  False
    2   0.50  False
    3   0.75  False
    4   1.00  False


