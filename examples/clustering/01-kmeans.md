# Clustering-based detection (k-means)

`HanctKmeans` clusters the sliding-window subsequences of a series and flags the
ones that lie far from their nearest centroid. With `seq == 1` it flags isolated
**anomalies**; with `seq > 1` it flags whole subsequences as **discords**.

> **Note on parity:** scikit-learn's k-means uses a different initialization and
> RNG than R's `stats::kmeans`, so results are approximately comparable; set
> `random_state` for reproducibility.


```python
from pyharbinger import HanctKmeans, HarEval, datasets

dataset = datasets.examples_anomalies()["simple"]
serie, event = dataset["serie"], dataset["event"]
print("ground-truth anomaly at:", dataset.index[event].tolist())
```

    ground-truth anomaly at: [49]


## Anomalies (seq = 1)


```python
model = HanctKmeans(random_state=0).fit(serie)
print("number of centroids:", model.centers)
det = model.detect(serie)
print("detected:", det.index[det["event"]].tolist())
print(model.evaluate(det, event)["confMatrix"])
```

    number of centroids: 3
    detected: [49]
                     event_TRUE  event_FALSE
    detection_TRUE            1            0
    detection_FALSE           0          100


## Discords (seq > 1)

With a subsequence length greater than one, flagged windows are reported as
discords, with `seq`/`seqlen` columns describing their length.


```python
model = HanctKmeans(seq=4, random_state=0).fit(serie)
det = model.detect(serie)
det[det["event"]].head()
```




    Empty DataFrame
    Columns: [idx, event, type, seq, seqlen]
    Index: []


