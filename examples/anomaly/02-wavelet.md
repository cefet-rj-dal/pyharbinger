# Wavelet anomaly detection

`HanrWavelet` decomposes the series with a MODWT multiresolution transform,
sums the detail bands into a magnitude signal, and flags large deviations.
It reuses the **native** `pytspredit` MODWT — no external wavelet dependency.


```python
from pyharbinger import HanrWavelet, HarEval, datasets

dataset = datasets.examples_anomalies()["tt"]
serie, event = dataset["serie"], dataset["event"]

model = HanrWavelet(filter="haar").fit(serie)
det = model.detect(serie)
print("detected:", det.index[det["event"]].tolist())
print(model.evaluate(det, event)["confMatrix"])
```

    detected: [11, 37, 49, 100]
                     event_TRUE  event_FALSE
    detection_TRUE            4            0
    detection_FALSE           2           95


Available filters: `haar` (default), `d4`, `la8`, `bl14`, `c6`; the number of
levels defaults to `floor(log2(n))` (matching R).

> **Parity:** `HanrWavelet` reproduces R `harbinger::hanr_wavelet` exactly on
> every `examples_anomalies` series — `pytspredit`'s native MODWT matches R
> `wavelets::modwt`.
