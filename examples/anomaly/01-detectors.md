# Anomaly detectors

Anomaly detectors flag individual observations that depart from the expected
behavior of the series. Each one follows the same `fit` → `detect` → `evaluate`
contract and returns a detection table with `type == "anomaly"` on flagged
points. We use `examples_anomalies$simple`, which carries a single injected
anomaly.


```python
import numpy as np
from pyharbinger import (
    HanrMl, HanrFft, HanrHistogram, HanrFbiad, HarEval, datasets,
)

dataset = datasets.examples_anomalies()["simple"]
serie, event = dataset["serie"], dataset["event"]
print("ground-truth anomaly at:", dataset.index[event].tolist())
```

    ground-truth anomaly at: [49]
    

## HanrFft — high-pass FFT filtering

Isolates high-frequency components via the FFT and flags strong departures.


```python
det = HanrFft().detect(serie)
print("detected:", det.index[det["event"]].tolist())
print(HanrFft().evaluate(det, event)["confMatrix"])
```

    detected: [49]
                     event_TRUE  event_FALSE
    detection_TRUE            1            0
    detection_FALSE           0          100
    

## FFT hybrids — spectral change-point cutoff

These variants keep the same FFT high-pass idea but choose the cutoff frequency
from a **change point in the power spectrum** (AMOC or BinSeg), optionally on the
cumulative spectrum. `HanrFftBinsegCusum` uses high-group candidate selection.


```python
from pyharbinger import (
    HanrFftAmoc, HanrFftBinseg, HanrFftAmocCusum, HanrFftBinsegCusum,
)

for name, det_model in [
    ("fft_amoc", HanrFftAmoc()),
    ("fft_binseg", HanrFftBinseg()),
    ("fft_amoc_cusum", HanrFftAmocCusum()),
    ("fft_binseg_cusum", HanrFftBinsegCusum()),
]:
    det = det_model.detect(serie)
    print(f"{name:18s} detected: {det.index[det['event']].tolist()}")
```

    fft_amoc           detected: [49]
    fft_binseg         detected: [49]
    fft_amoc_cusum     detected: [49]
    fft_binseg_cusum   detected: [49]
    

## HanrHistogram — histogram density

Flags observations that fall into low-density or out-of-range histogram bins.


```python
det = HanrHistogram().detect(serie)
det[det["event"]]
```




        idx  event     type
    0     0   True  anomaly
    49   49   True  anomaly



## HanrFbiad — forward/backward inertial

Compares each observation against both its forward and backward inertial context
(moving-average windows in each direction).


```python
det = HanrFbiad().detect(serie)
print("detected:", det.index[det["event"]].tolist())
```

    detected: [49]
    

## HanrMl — forecasting residuals

Wraps a `pytspredit` regressor: it forecasts the next value from a sliding window
and flags large residuals. This is the bridge detector that reuses the full
pydaltoolbox → pytspredit → pyharbinger stack. Here we use a deterministic
k-NN regressor.


```python
from pytspredit import TsKnn, TsNormGminmax

model = HanrMl(TsKnn(TsNormGminmax(), input_size=4, k=3), sw_size=15)
model = model.fit(serie)
det = model.detect(serie)
print("detected:", det.index[det["event"]].tolist())
print(model.evaluate(det, event)["confMatrix"])
```

    detected: [49]
                     event_TRUE  event_FALSE
    detection_TRUE            1            0
    detection_FALSE           0          100
    
