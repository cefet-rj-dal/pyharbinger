# Benchmark datasets (Eidolon miniatures + full data on demand)

The `united` benchmark collection (Yahoo A1–A4, gecco, MIT-BIH, NAB, oil_3w,
UCR) ships as compact **miniatures**: the first series of each dataset, truncated
to 30 rows. The complete artifact (every series, full length) is published
remotely and resolved on demand by `datasets.loadfulldata` — using only the
standard library, no extra dependency.


```python
from pyharbinger import datasets

datasets.list_benchmarks()[:6]
```




    ['A1Benchmark',
     'A2Benchmark',
     'A3Benchmark',
     'A4Benchmark',
     'gecco',
     'mit_bih_MLII']



## A miniature (ships with the package)


```python
mini = datasets.benchmark("gecco")
print("series:", list(mini))
mini["tp"].head()
```

    series: ['tp']





       idx  value  event  type
    0    1    6.5  False   NaN
    1    2    6.5  False   NaN
    2    3    6.5  False   NaN
    3    4    6.5  False   NaN
    4    5    6.5  False   NaN



Each dataset is a dict mapping a series name to its `DataFrame` (`idx`, `value`,
`event`, `type`; multivariate datasets such as `oil_3w_*` carry several value
columns).

## The full dataset, resolved on demand

`loadfulldata` downloads the complete artifact (a gzipped CSV in the
`cefet-rj-dal/united` repository) and returns the same structure as the
miniature — every series, at full length:


```python
print("full artifact URL:", datasets.benchmark_url("gecco"))
# full = datasets.loadfulldata("gecco")   # downloads ~10 MB
# print({k: len(v) for k, v in full.items()})
```

    full artifact URL: https://raw.githubusercontent.com/cefet-rj-dal/united/refs/heads/main/python/gecco.csv.gz


## General Harbinger examples


```python
examples = datasets.examples_harbinger()
{name: len(df) for name, df in examples.items()}
```




    {'nonstationarity': 1000,
     'global_temperature_yearly': 173,
     'global_temperature_monthly': 2076,
     'multidimensional': 200,
     'seattle_week': 53,
     'seattle_daily': 370}



> **Parity:** values are extracted from the R `united` datasets, so the
> miniatures and the full artifacts are numerically identical to R; the
> serialization is Python-native (CSV + `manifest.json`) following the Eidolon
> architecture.
