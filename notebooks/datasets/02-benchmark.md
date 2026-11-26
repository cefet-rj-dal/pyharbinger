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

## A miniature (ships with the package)

```python
mini = datasets.benchmark("gecco")
print("series:", list(mini))
mini["tp"].head()
```

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

## General Harbinger examples

```python
examples = datasets.examples_harbinger()
{name: len(df) for name, df in examples.items()}
```

> **Parity:** values are extracted from the R `united` datasets, so the
> miniatures and the full artifacts are numerically identical to R; the
> serialization is Python-native (CSV + `manifest.json`) following the Eidolon
> architecture.
