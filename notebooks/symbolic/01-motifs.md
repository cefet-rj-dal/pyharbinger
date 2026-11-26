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

# Symbolic motif & discord discovery (SAX / XSAX)

The Symbolic Aggregate approXimation (SAX) z-normalizes a series, bins it into
`alpha` quantile symbols, and slides a length-`w` window to form symbolic
*words*. Frequently repeated words mark **motifs**; rare, isolated words mark
**discords**. XSAX extends the alphabet to an alphanumeric base-36 code for finer
resolution.

```python
from pyharbinger import HdisSax, HmoSax, HmoXsax, TransSax, TransXsax, datasets

dataset = datasets.examples_motifs()["simple"]
serie = dataset["serie"]
```

## The symbolic representation

```python
sax = TransSax(26).fit(serie).transform(serie)
xsax = TransXsax(37).fit(serie).transform(serie)
print("SAX :", "".join(sax[:15]))
print("XSAX:", " ".join(xsax[:15]))
```

## Motif discovery (SAX)

Words occurring at least `qtd` times are motifs; overlapping repeats are
de-duplicated and the result is tagged `type == "motif"`.

```python
det = HmoSax(26, 3, 3).fit(serie).detect(serie)
print("motifs at:", det.index[det["event"]].tolist())
det[det["event"]].head()
```

## Motif discovery (XSAX)

```python
det = HmoXsax(37, 3, 3).fit(serie).detect(serie)
print("motifs at:", det.index[det["event"]].tolist())
```

## Discord discovery (SAX)

Discords are unique words that are not near any repeated pattern; the result is
tagged `type == "discord"`.

```python
det = HdisSax(26, 3, 3).fit(serie).detect(serie)
print("discords at:", det.index[det["event"]].tolist())
```

> **Parity:** the transforms and all three detectors reproduce the R
> `harbinger` package exactly — identical SAX/XSAX codes and identical detected
> index sets on every `examples_motifs` series (`simple`, `mitdb100`,
> `mitdb102`).
