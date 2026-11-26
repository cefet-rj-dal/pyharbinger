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

    SAX : KKJIHFEDCBBAAAA
    XSAX: 0F 0E 0E 0C 0A 08 06 04 02 02 01 00 00 00 01


## Motif discovery (SAX)

Words occurring at least `qtd` times are motifs; overlapping repeats are
de-duplicated and the result is tagged `type == "motif"`.


```python
det = HmoSax(26, 3, 3).fit(serie).detect(serie)
print("motifs at:", det.index[det["event"]].tolist())
det[det["event"]].head()
```

    motifs at: [24, 49, 74]





        idx  event   type  seq  seqlen
    24   24   True  motif  QST     3.0
    49   49   True  motif  QST     3.0
    74   74   True  motif  QST     3.0



## Motif discovery (XSAX)


```python
det = HmoXsax(37, 3, 3).fit(serie).detect(serie)
print("motifs at:", det.index[det["event"]].tolist())
```

    motifs at: [24, 49, 74]


## Discord discovery (SAX)

Discords are unique words that are not near any repeated pattern; the result is
tagged `type == "discord"`.


```python
det = HdisSax(26, 3, 3).fit(serie).detect(serie)
print("discords at:", det.index[det["event"]].tolist())
```

    discords at: [16, 20, 38, 42, 63, 67, 85, 89, 93]


> **Parity:** the transforms and all three detectors reproduce the R
> `harbinger` package exactly — identical SAX/XSAX codes and identical detected
> index sets on every `examples_motifs` series (`simple`, `mitdb100`,
> `mitdb102`).
