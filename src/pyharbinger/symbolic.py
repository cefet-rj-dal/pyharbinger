"""Symbolic (SAX/XSAX) representation and motif/discord discovery.

Python port of the R ``trans_sax`` / ``trans_xsax`` transformers and the
``hmo_sax`` / ``hmo_xsax`` (motif) and ``hdis_sax`` (discord) detectors.

The Symbolic Aggregate approXimation (SAX) discretizes a z-normalized series
into an alphabet of ``alpha`` symbols using quantile bins; XSAX extends the
alphabet to an alphanumeric base-36 code for finer resolution. Sliding windows
of symbols form *words*; frequently repeated words mark **motifs** while rare,
isolated words mark **discords**.

Theory
------
Each value is z-normalized and assigned to one of ``alpha`` equiprobable
quantile bins, then mapped to a symbol. A length-``w`` sliding window of symbols
is concatenated into a word. Motif discovery groups identical words and keeps
the ones occurring at least ``qtd`` times (greedily de-overlapped); discord
discovery keeps unique words that are not near any repeated pattern.

References
----------
- Lin J, Keogh E, Lonardi S, Chiu B (2007). A symbolic representation of time
  series, with implications for streaming algorithms. Data Mining and Knowledge
  Discovery 15, 107-144. doi:10.1007/s10618-007-0064-z
- Ogasawara E, Salles R, Porto F, Pacitti E (2025). Event Detection in Time
  Series. Springer. doi:10.1007/978-3-031-75941-3

Porting
-------
Mirrors ``trans_sax.R`` / ``trans_xsax.R`` / ``hmo_sax.R`` / ``hmo_xsax.R`` /
``hdis_sax.R``. Fidelity: exact (same detected-index sets on ``examples_motifs``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydaltoolbox import DalTransform

from .harbinger import Harbinger, as_series_array

__all__ = ["HdisSax", "HmoSax", "HmoXsax", "TransSax", "TransXsax"]

_SAX_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_XSAX_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- symbolic encoding -------------------------------------------------------

def _code_width(alpha: int, base: int) -> int:
    """Number of characters per symbol: smallest ``i`` with ``base**i >= alpha``.

    Mirrors R ``ceiling(log(alpha, base))`` robustly (no float round-off).
    """
    i = 1
    while base ** i < alpha:
        i += 1
    return i


def _to_symbol(num: int, base: int, chars: str) -> str:
    """Convert ``num`` to its base-``base`` string using ``chars`` (R ``convert_to_sax``)."""
    out = ""
    while num >= base:
        r = num % base
        out = chars[r] + out
        num //= base
    return chars[num] + out


def _symbol_codes(alpha: int, base: int, chars: str) -> list[str]:
    """Zero-padded symbol codes for ``0 .. alpha-1`` (R ``mycode``)."""
    width = _code_width(alpha, base)
    return [_to_symbol(k, base, chars).rjust(width, "0") for k in range(alpha)]


def _sax_encode(data: np.ndarray, alpha: int, base: int, chars: str) -> np.ndarray:
    """Z-normalize and quantile-bin ``data`` into symbolic words.

    Mirrors ``transform.trans_sax`` / ``transform.trans_xsax``: z-normalize with
    the sample standard deviation (``ddof=1``), compute ``alpha`` quantile breaks
    (type-7), bin with ``cut(..., include.lowest=TRUE)`` semantics, and map each
    bin to its padded symbol code.
    """
    data = np.asarray(data, dtype=float)
    mean = np.nanmean(data)
    sd = np.nanstd(data, ddof=1)
    norm = (data - mean) / sd

    probs = np.arange(0, alpha + 1) / alpha
    q = np.nanquantile(norm, probs)  # type-7 (linear) == R stats::quantile
    breaks = np.unique(q)  # ascending, de-duplicated (R unique(q))

    # R cut(norm, breaks, labels=FALSE, include.lowest=TRUE, right=TRUE):
    # bin k == (breaks[k-1], breaks[k]] (1-based), with the minimum folded into
    # bin 1. searchsorted(..., side="left") gives exactly that index.
    bins = np.searchsorted(breaks, norm, side="left")
    bins[bins == 0] = 1  # include.lowest: norm == min -> bin 1

    codes = _symbol_codes(alpha, base, chars)
    return np.array([codes[b - 1] for b in bins], dtype=object)


class TransSax(DalTransform):
    """SAX discretization of a numeric series into an alphabet of size ``alpha`` (2-26).

    Porting
    -------
    Mirrors ``trans_sax.R``. Fidelity: exact.
    """

    def __init__(self, alpha: int):
        super().__init__()
        if not (isinstance(alpha, (int, np.integer)) and alpha >= 2):
            raise ValueError("alpha must be at least 2")
        self.alpha = int(alpha)

    def transform(self, data, **kwargs) -> np.ndarray:
        return _sax_encode(as_series_array(data), self.alpha, 26, _SAX_CHARS)


class TransXsax(DalTransform):
    """Extended SAX (XSAX) discretization using an alphanumeric base-36 alphabet (2-36).

    Porting
    -------
    Mirrors ``trans_xsax.R``. Fidelity: exact.
    """

    def __init__(self, alpha: int):
        super().__init__()
        if not (isinstance(alpha, (int, np.integer)) and alpha >= 2):
            raise ValueError("alpha must be at least 2")
        self.alpha = int(alpha)

    def transform(self, data, **kwargs) -> np.ndarray:
        return _sax_encode(as_series_array(data), self.alpha, 36, _XSAX_CHARS)


# --- word utilities ----------------------------------------------------------

def _word_entropy(word: str) -> float:
    """Shannon entropy (base 2) of the character distribution of ``word``.

    Mirrors ``comp_word_entropy`` (``strsplit(word, "^")`` splits per character).
    """
    chars = list(word)
    n = len(chars)
    _, counts = np.unique(np.array(chars), return_counts=True)
    p = counts / n
    return float(-np.sum(p * np.log2(p)))


def _words(symbols: np.ndarray, w: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build length-``w`` symbol words over every sliding window.

    Returns ``(i, seq, entropy)`` where ``i`` is the 0-based window-start index,
    ``seq`` the concatenated word, and ``entropy`` its character entropy. Mirrors
    ``tspredit::ts_data`` + ``paste(collapse="")`` + ``comp_word_entropy``.
    """
    n = len(symbols)
    nwin = n - w + 1
    idx = np.arange(nwin)
    seq = ["".join(symbols[i:i + w]) for i in idx]
    entropy = np.array([_word_entropy(word) for word in seq], dtype=float)
    return idx, seq, entropy


def _grouped(frame: pd.DataFrame) -> pd.DataFrame:
    """Group windows by word: ``total_count`` and (constant-per-group) ``entropy``.

    Mirrors ``group_by(seq) |> summarise(total_count=n(), entropy=mean(entropy))``
    whose output is ordered by ``seq`` ascending.
    """
    grouped = (
        frame.groupby("seq", sort=True)
        .agg(total_count=("seq", "size"), entropy=("entropy", "mean"))
        .reset_index()
    )
    return grouped


def _arrange(grouped: pd.DataFrame) -> pd.DataFrame:
    """R ``arrange(desc(total_count), desc(entropy))`` with ties broken by ``seq`` asc."""
    return grouped.sort_values(
        by=["total_count", "entropy", "seq"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def _consecutive_firsts(refs: np.ndarray) -> np.ndarray:
    """First index of each run of consecutive integers (R ``split`` + ``sapply(min)``)."""
    refs = np.sort(refs)
    runs = np.split(refs, np.where(np.diff(refs) != 1)[0] + 1)
    return np.array([run[0] for run in runs])


# --- motif / discord detectors ----------------------------------------------

class _HmoSaxBase(Harbinger):
    """Shared SAX/XSAX motif-discovery logic (``hmo_sax`` / ``hmo_xsax``)."""

    _transform_cls: type
    _max_alpha: int | None

    def __init__(self, a: int, w: int, qtd: int = 2):
        super().__init__()
        ok = isinstance(a, (int, np.integer)) and a >= 1
        if self._max_alpha is not None and a > self._max_alpha:
            ok = False
        if not ok:
            bound = self._max_alpha if self._max_alpha is not None else "inf"
            raise ValueError(f"alphabet must be between 1 and {bound}")
        self.a = int(a)
        self.w = int(w)
        self.qtd = int(qtd)

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")
        falsemotifs = int(np.round(self.w / 2))
        self._store_refs(serie)

        symbols = self._transform_cls(self.a).transform(self.serie)
        idx, seq, entropy = _words(symbols, self.w)
        data = pd.DataFrame({"i": idx, "seq": seq, "entropy": entropy})

        result = _grouped(data)
        result = result[result["total_count"] >= self.qtd]
        result = _arrange(result)
        data = data[data["seq"].isin(result["seq"])]

        motifs = []
        for word in result["seq"].to_numpy():
            motif = data[data["seq"] == word]
            if len(motif) == 0:
                continue
            refs = motif["i"].to_numpy()
            vec = _consecutive_firsts(refs)
            motif = motif[motif["i"].isin(vec)]
            if len(vec) >= self.qtd:
                motifs.append(motif)
                pos = {r + k for r in refs for k in range(-falsemotifs, falsemotifs + 1)}
                data = data[~data["i"].isin(pos)]

        return self._build_detection(symbols, motifs, kind="motif")

    def _build_detection(self, symbols, parts, kind):
        n_sub = len(self.serie)
        mots = np.zeros(n_sub, dtype=bool)
        seqs = np.full(n_sub, None, dtype=object)
        if parts:
            merged = pd.concat(parts)
            positions = merged["i"].to_numpy()
            mots[positions] = True
            seqs[positions] = merged["seq"].to_numpy()

        detection = self._restore_refs(anomalies=mots)
        detection["type"] = detection["type"].astype(object)
        detection.loc[detection["type"] == "anomaly", "type"] = kind

        seq_full = np.full(self._n, None, dtype=object)
        seqlen_full = np.full(self._n, np.nan)
        seq_full[self.non_na] = seqs
        seqlen_full[self.non_na] = self.w
        detection["seq"] = seq_full
        detection["seqlen"] = seqlen_full
        return detection


class HmoSax(_HmoSaxBase):
    """Motif discovery via SAX: repeated symbolic words mark recurring subsequences.

    Parameters
    ----------
    a:
        Alphabet size (1-26).
    w:
        Word/window size.
    qtd:
        Minimum number of occurrences to classify a word as a motif.

    Porting
    -------
    Mirrors ``hmo_sax.R``. Fidelity: exact (identical detected-index sets on
    ``examples_motifs``).
    """

    _transform_cls = TransSax
    _max_alpha = 26


class HmoXsax(_HmoSaxBase):
    """Motif discovery via XSAX (larger alphanumeric alphabet).

    Parameters
    ----------
    a:
        Alphabet size (1-36).
    w:
        Word/window size.
    qtd:
        Minimum number of occurrences to classify a word as a motif.

    Porting
    -------
    Mirrors ``hmo_xsax.R``. Fidelity: exact.
    """

    _transform_cls = TransXsax
    _max_alpha = None


class HdisSax(Harbinger):
    """Discord discovery via SAX: rare words far from any repeated pattern.

    Parameters
    ----------
    a:
        Alphabet size (1-26).
    w:
        Word/window size.
    qtd:
        Reserved for API parity with :class:`HmoSax` (unused by discord logic).

    Porting
    -------
    Mirrors ``hdis_sax.R``. Fidelity: exact (identical detected-index sets on
    ``examples_motifs``). Note: R's ``hdis_sax`` tags the result ``type ==
    "motif"`` (a copy-paste slip); this port tags it ``type == "discord"``.
    """

    def __init__(self, a: int, w: int, qtd: int = 2):
        super().__init__()
        if not (isinstance(a, (int, np.integer)) and 1 <= a <= 26):
            raise ValueError("alphabet must be between 1 and 26")
        self.a = int(a)
        self.w = int(w)
        self.qtd = int(qtd)

    def detect(self, serie, **kwargs) -> pd.DataFrame:
        if serie is None:
            raise ValueError("No data was provided for computation")
        self._store_refs(serie)

        symbols = TransSax(self.a).transform(self.serie)
        idx, seq, entropy = _words(symbols, self.w)
        full = pd.DataFrame({"i": idx, "seq": seq, "entropy": entropy})

        grouped = _grouped(full)
        discords_seq = grouped[grouped["total_count"] == 1].sort_values(
            by=["entropy", "seq"], ascending=[False, True], kind="mergesort"
        )
        result = grouped[grouped["total_count"] > 1]

        # Remove windows near any repeated ("common") pattern.
        common = full[full["seq"].isin(result["seq"])]
        pos = {i + k for i in common["i"].to_numpy() for k in range(-self.w, self.w + 1)}
        data = full[~full["i"].isin(pos)]

        discords = []
        for word in discords_seq["seq"].to_numpy():
            discord = data[data["seq"] == word]
            if len(discord) == 0:
                continue
            refs = discord["i"].to_numpy()
            discords.append(discord)
            drop = {r + k for r in refs for k in range(-self.w, self.w + 1)}
            data = data[~data["i"].isin(drop)]

        return self._build_detection(symbols, discords)

    def _build_detection(self, symbols, parts):
        n_sub = len(self.serie)
        mots = np.zeros(n_sub, dtype=bool)
        seqs = np.full(n_sub, None, dtype=object)
        if parts:
            merged = pd.concat(parts)
            positions = merged["i"].to_numpy()
            mots[positions] = True
            seqs[positions] = merged["seq"].to_numpy()

        detection = self._restore_refs(anomalies=mots)
        detection["type"] = detection["type"].astype(object)
        detection.loc[detection["type"] == "anomaly", "type"] = "discord"

        seq_full = np.full(self._n, None, dtype=object)
        seqlen_full = np.full(self._n, np.nan)
        seq_full[self.non_na] = seqs
        seqlen_full[self.non_na] = self.w
        detection["seq"] = seq_full
        detection["seqlen"] = seqlen_full
        return detection
