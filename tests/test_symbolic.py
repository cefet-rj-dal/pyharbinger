"""Tests for the symbolic SAX/XSAX transforms and motif/discord detectors.

Faithful port of R ``trans_sax`` / ``trans_xsax`` / ``hmo_sax`` / ``hmo_xsax`` /
``hdis_sax``: the gate is exact parity of the detected index set against the R
reference on every ``examples_motifs`` series, plus the symbolic encoding itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyharbinger import (
    HdisSax,
    HmoSax,
    HmoXsax,
    TransSax,
    TransXsax,
    datasets,
)
from pyharbinger.symbolic import _word_entropy

MOTIFS = datasets.examples_motifs()


def _events(detection):
    return detection.loc[detection["event"], "idx"].tolist()


# R harbinger references on examples_motifs (0-based detected indices), verified
# via Rscript sourcing harbinger base + trans_sax/xsax + hmo/hdis_sax.
# Detectors: hmo_sax(26,3,3), hmo_xsax(37,3,3), hdis_sax(26,3,3).
R_SIMPLE = {
    "hmo_sax": [24, 49, 74],
    "hmo_xsax": [24, 49, 74],
    "hdis_sax": [16, 20, 38, 42, 63, 67, 85, 89, 93],
}
# Full-series discord references (motif sets are large; checked in a sweep test).
R_DISCORD = {"simple": [16, 20, 38, 42, 63, 67, 85, 89, 93], "mitdb100": [], "mitdb102": [3337]}

# First SAX/XSAX symbols of the z-normalized `simple` series (R transform output).
R_SAX_SIMPLE_HEAD = ["K", "K", "J", "I", "H", "F", "E", "D", "C", "B", "B", "A", "A", "A", "A"]
R_XSAX_SIMPLE_HEAD = [
    "0F", "0E", "0E", "0C", "0A", "08", "06", "04", "02", "02", "01", "00", "00", "00", "01",
]


@pytest.fixture
def simple():
    return MOTIFS["simple"]["serie"].to_numpy()


# --- transforms --------------------------------------------------------------

def test_trans_sax_matches_r(simple):
    assert list(TransSax(26).fit(simple).transform(simple)[:15]) == R_SAX_SIMPLE_HEAD


def test_trans_xsax_matches_r(simple):
    assert list(TransXsax(37).fit(simple).transform(simple)[:15]) == R_XSAX_SIMPLE_HEAD


def test_trans_sax_alphabet_size(simple):
    # 26 symbols -> single A..Z characters.
    symbols = set(TransSax(26).transform(simple))
    assert symbols.issubset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def test_trans_sax_rejects_tiny_alphabet():
    with pytest.raises(ValueError):
        TransSax(1)


# --- word entropy ------------------------------------------------------------

def test_word_entropy():
    assert _word_entropy("AAA") == pytest.approx(0.0)
    assert _word_entropy("AAB") == pytest.approx(0.9182958, abs=1e-6)
    assert _word_entropy("ABC") == pytest.approx(1.584963, abs=1e-6)


# --- motif / discord exact parity on `simple` --------------------------------

def test_hmo_sax_simple(simple):
    assert _events(HmoSax(26, 3, 3).fit(simple).detect(simple)) == R_SIMPLE["hmo_sax"]


def test_hmo_xsax_simple(simple):
    assert _events(HmoXsax(37, 3, 3).fit(simple).detect(simple)) == R_SIMPLE["hmo_xsax"]


def test_hdis_sax_simple(simple):
    assert _events(HdisSax(26, 3, 3).fit(simple).detect(simple)) == R_SIMPLE["hdis_sax"]


# --- discord parity across all series (incl. the large ECG series) -----------

@pytest.mark.parametrize("name", ["simple", "mitdb100", "mitdb102"])
def test_hdis_sax_all_series(name):
    serie = MOTIFS[name]["serie"].to_numpy()
    assert _events(HdisSax(26, 3, 3).fit(serie).detect(serie)) == R_DISCORD[name]


# --- detection-table contract ------------------------------------------------

def test_motif_detection_contract(simple):
    det = HmoSax(26, 3, 3).fit(simple).detect(simple)
    assert list(det.columns[:3]) == ["idx", "event", "type"]
    flagged = det[det["event"]]
    assert (flagged["type"] == "motif").all()
    assert (flagged["seqlen"] == 3).all()
    # Each flagged window carries its symbolic word (length w).
    assert flagged["seq"].notna().all()
    assert (flagged["seq"].str.len() == 3).all()


def test_discord_detection_type(simple):
    det = HdisSax(26, 3, 3).fit(simple).detect(simple)
    flagged = det[det["event"]]
    assert (flagged["type"] == "discord").all()
    assert (flagged["seqlen"] == 3).all()


def test_detect_none_raises():
    with pytest.raises(ValueError):
        HmoSax(26, 3, 3).detect(None)


def test_restore_over_nas():
    # A leading NA shifts the original indexing; detections must map back correctly.
    base = MOTIFS["simple"]["serie"].to_numpy()
    serie = np.concatenate([[np.nan], base])
    det = HmoSax(26, 3, 3).fit(serie).detect(serie)
    # Same motif positions, shifted by one (the NA row is never flagged).
    assert _events(det) == [i + 1 for i in R_SIMPLE["hmo_sax"]]
    assert not det.loc[0, "event"]
