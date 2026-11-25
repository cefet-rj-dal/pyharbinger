"""pyharbinger -- A unified time series event detection framework.

Python port of the R ``harbinger`` framework, the capstone of the DAL family.
It builds on ``pydaltoolbox`` (the Experiment Line foundation) and ``pytspredit``
(time series prediction) to unify many event-detection methods -- anomalies,
change points, and motifs/discords -- under one interface (``fit`` -> ``detect``
-> ``evaluate``) and a common evaluation protocol, wrapping numpy / pandas /
scikit-learn / scipy / ruptures with thin adapters.
"""
from __future__ import annotations

from . import datasets, utils
from .anomaly import (
    HanrArima,
    HanrFbiad,
    HanrFft,
    HanrFftAmoc,
    HanrFftAmocCusum,
    HanrFftBinseg,
    HanrFftBinsegCusum,
    HanrHistogram,
    HanrMl,
)
from .changepoint import (
    HcpAmoc,
    HcpBinseg,
    HcpBocpd,
    HcpCfArima,
    HcpCfEts,
    HcpCfLr,
    HcpChow,
    HcpGft,
    HcpJoinpoint,
    HcpKswin,
    HcpPageHinkley,
    HcpPelt,
    HcpScp,
)
from .classification import HancMl
from .clustering import HanctKmeans
from .ensemble import HarEnsembleFuzzy
from .evaluation import HarEval, HarEvalSoft
from .harbinger import Harbinger
from .multivariate import HmuPca
from .utils import (
    har_candidate_selection_firstgroup,
    har_candidate_selection_highgroup,
    har_candidate_selection_referencedistribution,
    har_deviation_huber,
    har_deviation_l1,
    har_deviation_l2,
    har_filter_boxplot,
    har_filter_gaussian,
    har_filter_grubbs,
    har_filter_mad,
    har_filter_none,
    har_filter_ratio,
    mas,
)

__version__ = "0.4.8"

__all__ = [
    "HancMl",
    "HanctKmeans",
    "HanrArima",
    "HanrFbiad",
    "HanrFft",
    "HanrFftAmoc",
    "HanrFftAmocCusum",
    "HanrFftBinseg",
    "HanrFftBinsegCusum",
    "HanrHistogram",
    "HanrMl",
    "HarEnsembleFuzzy",
    "HarEval",
    "HarEvalSoft",
    "Harbinger",
    "HcpAmoc",
    "HcpBinseg",
    "HcpBocpd",
    "HcpCfArima",
    "HcpCfEts",
    "HcpCfLr",
    "HcpChow",
    "HcpGft",
    "HcpJoinpoint",
    "HcpKswin",
    "HcpPageHinkley",
    "HcpPelt",
    "HcpScp",
    "HmuPca",
    "datasets",
    "har_candidate_selection_firstgroup",
    "har_candidate_selection_highgroup",
    "har_candidate_selection_referencedistribution",
    "har_deviation_huber",
    "har_deviation_l1",
    "har_deviation_l2",
    "har_filter_boxplot",
    "har_filter_gaussian",
    "har_filter_grubbs",
    "har_filter_mad",
    "har_filter_none",
    "har_filter_ratio",
    "mas",
    "utils",
]
