"""pyharbinger -- A unified time series event detection framework.

Python port of the R ``harbinger`` framework, the capstone of the DAL family.
It builds on ``pydaltoolbox`` (the Experiment Line foundation) and ``pytspredit``
(time series prediction) to unify many event-detection methods -- anomalies,
change points, and motifs/discords -- under one interface (``fit`` -> ``detect``
-> ``evaluate``) and a common evaluation protocol, wrapping numpy / pandas /
scikit-learn / scipy / ruptures with thin adapters.
"""
from __future__ import annotations

from . import datasets

__version__ = "0.1.0"

__all__ = [
    "datasets",
]
