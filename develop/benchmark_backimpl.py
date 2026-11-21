"""Action-1++ benchmark for the back-implementation "++" re-engineering pass.

Since the ``examples_changepoints`` datasets ship a ground-truth ``event``
column, the acceptance criterion for the re-engineered ("++") back-implementations
is no longer "matches the reference R package" but **"detects the ground truth at
least as well as the original package"**. This script measures the SoftED-F1
(tolerant soft evaluation, :class:`HarEvalSoft`) of each detector against the
ground truth, on each of the seven series, and compares it to the F1 of the
reference package's own detections.

Run from the repo root::

    python develop/benchmark_backimpl.py

NaN F1 (a detector that fires nothing on a series that has a true event) is
counted as 0.0, so the per-detector total is the sum of SoftED-F1 over the seven
series (max 7.0). Higher is better; a "++" version should beat (or at least match)
its ``R-ref`` column.
"""
from __future__ import annotations

import numpy as np

from pyharbinger import (
    HarEvalSoft,
    HcpBocpd,
    HcpCfArima,
    HcpCfEts,
    HcpChow,
    HcpGft,
    datasets,
)

K = 15  # SoftED tolerance window (samples)

# Reference R-package detected indices (0-based), measured in the faithful-port
# milestones (0.4.1 strucchange / 0.4.2 forecast / 0.4.3 ocp). These are the
# baseline the re-engineered detectors must beat on detection quality.
R_REFERENCE = {
    "chow": {
        "simple": [50], "sinusoidal": [50], "incremental": [86], "abrupt": [100],
        "volatility": [149], "increasing_amplitude": [817], "complex": [200],
    },
    "gft": {
        "simple": [49, 64], "sinusoidal": [15, 30, 45, 60, 79], "incremental": [86],
        "abrupt": [100, 149], "volatility": [123, 170],
        "increasing_amplitude": [629, 816], "complex": [200, 307, 391],
    },
    "cf_arima": {
        "simple": [50], "sinusoidal": [55], "incremental": [3, 13, 50, 103, 150],
        "abrupt": [3, 100, 102, 136, 194, 195],
        "volatility": [136, 155, 160, 167, 168, 192, 194, 195],
        "increasing_amplitude": [699], "complex": [200],
    },
    "cf_ets": {
        "simple": [50], "sinusoidal": [], "incremental": [13, 50, 103, 113, 150],
        "abrupt": [13, 100, 109, 136, 194],
        "volatility": [136, 149, 160, 173, 192, 194],
        "increasing_amplitude": [199, 699], "complex": [200],
    },
    "bocpd": {
        "simple": [8, 13, 18, 23, 28, 33, 38, 59, 69, 74, 79, 84, 89],
        "sinusoidal": [7, 23, 32, 37, 60, 69, 74, 89],
        "incremental": [128, 175], "abrupt": [100, 149], "volatility": [123, 149],
        "increasing_amplitude": [
            8, 25, 38, 58, 70, 89, 101, 121, 133, 152, 165, 182, 192, 200, 218,
            228, 245, 258, 278, 290, 309, 321, 353, 371, 384, 403, 415, 435, 448,
            465, 476, 501, 509, 526, 539, 559, 571, 590, 602, 622, 634, 653, 664,
            700, 713, 727, 760, 778, 791, 810, 822, 854, 872, 885, 904, 916, 936,
            949, 966, 976,
        ],
        "complex": [9, 101, 107, 200, 311, 329, 351, 366, 372, 385, 390, 410],
    },
}

DETECTORS = {
    "chow": HcpChow,
    "gft": HcpGft,
    "cf_arima": HcpCfArima,
    "cf_ets": HcpCfEts,
    "bocpd": HcpBocpd,
}


def softed_f1(detection_bool: np.ndarray, event_bool: np.ndarray) -> float:
    f1 = HarEvalSoft(sw_size=K).evaluate(detection_bool, event_bool)["F1"]
    return 0.0 if (f1 is None or np.isnan(f1)) else float(f1)


def _bool_from_idx(idx, n: int) -> np.ndarray:
    b = np.zeros(n, dtype=bool)
    b[[i for i in idx if 0 <= i < n]] = True
    return b


def run(detectors=None) -> dict[str, tuple[float, float]]:
    """Return ``{name: (our_total_f1, r_ref_total_f1)}`` over the 7 series."""
    cp = datasets.examples_changepoints()
    detectors = detectors or DETECTORS
    totals: dict[str, tuple[float, float]] = {}
    for name, cls in detectors.items():
        print(f"\n== {name} ==")
        print(f"{'series':22s} {'ours F1':>8s} {'R-ref F1':>9s}  n_gt")
        ours = rref = 0.0
        for s, df in cp.items():
            n = len(df)
            gt = df["event"].to_numpy(dtype=bool)
            det = cls().detect(df["serie"].to_numpy(dtype=float))
            f_ours = softed_f1(det["event"].to_numpy(dtype=bool), gt)
            f_r = softed_f1(_bool_from_idx(R_REFERENCE[name].get(s, []), n), gt)
            ours += f_ours
            rref += f_r
            print(f"{s:22s} {f_ours:8.3f} {f_r:9.3f}  {int(gt.sum())}")
        print(f"{'TOTAL (max 7.0)':22s} {ours:8.3f} {rref:9.3f}")
        totals[name] = (ours, rref)
    return totals


if __name__ == "__main__":
    run()
