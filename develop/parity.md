# Parity Verification for the `harbinger` → `pyharbinger` Port

> **Purpose.** How we verify that `pyharbinger` (the Python port) reproduces
> `harbinger` (the original R package), specialized to **time-series event detection**
> (anomalies, change points, motifs/discords). It **shares the general methodology** of
> `C:\Git\dal\pydaltoolbox\develop\parity.md` (the backbone for an article on verifying
> behavioral parity between two implementations of the same data-science toolkit). Read
> that document for the full treatment of the central hazard, the taxonomy of
> disagreement, bug localization, and threats to validity; this one adapts the invariants
> to event detection and records the `harbinger`-specific case studies.

---

## 1. The shared protocol (condensed)

1. **Freeze** the input once (here: `examples_anomalies` / `examples_changepoints`,
   embedded as CSV, byte-identical to the R `.RData`); both sides read the same rows.
2. **Choose** the coarsest decision-level invariant per family (for a detector: the
   **set of detected indices**, not the internal score vector).
3. **Compare** decisions, not representations.
4. **Classify** each disagreement: exact / benign-stochastic / benign-numeric-or-solver /
   weak-default / structural / real defect.
5. **Evidence** every "benign" verdict (a seed, an agreeing regime, or a formula).
6. **Localize** defects by bisecting to the smallest primitive and instrumenting it.
7. **Fix** defects on **both** sides (ask before editing the reference); re-check the ripple.
8. **Lock** the outcome with a guard test next to its frozen input.

**Central hazard:** never compare two runs on their own inputs. Every detector is run on
the shared `examples_*` series, so the input is frozen.

### 1b. The `++` gate (back-implementation / genuine improvement)
Unlike a faithful port, a **back-implementation that improves the method (`++`)** is held
to a different bar. **Ação 1++ (Eduardo, 2026-11-21):** the test is **not** "matches the
reference package", it is **"detects the truth as well or better"** — measured by
**SoftED-F1 against the ground-truth `event` column** that `examples_*` already carries,
compared to the F1 of the reference package itself (R, or the original lib). Parity with R
becomes a documented note (`adapted (++)`), not a requirement. Official instrument:
`develop/benchmark_backimpl.py` (F1 vs ground truth, summed over the series).

**Not a blind rule — case by case.** The `Python = canonical rewrite / R = backport`
inversion is the *tendency*, not an absolute. The target is **niche/atypical** packages
(no maintained equivalent, build friction, or auto-selection that breaks parity). Canonical,
stable libs stay wrapped (`scikit-learn`, `scipy`, `ruptures`, `stumpy`, `arch`,
`EMD-signal`, `PyWavelets`). Decided jointly, never by automation.

---

## 2. Invariants per event-detection family

| Family | Invariant compared |
|---|---|
| Pure detectors (fft/hist/fbiad/kswin/page_hinkley) | the **set of detected `idx`** (exact) + residual/score vector (tol ≤1e-4) |
| `HanrMl` (regressor bridge) | detected set + residuals; inherits the pytspredit regressor's tolerance (knn/elm exact; rf/mlp RNG) |
| Evaluation (`HarEval`/`HarEvalSoft`) | the **confusion matrix + metrics** (TP/FP/FN/TN, precision/recall/F1; soft via Hungarian) — exact |
| Change points via `ruptures` (amoc/binseg/pelt) | breakpoint **positions** (approximate — R `changepoint::cpt.*` uses MBIC, not reproduced by `ruptures`) |
| Native change points (cf_lr/scp/joinpoint) | detected set (exact on well-conditioned series) |
| **`++` back-impls (chow/gft, cf_arima/cf_ets, bocpd)** | **SoftED-F1 vs ground truth** (the `++` gate, §1b) — beats the reference |
| Clustering (`HanctKmeans`/`HancMl`) | detected set; init/RNG differs from `stats::kmeans` → documented tolerance |

**Deterministic detectors are the most stringent case:** demand the exact detected set and
the score vector to ≈1e-4; there is no split variance or RNG to hide behind.

---

## 3. Reference side

- Parity references are generated against the **R `harbinger` master** (R 4.5.0 at
  `/c/R/R-4.5.0`; `pkgload::load_all` of `c:\Git\dal\harbinger`).
- **Reference is not ground truth:** where the R behavior is itself wrong, fix it on the R
  side with Eduardo's OK — parity means *agreement after both are correct* (see §5.1).
- For a `++` detector, the "reference" in the gate is the reference package's **F1**, not
  its output; the port is *expected* to diverge and win.

---

## 4. The committed-parity ritual

Each ported detector lands with (a) an R↔Python check on the frozen `examples_*` series
and (b) a guard test in `tests/`. The `++` detectors additionally carry the SoftED-F1 gate
in `develop/benchmark_backimpl.py`. Current guards: `test_changepoint*.py` (incl.
`_structural`, `_cf_forecast`, `_bocpd`, `_extended`), `test_anomaly.py`,
`test_fft_hybrids.py`, `test_evaluation.py`, `test_clustering.py`,
`test_classification.py`, `test_multivariate.py`, `test_datasets.py`, `test_utils.py`.

> **TODO (1.0.0 hardening):** promote the frozen references into `develop/parity/<fam>.R`
> generators + R-free `tests/test_quality_<fam>.py` guards, as done retroactively in
> pytspredit M3. Today several checks are asserted in the unit tests rather than
> regenerated from a committed `.R` script.

---

## 5. Case studies

### 5.1 `hmu_pca` — a degenerate detector in the reference (fixed on both sides)
R `hmu_pca` kept **all** principal components → exact reconstruction → residual = float
noise (~1e-31): the detector was degenerate and detected nothing real. **Parity testing
found a bug in the original, not the port.** Fix (Eduardo's OK): added `num_components`/
`variance` params; reconstruct from the leading components (default = 90% variance, capped
at `p-1` so ≥1 component is always dropped). Now the default recovers the real anomaly (idx
50 in `simple`/`contextual`, res max 1.054). Python `HmuPca` was aligned to the fixed R →
exact parity. Lesson: **reference is not ground truth**; a detector that "never disagrees
with itself" can still be wrong.

### 5.2 The `++` back-implementations — improvement as an evidenced claim (0.4.4–0.4.6)
The 0.4.x natives began as faithful ports (the BOCPD was nearly line-by-line of `ocp`).
Eduardo asked to turn them into genuine **`++`** rereadings — advancing the method to
*contribute*, not copy. Each passed the §1b gate (SoftED-F1 vs ground truth, sum over 7
series, max 7.0):

| ++ release | Detectors | Innovation | Gate (port++ / faithful / R) |
|---|---|---|---|
| 0.4.4 strucchange++ | `HcpChow`, `HcpGft` | mean+variance cost (not mean-only) + LWZ penalty (Liu-Wu-Zidek) vs BIC | Chow **3.73** / 3.40 / 3.31 · GFT **3.17** / 2.44 / 2.20 |
| 0.4.5 forecast++ | `HcpCfArima`, `HcpCfEts` | native robust ChangeFinder = SDAR (Takeuchi-Yamanishi) ∪ mean+variance break; pure numpy (drops `statsmodels`) | cf_arima **3.84** / 2.78 / 2.92 · cf_ets **3.64** / 2.66 / 2.66 |
| 0.4.6 ocp++ | `HcpBocpd` | Bayesian **autoregressive** UPM (slope/dynamics, not just level/var) + log-space ε-pruning (streaming) + predictive-surprise extraction | bocpd **3.59** / (R `ocp` 1.12) — wins 6/7, ~3.2× |

Reusable findings: (1) SDAR alone is blind to smooth slope ramps → **union with
mean+variance** recovers them (forecast++); (2) BOCPD-AR recovers slope changes that
mean+variance misses (the inverted mirror of the same gap) — union there *hurts*, so BOCPD
stayed pure. All three kept `References:` to the original **and** cite the papers that
inspired each `++`.

### 5.3 Dependency reduction as the companion of `++`
The `++` pass is also a **dep-reduction** pass across the whole DAL stack: Python
pyharbinger dropped `statsmodels`; R harbinger dropped `strucchange` and `ocp` entirely.
The last R `forecast` user is `hanr_arima` (anomaly) — its `++` reread (SDAR-based
`HanrArima`) is the next step and removes `forecast` from R for good.

### 5.4 `ruptures` wheel friction — version drift is a parity threat (M0/M3)
On Python 3.14 only `ruptures` 1.0.6 ships a pure wheel (1.1.x needs a C compiler). Pin
relaxed to `ruptures>=1.0`; install with `pip install --prefer-binary`. Lesson: a dep's
**wheel availability for the target Python** is a release gate — and a motive for the
`ruptures`→native back-implementation candidate (P3).

---

## 6. Parity matrix (status)

Verdict legend: **exact** (≤~1e-4, deterministic set+scores) · **approximate** (positions
overlap; backend/penalty differs) · **stochastic** (RNG: seed/regime faithful) ·
**adapted ++** (improves on purpose; gate = beats the reference F1, §1b) · **🐞** (bug
fixed on both sides).

| Family (R → Py) | Invariant | Verdict | Evidence |
|---|---|---|---|
| `harbinger`/utils/deviations/filters/selection | structure + values | **exact** | ≤1e-5 on all primitives |
| `har_eval`/`har_eval_soft` → `HarEval`/`HarEvalSoft` | confusion + metrics | **exact** | hard + soft (Hungarian) ≤1e-5; keys unified to TP/FP/FN/TN |
| `hanr_fft`/`hanr_hist`/`hanr_fbiad` → `HanrFft`/`HanrHistogram`/`HanrFbiad` | detected set + residuals | **exact** | match on `examples_anomalies$simple` (anomaly idx 50) ≤1e-4 |
| `hanr_ml` → `HanrMl` | detected set + residuals | **exact / stochastic** | kNN exact; ELM `nhid<input_size`+purelin → RNG (use `nhid≥input_size`) |
| `hanr_arima` → `HanrArima` | SoftED-F1 vs truth | **adapted ++** | reuses `pytspredit.fit_arima` (DAL auto.arima) not `forecast`; 9.03 > R `hanr_arima` 8.58 (sum/10 on `examples_anomalies`, `auto="aicc"`, §5.x) |
| `hanr_fft_sma` → `HanrFftSma` | detected set | **exact** | all 10 `examples_anomalies`; `spec.pgram` port picks the same MA window (27/64/171); R exposes no `res`, so the set of idx is the invariant |
| `hanr_fft_amoc/binseg(+cusum)` → `HanrFft*` | detected set + res_sum | **exact** | amoc/binseg/amoc_cusum/binseg_cusum all idx 50; cutindex 5/5/4/39 match `cpt.mean` |
| `hcp_amoc`/`hcp_binseg` → `HcpAmoc`/`HcpBinseg` | breakpoint positions | **approximate** | match 3/4, ~off-by-1 (ruptures ≠ MBIC) |
| `hcp_pelt` → `HcpPelt` | breakpoint positions | **approximate** | penalty `3·log n` (MBIC not reproducible in ruptures) |
| `hcp_kswin` → `HcpKswin` | detected set | **stochastic** | KS subsample RNG (`random_state`) |
| `hcp_page_hinkley` → `HcpPageHinkley` | detected set | **exact** | 29/36/110/∅ on the 4 series |
| `hcp_cf_lr` → `HcpCfLr` | detected set | **exact** | all 7 `examples_changepoints` |
| `hcp_scp` → `HcpScp` | detected set | **exact** | all 7 series |
| `hcp_joinpoint` → `HcpJoinpoint` | detected set | **exact** (all 7) | tolerant tie-break (`_JOINPOINT_TOL`) + RSS floor (`_JOINPOINT_RSS_FLOOR`) make the DP backend-independent → R↔Python identical on all 7 series (incl. sinusoidal ties and the `simple` ramp, which now cleanly yields the single cp 50); same fix backported to R |
| `hcp_chow`/`hcp_gft` → `HcpChow`/`HcpGft` | SoftED-F1 vs truth | **adapted ++** | 3.73 / 3.17 > R (§5.2) |
| `hcp_cf_arima`/`hcp_cf_ets` → `HcpCfArima`/`HcpCfEts` | SoftED-F1 vs truth | **adapted ++** | 3.84 / 3.64 > R (§5.2); pure numpy |
| `hcp_bocpd` → `HcpBocpd` | SoftED-F1 vs truth | **adapted ++** | 3.59 vs R `ocp` 1.12 (§5.2) |
| `hanct_kmeans` → `HanctKmeans` | detected set | **stochastic** | event=50, centers=3; k-means init RNG |
| `hanc_ml` → `HancMl` | detected set | **exact / stochastic** | `ClaKnn(k=3)` exact (events 10,21); probas backend-dependent |
| `hmu_pca` → `HmuPca` | detected set + residuals | **exact 🐞** | idx 50, res max 1.054 after the R bug fix (§5.1) |
| `har_ensemble_fuzzy` → `HarEnsembleFuzzy` | consensus set | **adapted** | fuzzy consensus + absolute floor (0.4.7) |
| `trans_sax`/`trans_xsax` → `TransSax`/`TransXsax` | symbolic codes | **exact** | identical SAX/XSAX words on `simple` (z-norm + type-7 quantile + `cut` include.lowest + base-26/36 padded codes) |
| `hmo_sax`/`hmo_xsax`/`hdis_sax` → `HmoSax`/`HmoXsax`/`HdisSax` | detected set | **exact** | identical detected-idx sets on all 3 `examples_motifs` series (simple + mitdb100/102). 🐞 R `hdis_sax` tags `type=="motif"` (copy-paste slip); port uses `type=="discord"` (parity is on the idx set) |

---

## 7. Open items
- Confirm with Eduardo the R `hdis_sax` `type=="motif"` copy-paste slip (port uses
  `"discord"`); fix in the R backport if agreed (idx-set parity is unaffected).
- Promote the asserted checks into `develop/parity/<fam>.R` + `tests/test_quality_*`
  (R-free), mirroring the pytspredit M3 backfill.
- Finish the `hanr_arima`++ gate on `examples_anomalies` (10 series) and record its row.
- When the benchmark/streaming datasets land, add element-wise dataset parity guards
  (mirror pytspredit `test_quality_datasets.py`).
