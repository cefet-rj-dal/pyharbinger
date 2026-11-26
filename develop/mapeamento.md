# Mapeamento R ↔ Python (harbinger ↔ pyharbinger)

Correspondência entre a API pública do pacote R **harbinger** (76 `export()` no
NAMESPACE) e o port Python **pyharbinger**. Convenções: construtores S3 em
`snake_case` viram **PascalCase**; genéricos S3 (`fit`/`detect`/`evaluate`) viram
**métodos de instância**; `data.frame` → `pandas.DataFrame`; série = `pd.Series`/vetor;
detecção = DataFrame `[idx, event, type, (seq, seqlen)]`.

> A **matriz de paridade autoritativa** (invariante · veredito · evidência · guard)
> vive em `parity.md`. Aqui fica só a **correspondência de símbolos** e o status.

**Legenda:** ✅ portado · 🔜 Ciclo 1 (baixa dep, falta) · 🧩 Ciclo 2 (dep pesada, *extra*) ·
⬜ pendente (infra/streaming) · ➕ só no Python (++ sem contraparte 1:1)

## Núcleo / contrato (Experiment Line)
| R (arquivo) | R (símbolo) | Python | Status |
|---|---|---|---|
| `harbinger.R` | `harbinger`, `detect` | `Harbinger(DalBase)`, `.detect()` | ✅ |
| `harbinger_utils.R` | `harutils`, `har_store_refs`/`har_restore_refs` | `utils`, `_store_refs`/`_restore_refs` | ✅ |
| `harbinger_utils_deviation_measures.R` | desvios L1/L2/Huber | `har_deviation_l1/l2/huber` | ✅ |
| `harbinger_utils_filter_criteria.R` | none/boxplot/gaussian/mad/grubbs/ratio | `har_filter_*` → `FilterResult` | ✅ |
| `harbinger_utils_candidate_selection.R` | firstgroup/highgroup/referencedistribution | `har_candidate_selection_*` → `SelectionResult` | ✅ |
| `mas.R` | `mas` | `mas` | ✅ |

## Avaliação
| R | Python | Status |
|---|---|---|
| `har_eval.R` (`har_eval`) | `HarEval` | ✅ |
| `har_eval_soft.R` (`har_eval_soft`, SoftED) | `HarEvalSoft` (Hungarian via `scipy.optimize.linear_sum_assignment`) | ✅ |

## Anomalia (`hanr_*`, `han_*`)
| R | Python | Backend | Status |
|---|---|---|---|
| `hanr_ml.R` | `HanrMl` | ponte regressor pytspredit (prevê 1-passo sobre janelas) | ✅ |
| `hanr_fft.R` | `HanrFft` | FFT passa-alta | ✅ |
| `hanr_hist.R` | `HanrHistogram` | densidade de bins (`r_pretty` = `pretty()`/`hist()`) | ✅ |
| `hanr_fbiad.R` | `HanrFbiad` | inercial forward+backward | ✅ |
| `hanr_fft_amoc.R` / `hanr_fft_binseg.R` | `HanrFftAmoc` / `HanrFftBinseg` | FFT + cutoff via ruptures | ✅ |
| `hanr_fft_amoc_cusum.R` / `hanr_fft_binseg_cusum.R` | `HanrFftAmocCusum` / `HanrFftBinsegCusum` | idem + CUSUM | ✅ |
| `hanr_fft_sma.R` | `HanrFftSma` | periodograma (`spec.pgram`) → janela da média móvel + resíduo da SMA | ✅ (port fiel; paridade exata nas 10 séries) |
| `hanr_arima.R` | `HanrArima` | **ARIMA++** reusa `pytspredit.fit_arima` (substituto DAL do auto.arima) + resíduos in-sample → derruba `forecast` do R | ✅ ➕ |
| `hanr_garch.R` | `HanrGarch` | `arch` (sGARCH+ARMA) | 🧩 `[garch]` |
| `hanr_emd.R` / `hanr_remd.R` / `hanr_rtad.R` | `HanrEmd` / `HanrRemd` / `HanrRtad` | EMD (`hht` no R) | 🧩 `[emd]` |
| `hanr_wavelet.R` | `HanrWavelet` | wavelet (`wavelets` no R) | 🧩 `[wavelet]` |
| `han_autoencoder.R` | `HanAutoencoder` | autoencoder real (torch — `AutoencBaseEd` do pydaltoolbox é STUB identidade) | 🧩 `[deep]` |

## Change point (`hcp_*`)
| R | Python | Backend | Status |
|---|---|---|---|
| `hcp_amoc.R` / `hcp_binseg.R` / `hcp_pelt.R` | `HcpAmoc` / `HcpBinseg` / `HcpPelt` | `ruptures` (custo normal) | ✅ |
| `hcp_kswin.R` / `hcp_page_hinkley.R` | `HcpKswin` / `HcpPageHinkley` | online (KS / Page-Hinkley) | ✅ |
| `hcp_cf_lr.R` | `HcpCfLr` | ChangeFinder 2-estágios (regressão linear) | ✅ |
| `hcp_scp.R` / `hcp_joinpoint.R` | `HcpScp` / `HcpJoinpoint` | OLS segmentado / joinpoint (DP) nativos | ✅ |
| `hcp_chow.R` / `hcp_gft.R` | `HcpChow` / `HcpGft` | **strucchange++** nativo (mean+variance + LWZ) | ✅ ➕ |
| `hcp_cf_arima.R` / `hcp_cf_ets.R` | `HcpCfArima` / `HcpCfEts` | **forecast++** nativo (ChangeFinder robusto SDAR + mean+var) | ✅ ➕ |
| `hcp_bocpd.R` (+`bocpd_ar.R`) | `HcpBocpd` | **ocp++** nativo (BOCPD com UPM autorregressivo) | ✅ ➕ |
| `hcp_garch.R` | `HcpGarch` | `arch` + OLS 1º estágio | 🧩 `[garch]` |
| `hcp_waypoint.R` | `HcpWaypoint` | autoencoder real (torch) | 🧩 `[deep]` |

## Clustering / classificação / multivariado
| R | Python | Backend | Status |
|---|---|---|---|
| `hanct_kmeans.R` | `HanctKmeans` | k-means sklearn (anomalia seq=1 / discord seq>1) | ✅ |
| `hanc_ml.R` | `HancMl` | classificador pydaltoolbox/sklearn (caminho multivariado) | ✅ |
| `hmu_pca.R` | `HmuPca` | PCA sklearn (🐞 bug do R corrigido — ver parity.md) | ✅ |
| `hanct_dtw.R` | `HanctDtw` | DTW (`dtwclust` no R → `dtaidistance`/`tslearn`) | 🧩 `[dtw]` |

## Motif / discord (simbólico + matrix profile)
| R | Python | Backend | Status |
|---|---|---|---|
| `trans_sax.R` / `trans_xsax.R` | `TransSax` / `TransXsax` | SAX/xSAX nativo | ✅ (paridade exata) |
| `hmo_sax.R` / `hmo_xsax.R` / `hdis_sax.R` | `HmoSax` / `HmoXsax` / `HdisSax` | motif/discord via SAX nativo | ✅ (paridade exata nas 3 séries) |
| `hmo_mp.R` / `hdis_mp.R` | `HmoMp` / `HdisMp` | matrix profile (`tsmp` no R → `stumpy`) | 🧩 `[motif]` |

## Ensembles / visualização
| R | Python | Status |
|---|---|---|
| `har_ensemble_fuzzy.R` (`har_ensemble_fuzzy`, `fuzzify`) | `HarEnsembleFuzzy` (consenso fuzzy + piso absoluto) | ✅ |
| `har_ensemble.R` (`har_ensemble`) | `HarEnsemble` (voto majoritário) | ✅ (paridade exata) |
| `har_plot.R` / `har_plot_ensemble.R` | `har_plot` / `har_ensemble_plot` / `har_ensemble_plot_models` | ✅ (matplotlib, dep do núcleo; adaptado) |

## Streaming / online / infra (fecha o Ciclo 1)
| R | Python | Status |
|---|---|---|
| `har_online_memory.R` (`har_memory_full/last_observations/sliding`) | `HarMemoryFull`/`HarMemoryLastObservations`/`HarMemorySliding` + `har_memory_label` | ✅ (paridade exata) |
| `har_online_sources.R` (`har_source_callback/dataframe/kafka/simulated`, `source_info`, `ingest`, `next_observation`, `is_finished`) | `HarSourceSimulated`/`HarSourceDataframe`/`HarSourceCallback` (`.next_observation`/`.info`) | ✅ (kafka pulado — reticulate-only no R) |
| `har_online_executor.R` / `har_online_session.R` (`har_online_detect_only/incremental/refit_full`, `run_online`, `step_online`, `har_online_session`) | `HarOnlineRefitFull`/`HarOnlineDetectOnly`/`HarOnlineIncremental` + `HarOnlineSession` (`.fit`/`.ingest`/`.step`/`.run`/`.is_finished`/`.collect_*`) | ✅ (paridade exata: eventos+DP+DL) |
| `har_stream_eval.R` / `har_stream_experiment.R` (`collect_batch_log/detection/trace`) | avaliação em streaming | ⬜ |
| `united.R` / `united_loadfulldata.R` (`united`, `loadfulldata`) | experimento "united" + datasets completos | ⬜ (replicar o `loadfulldata` do pytspredit — ver planejamento.md) |

## Datasets
| R | Python | Status |
|---|---|---|
| `data.R` — `examples_anomalies` (10 séries) | `datasets.examples_anomalies()` | ✅ (CSV, paridade exata) |
| `data.R` — `examples_changepoints` (7 séries) | `datasets.examples_changepoints()` | ✅ (CSV, paridade exata) |
| `data.R` — `examples_motifs` (3 séries: simple + mitdb100/102 c/ coluna `symbol`) | `datasets.examples_motifs()` | ✅ (CSV, paridade exata) |
| miniaturas de benchmark (Yahoo A1–A4, gecco, MIT-BIH, NAB, oil_3w, UCR) + `examples_motifs`/`examples_harbinger` | `datasets.*` + `loadfulldata` | ⬜ (release de streaming/datasets — padrão miniatura + `[bench]` do pytspredit) |
