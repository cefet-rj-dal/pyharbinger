Retomada do pyharbinger (c:\Git\dal\pyharbinger). Rode a partir desta pasta. Leia primeiro a
memória do projeto (pyharbinger-port, back-implementation-initiative, boeing-versioning,
verify-dependency-backends — se existirem nesta máquina) e, no develop/: planejamento.md (só o
que falta), historico.md (a história), mapeamento.md (R↔Python), parity.md (metodologia + matriz).

PRINCÍPIOS (confirme em poucas linhas e siga)
- Python = releitura canônica / R = backport; **incrementar métodos ++ + reduzir dependência**
  em todo o stack DAL. **Não é regra cega** — caso a caso com o Eduardo; libs canônicas
  (sklearn/scipy/ruptures/stumpy/arch/...) seguem embrulhadas.
- **Gate Ação 1++** = detectar a verdade tão bem ou melhor via **SoftED-F1 vs a coluna `event`**
  (ground truth), comparado ao F1 do pacote de referência — não paridade com ele. Instrumento:
  develop/benchmark_backimpl.py. Port fiel continua no gate de paridade com o R.
- Doc: Teoria + References (paper) + Porting em toda classe; dep-removal vai no commit, não na doc.
- Numeração Boeing só no R após 1.0.0.

ESTADO (2026-11-25)
- **PyPI latest = 0.4.6.** `main` tem a **0.4.7 (ensemble fuzzy, `HarEnsembleFuzzy`) NÃO
  publicada** (commits `f2d80d1`, `1c7cfc8` — este último moveu o conversor p/ notebooks/).
  `main` é a única branch de código (milestones apagados).
- **Pronto:** núcleo + avaliação (`HarEval`/`HarEvalSoft`) + anomalias base (`HanrMl`/`HanrFft`/
  `HanrHistogram`/`HanrFbiad`) + 4 FFT-híbridos + change points (ruptures `HcpAmoc/Binseg/Pelt`
  + nativos `HcpCfLr/Scp/Joinpoint/Kswin/PageHinkley` + os 3 **++**: `HcpChow/Gft` strucchange++,
  `HcpCfArima/CfEts` forecast++, `HcpBocpd` ocp++) + clustering (`HanctKmeans`) + multivariado
  (`HancMl`/`HmuPca`) + `HarEnsembleFuzzy`. Ver mapeamento.md para o que falta.
- **Deps:** Python largou `statsmodels`; núcleo = pydaltoolbox/pytspredit/numpy/pandas/sklearn/
  scipy/ruptures (+matplotlib p/ plots). R harbinger master **2.1.767**: largou `strucchange` e
  `ocp`; ainda importa `forecast` (último usuário = `hanr_arima`), `changepoint`, e as libs do
  Ciclo 2 (`hht`/`wavelets`/`rugarch`/`tsmp`/`dtwclust`).

PRÓXIMO (decidir comigo qual — ver planejamento.md)
- (a) **Publicar a 0.4.7** (ensemble fuzzy) como está, ou juntá-la ao próximo marco.
- (b) **`hanr_arima`++** — detector de anomalia por AR robusto (SDAR), reusando
  `changefinder_robust` / `_sdar_ar_scores`/`_causal_ma`. Cria `HanrArima` (Python, sem
  statsmodels) e permite o backport R **largar o `forecast` de vez**. Gate Ação 1++ sobre
  `examples_anomalies` (10 séries de ANOMALIA). Leia `harbinger/R/hanr_arima.R` (forecast) e
  `R/changefinder_robust.R` (SDAR) antes.
- Depois, Ciclo 1: `HanrFftSma` → simbólico (SAX/motif/discord) → ensemble simples + plots →
  streaming/online + datasets de benchmark (⚠️ replicar o `loadfulldata` do pytspredit:
  miniaturas + extra `[bench]`). Ciclo 2 (extras, a refinar): motif-MP/GARCH/EMD/wavelet/DTW/deep.

WORKFLOW
1. Branch por milestone saindo da `main`; nada na main sem o Eduardo pedir.
2. Verde sempre: `ruff check .` + `pytest` + `python notebooks/conversor.py` (UM argumento:
   família ou arquivo). Datasets congelados em src/pyharbinger/data/ (examples_anomalies/
   changepoints.csv).
3. Paridade R via `/c/R/R-4.5.0/bin/Rscript.exe` + `pkgload::load_all` de c:\Git\dal\harbinger;
   comparar o CONJUNTO de índices detectados (idx 0-based). Bug em qualquer lado → corrige nos dois.
4. Atualizar develop/{planejamento,historico,mapeamento,parity}.md + a memória a cada marco.
5. Commits terminam com `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Pausa p/ revisão.
   **Não publicar no PyPI nem commitar/push no R sem o "pode fazer" explícito do Eduardo**
   (token `daldeploy` em C:\Users\eduar\.pypirc).

GOTCHAS
- py3.14: `ruptures` só tem wheel pura na 1.0.6 (1.1.x exige compilador) → `pip install
  --prefer-binary`. Candidato a back-impl nativo (remove a dep do núcleo).
- `AutoencBaseEd` do pydaltoolbox é STUB identidade — autoencoder/waypoint precisam de torch real.
- Validar wheel de toda dep nova para o Python alvo ANTES de adotá-la.
