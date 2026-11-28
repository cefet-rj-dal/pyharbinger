Retomada do pyharbinger (c:\Git\dal\pyharbinger). Rode a partir desta pasta. Leia primeiro a
memória do projeto (pyharbinger-port, back-implementation-initiative, boeing-versioning,
verify-dependency-backends — se existirem nesta máquina) e, no develop/: planejamento.md (só o
que falta), historico.md (a história), mapeamento.md (R↔Python), parity.md (metodologia + matriz).

ATUALIZAÇÃO PRIORITÁRIA (2026-11-28): o estado detalhado abaixo é histórico. O `main` atual
está em 0.12.0. O item Deep foi concluído externamente no pydaltoolboxdp 0.6.0 local:
`HanAutoencoder` e `HcpWaypoint` especializam `pyharbinger.Harbinger`, sem adicionar torch ou
stubs a este pacote. A paridade R↔Python foi exata a `1e-7`; os 12 exemplos concretos foram
transferidos de harbinger para daltoolboxdp. Próximo trabalho próprio continua sendo EMD,
conforme `planejamento.md`.

PRINCÍPIOS (confirme em poucas linhas e siga)
- Python = releitura canônica / R = backport; **incrementar métodos ++ + reduzir dependência**
  em todo o stack DAL. **Não é regra cega** — caso a caso com o Eduardo; libs canônicas
  (sklearn/scipy/ruptures/stumpy/arch/...) seguem embrulhadas.
- **Gate Ação 1++** = detectar a verdade tão bem ou melhor via **SoftED-F1 vs a coluna `event`**
  (ground truth), comparado ao F1 do pacote de referência — não paridade com ele. Instrumento:
  develop/benchmark_backimpl.py. Port fiel continua no gate de paridade com o R.
- Doc: Teoria + References (paper) + Porting em toda classe; dep-removal vai no commit, não na doc.
- Numeração Boeing só no R após 1.0.0.

ESTADO (2026-11-25, fim da sessão)
- **PyPI latest = 0.4.8.** Tudo mesclado no `main` (única branch de código; milestones apagados).
  0.4.7 = ensemble fuzzy (`HarEnsembleFuzzy`) + fix de backend do `HcpJoinpoint`; 0.4.8 =
  `hanr_arima`++.
- **Pronto:** núcleo + avaliação (`HarEval`/`HarEvalSoft`) + anomalias base (`HanrMl`/`HanrFft`/
  `HanrHistogram`/`HanrFbiad`) + **`HanrArima`** (ARIMA++) + 4 FFT-híbridos + change points
  (ruptures `HcpAmoc/Binseg/Pelt` + nativos `HcpCfLr/Scp/Joinpoint/Kswin/PageHinkley` + os 3
  **++**: `HcpChow/Gft` strucchange++, `HcpCfArima/CfEts` forecast++, `HcpBocpd` ocp++) +
  clustering (`HanctKmeans`) + multivariado (`HancMl`/`HmuPca`) + `HarEnsembleFuzzy`. Ver
  mapeamento.md para o que falta.
- **`HcpJoinpoint`:** DP agora independente de backend (desempate tolerante + piso de RSS) →
  paridade R↔Python **exata nas 7 séries** (antes quebrava sob numpy 2.5.5).
- **`HanrArima` (0.4.8):** reusa `pytspredit.fit_arima` (substituto DAL do auto.arima; default
  `auto="aicc"`, `cv`=ARIMA++ é pior p/ detecção por resíduo) + resíduos in-sample, zera warm-up
  ANTES das estatísticas (mata o transiente de init do statsmodels). Gate 9.03 > R-forecast 8.58.
  **Decisão travada:** previsão (`TsArima`) usa `cv`; detecção (`HanrArima`) usa `aicc`.
- **Deps:** núcleo = pydaltoolbox/pytspredit/numpy/pandas/sklearn/scipy/ruptures (+matplotlib p/
  plots); `statsmodels` volta só transitivo via pytspredit (no `HanrArima`). R harbinger master
  **2.1.787**: largou `strucchange`, `ocp` **e `forecast`**; ainda importa `changepoint` e as libs
  do Ciclo 2 (`hht`/`wavelets`/`rugarch`/`tsmp`/`dtwclust`).

PRÓXIMO (decidir comigo qual — ver planejamento.md)
- Ciclo 1, na ordem: **`HanrFftSma`** (filtro espectral + média móvel; bug de recycling já
  corrigido no R master — portar o R `hanr_fft_sma.R`) → simbólico (SAX/motif/discord: `TransSax`/
  `TransXsax`, `HmoSax`/`HmoXsax`/`HdisSax`) → ensemble simples (`HarEnsemble`) + visualização
  (`har_plot`*) → streaming/online + datasets de benchmark (⚠️ replicar o `loadfulldata` do
  pytspredit: miniaturas + extra `[bench]`).
- Ciclo 2 (extras, a refinar): motif-MP/GARCH/EMD/wavelet/DTW/deep.

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
