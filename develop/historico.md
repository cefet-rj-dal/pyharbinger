# Histórico — pyharbinger (port Python do harbinger)

Relato cronológico do trabalho concluído. O que falta está em `planejamento.md`; a
correspondência de classes em `mapeamento.md`; a metodologia e a matriz de paridade em
`parity.md`. Ao concluir uma atividade, registramos aqui.

Projeto: port Python do pacote R **harbinger** (CEFET/RJ, lab DAL) — framework unificado de
**detecção de eventos em séries temporais** (anomalias, change points, motifs/discords),
preservando a **Experiment Line** (`fit`/`detect`/`evaluate`). É o **último da família DAL** e
depende de **`pydaltoolbox` + `pytspredit`** (ambos no PyPI). Princípios: embrulhar libs
clássicas (numpy/pandas/scikit-learn/scipy/ruptures) com adapters finos e fiéis ao R;
back-implementar nativamente o que é nicho/atípico, **como versão ++** (avanço real de método,
com atribuição ao original); docstrings com Teoria + `References:` + `Porting:`; bug em qualquer
lado corrige-se nos dois repositórios; comparar R vs Python a cada passo.

---

## Bloco A / v1 — pipeline de detecção enxuto (0.1.0, 2026-11-18, PyPI)
Consolidado em 4 milestones (M0→M3), cada uma em branch própria, paridade R vs Python antes
de pausar. **Escopo decidido (Eduardo): trilha enxuta, sem motif; pausar a cada milestone.**
- **M0 — Scaffolding:** `pyproject` (deps pydaltoolbox+pytspredit+numpy/pandas/sklearn/scipy/
  ruptures), `src/pyharbinger/`, pipeline de docs executáveis. Datasets `examples_anomalies`
  (10 séries) + `examples_changepoints` (7) extraídos do R → CSV tidy, paridade exata. **Nota
  py3.14:** `ruptures` só tem wheel pura na 1.0.6 (1.1.x exige compilador C) → pin `>=1.0`,
  instalar com `--prefer-binary`.
- **M1 — Núcleo + Avaliação:** `utils.py` (desvios L1/L2/Huber; filtros none/boxplot/gaussian/
  mad/grubbs/ratio; seleção firstgroup/highgroup/referencedistribution), `harbinger.py`
  (`Harbinger(DalBase)` + `fit`/`detect`/`evaluate` + `_store_refs`/`_restore_refs`),
  `evaluation.py` (`HarEval` + `HarEvalSoft`, Hungarian via scipy). **Paridade EXATA** (tudo
  ≤1e-5). Chaves de métrica unificadas em TP/FP/FN/TN (hard e soft). `idx` 0-based.
- **M2 — Anomalias:** `HanrMl` (ponte pytspredit), `HanrFft`, `HanrHistogram` (`r_pretty` =
  `pretty()`/`hist()` do R), `HanrFbiad`. **Paridade EXATA** em `examples_anomalies$simple`.
  Tolerância: `HanrMl` com ELM `nhid<input_size`+purelin depende do RNG (como no pytspredit).
- **M3 — Change points + clustering + closeout:** `HcpAmoc/Binseg/Pelt` (ruptures), `HcpKswin`,
  `HcpPageHinkley`, `HanctKmeans`. **Paridade:** PageHinkley EXATO; HanctKmeans bate;
  AMOC/BinSeg ~off-by-1 (ruptures ≠ MBIC, aproximado). `HancMl` adiado p/ 0.2.0 (multivariado).
  Build + twine check + install em venv limpo OK. **`pyharbinger 0.1.0` PUBLICADO no PyPI** —
  família DAL completa em Python (pydaltoolbox + pytspredit + pyharbinger).

---

## Roadmap 0.2.0 → 1.0.0 — DOIS CICLOS (refinado 2026-11-19, Eduardo)
Estrutura em duas rodadas sobre os mesmos temas: **Ciclo 1** varre tudo só com classes de
baixa dep (✅ reuso / 🔨 nativo); **Ciclo 2** volta aos temas com as classes de dep pesada
(uma família de *extra* por release). Decisões: (A) empacotamento misto (`matplotlib` no
núcleo; pesadas como extras com import preguiçoso + erro acionável, padrão `Suggests` do R);
(B) granularidade livre; (C) datasets miniatura + `loadfulldata` = replicar o padrão do
pytspredit (benchmark de detecção: Yahoo A1–A4, gecco, MIT-BIH, NAB, oil_3w, UCR).

### Ciclo 1 — baixa dependência
- **0.2.0 — Aprendizado & multivariado** (2026-11-19): caminho **multivariado** na base
  (`_store_refs` aceita DataFrame; linha presente só se todas as colunas não-NA) +
  `HancMl` (classificador pydaltoolbox) + `HmuPca` (PCA sklearn). **Paridade:** `HancMl`
  com `ClaKnn(k=3)` exato; `HmuPca` exato **após corrigir bug no R** (ver abaixo).
  **Correção de escopo:** `AutoencBaseEd` do pydaltoolbox é **stub identidade** →
  `HanAutoencoder`/`HcpWaypoint` precisam de torch → movidos p/ o Ciclo 2 (`[deep]`).
  **Lição:** não assumir que uma classe-base do pydaltoolbox é implementação completa —
  abrir o arquivo e verificar o backend real antes de classificar a dep.
- **🐞 Bug corrigido no R `harbinger` (`R/hmu_pca.R`)** (2026-11-19, OK do Eduardo): o
  `hmu_pca` mantinha TODAS as componentes → reconstrução exata → resíduo = ruído de float
  (~1e-31), detector degenerado. Fix: params `num_components`/`variance`, reconstrução pelas
  componentes líderes (default 90% da variância, teto `p-1`). Python alinhado → paridade exata.
- **0.3.0 — Espectral (anomalia)** (2026-11-20): `mas` + 4 híbridos FFT-cut
  (`HanrFftAmoc/Binseg/AmocCusum/BinsegCusum`; cutoff = change-point no espectro via ruptures).
  **Paridade EXATA** em `examples_anomalies$simple`. **`HanrFftSma` ADIADO** (bug de recycling
  no R: `serie - ts_sma` com lengths incompatíveis; corrigido depois no R master `cb8c89d9`).
- **0.4.0 — Change point estendido (nativos exatos)** (2026-11-20): `HcpCfLr`, `HcpScp`,
  `HcpJoinpoint` (lm/DP/correlação puros). **Paridade:** CfLr e Scp EXATOS nas 7 séries;
  Joinpoint exato em bem-condicionadas (RSS→0 roundoff na `simple`, mas cp real sempre pego).
  **Perf:** joinpoint é O(n³·k) — testes usam só séries curtas.

### Iniciativa back-implementation + Pass ++ (Eduardo, 2026-11-20/21)
Além de "não reimplementar", trilha paralela para **trocar deps de terceiros por implementação
nativa** onde faz sentido (nicho/atrito/auto-seleção que quebra paridade), tornando o stack DAL
autocontido. **Não é regra cega** — caso a caso, libs canônicas seguem embrulhadas (ex.: MLP do
sklearn NÃO vira back-impl). **Decisão 2026-11-20:** uma versão por pacote de origem
(0.4.1 strucchange, 0.4.2 forecast, 0.4.3 ocp). **Decisão 2026-11-21 (++):** transformar os
ports fiéis em **versões ++** (avanço real de método, para contribuir com a ciência). Isso
**muda o gate (Ação 1++):** não é mais "bate com o pacote", é **"detecta a verdade tão bem ou
melhor"** (SoftED-F1 vs a coluna `event`, comparado ao F1 do pacote de referência); paridade
com o R vira nota `adapted (++)`. Instrumento: `develop/benchmark_backimpl.py`.

- **0.4.1–0.4.3 — ports fiéis nativos** (2026-11-21): `HcpChow`/`HcpGft` (strucchange
  reverse-engenheirado), `HcpCfArima`/`HcpCfEts` (auto.arima/ets com seleção própria via
  statsmodels), `HcpBocpd` (BOCPD Adams&MacKay, puro numpy/scipy — `ocp` só em *Suggests* no R).
  **Gate Ação 1:** BOCPD **EXATO** 7/7 (instalei o `ocp` do arquivo do CRAN p/ comparar);
  Chow/GFT exatos em 5–6/7; cf_ets exato, cf_arima com `d` exato e `(p,q)` divergindo (tol).
- **0.4.4 — strucchange++** (2026-11-21): custo **mean+variance** (não só média) + penalidade
  **LWZ** (Liu-Wu-Zidek) no lugar do BIC. **Gate:** Chow 3.73 / GFT 3.17 (soma/7) > R (3.31 / 2.20).
- **0.4.5 — forecast++** (2026-11-21): **ChangeFinder robusto** nativo = SDAR (Takeuchi-Yamanishi)
  ∪ quebra mean+variance; **puro numpy, derruba `statsmodels`**. **Gate:** cf_arima 3.84 /
  cf_ets 3.64 > R (2.92 / 2.66). Descoberta: SDAR puro é cego a rampas suaves → a união resolve.
- **0.4.6 — ocp++** (2026-11-21): BOCPD com **UPM autorregressivo Bayesiano** (sensível a
  mudança de dinâmica/slope) + recursão log-space com poda ε (**streaming** O(n·R̄)) + extração
  por surpresa preditiva (z-robusto causal). **Gate:** bocpd 3.59 vs R `ocp` 1.12 — vence 6/7
  (~3,2×). União com mean+variance piora (dilui `simple`) → ficou BOCPD puro.
- **Backports R:** harbinger master largou `strucchange` e `ocp` inteiros; bump Boeing até
  **2.1.757**. Python pyharbinger largou `statsmodels`.

### Publicações (PyPI)
0.1.0 · 0.2.0 · 0.3.0 · 0.4.0 · 0.4.2 · 0.4.4 · 0.4.5 · 0.4.6. (Nem toda milestone dev-only
virou release; a última no ar é a **0.4.6**.)

### 0.4.7 — ensemble fuzzy (na main, NÃO publicado)
`HarEnsembleFuzzy` (consenso fuzzy + piso absoluto) + conversor de exemplos movido para
`notebooks/conversor.py` (alinha com pytspredit). Está em `main` (commits `f2d80d1`,
`1c7cfc8`) mas **ainda não foi ao PyPI** (latest lá = 0.4.6). *(Obs.: o `nextprompt` antigo
propunha 0.4.7 = `hanr_arima`++; o que de fato saiu na 0.4.7 foi o ensemble fuzzy — o
`hanr_arima`++ segue pendente.)*

---

## Estado atual (o resto está em `planejamento.md`)
- **pyharbinger:** PyPI latest **0.4.6**; `main` tem a **0.4.7** (ensemble fuzzy) **não
  publicada**. Depende de pydaltoolbox + pytspredit (PyPI).
- **Cobertura:** núcleo + avaliação + anomalias base + FFT-híbridos + change points (ruptures
  + nativos + os 3 ++ strucchange/forecast/ocp) + clustering + multivariado (HancMl/HmuPca) +
  ensemble fuzzy. **Falta** (Ciclo 1): `hanr_arima`++, `HanrFftSma`, simbólico (SAX/motif/
  discord), ensemble simples + plots, streaming/online + datasets de benchmark/`loadfulldata`.
  **Ciclo 2** (extras): motif-MP (`stumpy`), GARCH (`arch`), EMD (`EMD-signal`), wavelet
  (`PyWavelets`), DTW, deep/torch.
- **R harbinger:** master em **2.1.767**; largou `strucchange` e `ocp`; ainda importa
  `forecast` (último usuário = `hanr_arima`), `changepoint`, e as libs do Ciclo 2 (`hht`,
  `wavelets`, `rugarch`, `tsmp`, `dtwclust`). CRAN/commits do R = Eduardo.
- `main` é a única branch de código (os `milestone-*` já mesclados foram apagados).
