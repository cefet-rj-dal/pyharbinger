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
- **Backports R:** harbinger master largou `strucchange`, `ocp` **e `forecast`** inteiros; bump
  Boeing até **2.1.787**. Python pyharbinger largou `statsmodels` (dos change points; volta
  transitivo via pytspredit no `HanrArima`).

### Publicações (PyPI)
0.1.0 · 0.2.0 · 0.3.0 · 0.4.0 · 0.4.2 · 0.4.4 · 0.4.5 · 0.4.6 · 0.4.7 · 0.4.8. (Nem toda
milestone dev-only virou release; a última no ar é a **0.4.8**.)

### 0.4.7 — ensemble fuzzy + fix de backend do joinpoint (PyPI, 2026-11-25)
`HarEnsembleFuzzy` (consenso fuzzy + piso absoluto) + conversor de exemplos movido para
`notebooks/conversor.py` (alinha com pytspredit) — commits `f2d80d1`/`1c7cfc8`. **Publicada no
PyPI em 2026-11-25** (Eduardo autorizou). Antes do upload, o verde não estava limpo: 2 erros de
ruff (import/`__all__` do próprio commit da 0.4.7) + o teste de paridade R do **`HcpJoinpoint`**
falhando sob numpy 2.5.5. Diagnóstico: o DP do joinpoint é **tie-degenerate** em séries
(quase-)simétricas/perfeitamente lineares — vários conjuntos de breaks com o MESMO RSS total até
o roundoff (sinusoidal `[22,45,58,76,92]` e o R `[10,26,44,57,80]`, ambos RSS 5.312944); o `<`
estrito deixava o backend de álgebra (numpy `lstsq` vs R `lm`) decidir o empate, e o numpy novo
o virou. **Fix (commit `67cc193`):** desempate tolerante (`_JOINPOINT_TOL`, mantém o co-ótimo
mais cedo) + piso de RSS (`_JOINPOINT_RSS_FLOOR`) na seleção → DP determinístico e independente
de backend; `simple` passa a recuperar limpo o único cp 50. **Mesmo fix backportado para o R**
(`hcp_joinpoint.R`, harbinger master `723607f0`, versão **2.1.767→2.1.777**). **Paridade R↔Python
EXATA nas 7 séries.** Teste reescrito para paridade-R exata nas 5 séries rápidas (complex e
increasing_amplitude verificadas offline — lentas, O(k·n²)).

### 0.4.8 — `hanr_arima`++ (PyPI, 2026-11-25)
`HanrArima` (anomaly.py) = releitura do R `hanr_arima`: em vez do `forecast::auto.arima`, reusa o
**`pytspredit.fit_arima`** (substituto DAL do auto.arima — pedido do Eduardo, não reinventar) +
resíduos ARIMA in-sample → `har_distance`/`har_outliers`/check → warm-up `sw_size=max(p,d+1,q)`.
**++ de robustez:** o `.resid` do statsmodels tem transiente de inicialização difusa nos
primeiros pontos (o Kalman do forecast não tem) que dominava o threshold e escondia a anomalia →
zera-se o warm-up **antes** das estatísticas. **Gate Ação 1++ PASSOU:** SoftED-F1 soma/10 em
`examples_anomalies` `aicc` **9.03** > R `hanr_arima`-forecast **8.58** (default `auto="aicc"`; o
`cv`/ARIMA++ dá 8.44, é afinado p/ previsão). **Backport R** (harbinger master `fcb39c17`,
2.1.777→**2.1.787**): `hanr_arima` passa a usar `tspredit::fit_arima`; **`forecast` 100% removido**
do R (DESCRIPTION+NAMESPACE; era o último usuário). R-novo = 9.03 (= Python); detecções R↔Python
aproximadas (statsmodels vs `stats::arima`), qualidade de detecção igual. Decisão travada: previsão
(`TsArima`) usa `auto="cv"`; detecção por resíduo (`HanrArima`) usa `auto="aicc"` — ambos
parametrizáveis.

### 0.5.0 — `HanrFftSma` (port fiel, branch `milestone-fft-sma`, 2026-11-26)
`HanrFftSma` (anomaly.py) = releitura **fiel** do R `hanr_fft_sma` (nome enganoso: não usa FFT no
caminho de detecção — usa o periodograma só para achar a janela da média móvel). `_fft_sma_window`
é port fiel do `stats::spec.pgram` (detrend linear + taper cosseno dividido 10% + zero-pad para
`nextn`; frequência dominante → `round(n/k)` → janela clampada em `[3, n-1]`); só o argmax importa,
então as constantes de escala do `spec.pgram` são descartadas. O resíduo é a SMA trailing alinhada
(`filtered[w-1:] = serie[w-1:] - mas(serie, w)`, com `w-1` pontos de warm-up zerados antes das
estatísticas) — já com o fix de recycling do R master. **Paridade R↔Python EXATA nas 10 séries
`examples_anomalies`** (janela idêntica 27/64/171; conjuntos de idx idênticos, incl.
decreasing_amplitude com 13 pontos). `test_fft_sma.py` (idx exato por série + janela por série +
warm-up + guarda de `None`). R não expõe `res` na detecção (só idx/event/type), então o invariante
é o conjunto de índices. Versões 0.4.8→0.5.0. Abre o tema espectral/simbólico do Ciclo 1.

## M15 — simbólico SAX/XSAX (motif/discord), port FIEL (0.6.0, 2026-11-26, branch `milestone-symbolic-sax`)
Primeiro marco do tema simbólico do Ciclo 1. `src/pyharbinger/symbolic.py`: `TransSax`/`TransXsax`
(discretização SAX/xSAX — z-norm ddof=1 + quantis type-7 + `cut(include.lowest)` via
`searchsorted` + códigos base-26/36 com pad `str_pad`), `HmoSax`/`HmoXsax` (descoberta de motifs:
palavras repetidas ≥ `qtd`, de-overlap guloso por runs consecutivos + exclusão `±round(w/2)`) e
`HdisSax` (discords: palavras únicas longe de qualquer padrão repetido, exclusão `±w`). Reusa a
janela deslizante no espaço simbólico (equivalente a `tspredit::ts_data` sobre strings); entropia
de Shannon base-2 por palavra guia o desempate (`arrange(desc(total_count), desc(entropy))`,
ties→seq asc). Dataset `examples_motifs` (3 séries: `simple` + ECG `mitdb100`/`mitdb102` com coluna
extra `symbol`) extraído do `.RData` → CSV tidy; `datasets.examples_motifs()` + loader generalizado
p/ colunas extras. **Paridade R↔Python EXATA:** códigos SAX/XSAX idênticos em `simple` + conjuntos
de idx detectados idênticos nas 3 séries × 3 detectores (hmo_sax/hmo_xsax/hdis_sax), incl. as ECG
de 7202 pontos (motifs ~1200–1500 janelas; discords simple=9, mitdb100=∅, mitdb102=[3337]). R não
expõe `res` → invariante = conjunto de idx. `test_symbolic.py` (+15: transforms, entropia, paridade
exata por série, contrato da tabela, restore sobre NA). 194 testes, ruff limpo, notebook
`symbolic/01-motifs` compila. Versões 0.5.0→0.6.0. **🐞 Achado:** R `hdis_sax` rotula o resultado
como `type=="motif"` (lapso de copy-paste de `hmo_sax`); o port usa `type=="discord"` (correto) —
a confirmar com o Eduardo p/ corrigir no backport R (paridade é sobre idx, não sobre `type`).

## M16 — HarEnsemble (voto majoritário) + fix do HanrHistogram (0.7.0, 2026-11-26, branch `milestone-ensemble`)
Segundo passo do Ciclo-1 pós-simbólico. `src/pyharbinger/ensemble.py`: **`HarEnsemble`** — port
fiel do R `har_ensemble` (voto majoritário: `res = soma dos votos`, evento onde `res > m/2`; tipo =
`"changepoint"` se algum membro votar change point, senão `"anomaly"`). Reusa `Harbinger` base; `fit`
treina cada membro na série sem NAs; `detect` empilha os votos. **🐞 Bug latente encontrado +
corrigido:** o ensemble revelou que `HanrHistogram` divergia do R na série `tt` (R:
`0,11,23,37,49,89,100`; Py: `11,37,89,100`). Causa: `np.histogram` conta com bins **fechados à
esquerda** `[a,b)`, mas `graphics::hist` do R usa **fechados à direita** (`right=TRUE`) `(a,b]` — o
valor `1.0` (e o bin `(1.0,1.5]`) caía em bin diferente, deslocando a densidade por cima/baixo do
limiar `0.05`. R é canônico aqui → **fix só no Python** (contagem right-closed via
`searchsorted(side="left")-1` + `bincount`); a busca do bin na detecção (findInterval left-closed via
`searchsorted(side="right")`) já estava certa. **`HanrHistogram` agora EXATO nas 10 séries**
`examples_anomalies` (antes só validado em `simple`). `HarEnsemble` **EXATO vs R** em
simple/contextual/tt/volatile (membros fft+fbiad+hist). 202 testes (+8: 4 ensemble-paridade
parametrizados + 3 unit + 1 regressão histograma right-closed), ruff limpo, notebook
`ensemble/01-ensemble` compila. Versões 0.6.0→0.7.0. **Falta do item 2 do Ciclo 1:** plots
(`har_plot*`). **Também nesta sessão:** R `hdis_sax` corrigido p/ `type=="discord"` (commit local
`f2a9193c`, versão Boeing **2.2.707** — novo minor reinicia a sequência; push ao master bloqueado
pelo harness, pendente do Eduardo).

## M17 — plots `har_plot*` (matplotlib), fecha o item-2 do Ciclo 1 (0.8.0, 2026-11-26, branch `milestone-plots`)
`src/pyharbinger/plots.py` — port adaptado dos helpers R (ggplot2/patchwork → **matplotlib**, agora
dep do núcleo): **`har_plot`** (série + pontos coloridos por TP verde / FN azul / FP vermelho /
segmento de motif roxo; change points como linhas verticais tracejadas; `yline` pontilhadas;
`motifs_seqs` expandindo `seq`/`seqlen`), **`har_ensemble_plot`** (2 painéis: score de consenso +
limiar, e eventos detectados) e **`har_ensemble_plot_models`** (série original + uma faixa por membro,
via `model_events`). Sem gate numérico (helper visual) — testes são smoke (estrutura da figura:
painéis, nº de linhas de CP, linhas yline, nº de faixas por membro, cores). `matplotlib>=3.5` entra
em `dependencies` do `pyproject`. 210 testes (+8 test_plots.py, backend Agg), ruff limpo, notebook
`plots/01-plots` compila, build+twine PASS. Versões 0.7.0→0.8.0. **Fecha o item-2 do Ciclo 1**
(ensemble + visualização). **Falta do Ciclo 1:** só o item-3 (streaming/online + datasets de
benchmark/`loadfulldata`).

---

## Estado atual (o resto está em `planejamento.md`)
- **pyharbinger:** PyPI latest **0.7.0** (`HarEnsemble` + fix HanrHistogram, publicada); **0.8.0**
  (plots `har_plot*`) em branch `milestone-plots`, aguardando revisão. Depende de pydaltoolbox +
  pytspredit + **matplotlib** (PyPI).
- **Cobertura:** núcleo + avaliação + anomalias base + FFT-híbridos + change points (ruptures
  + nativos + os 3 ++ strucchange/forecast/ocp) + clustering + multivariado (HancMl/HmuPca) +
  ensemble (`HarEnsemble` majoritário + `HarEnsembleFuzzy`) + `HanrArima` (ARIMA++) + `HanrFftSma`
  + **simbólico (`TransSax`/`TransXsax`/`HmoSax`/`HmoXsax`/`HdisSax`)** + **plots (`har_plot`/
  `har_ensemble_plot`/`har_ensemble_plot_models`)**. **Falta** (Ciclo 1): só streaming/online +
  datasets de benchmark/`loadfulldata`. **Ciclo 2** (extras): motif-MP (`stumpy`), GARCH (`arch`),
  EMD (`EMD-signal`), wavelet (`PyWavelets`), DTW, deep/torch.
- **R harbinger:** master em **2.1.787**; largou `strucchange`, `ocp` **e `forecast`**; ainda
  importa `changepoint` e as libs do Ciclo 2 (`hht`, `wavelets`, `rugarch`, `tsmp`, `dtwclust`).
  CRAN/commits do R = Eduardo.
- `main` é a única branch de código (os `milestone-*` já mesclados foram apagados).
