# Planejamento — `pyharbinger` (port do R `harbinger`)

> Documento vivo. Atualizado a cada milestone com decisões e status.
> Segue o método validado em `pydaltoolbox` e `pytspredit` (ambos no PyPI).
> Kickoff: [develop/PROMPT.md](PROMPT.md).

## Missão

Portar o pacote R **`harbinger`** (detecção de eventos em séries temporais —
anomalias, change points, motifs/discords) para **`pyharbinger`**. É o último da
família DAL e depende de **`pydaltoolbox` + `pytspredit`**.

Princípio central (herdado): **não reimplementar algoritmos** — envolver libs
clássicas (`numpy`, `pandas`, `scikit-learn`, `scipy`, `ruptures`, `stumpy`, ...)
com a interface uniforme da Linha de Experimento. A contribuição é o framework
unificado de detecção + avaliação de eventos.

## Contrato (do R `harbinger.R`)

- `Harbinger` (base) estende o `DalBase` do pydaltoolbox.
- `detect(serie)` → `pandas.DataFrame` com colunas `idx` (int), `event` (bool),
  `type` (str ∈ {`"anomaly"`, `"changepoint"`, `"discord"`, `"motif"`, `""`}).
  Detectores de sequência adicionam colunas `seq`/`seqlen`.
- `fit(serie)` encadeável (retorna self); para muitos detectores é no-op.
- `evaluate(detection, event)` delega para `HarEval` por padrão.
- Helpers internos: `har_store_refs` (guarda a série, posições não-NA, placeholders)
  e `har_restore_refs` (restaura detecções ao índice original e monta o DataFrame).

## Convenções (idênticas a pydaltoolbox/pytspredit)

- Classes **PascalCase** (`Harbinger`, `HanrMl`, `HcpPelt`, `HarEval`,
  `HarEvalSoft`). Funções utilitárias snake_case quando forem funções.
- Dados: pandas/numpy. Série = `pd.Series`/vetor; detecção = DataFrame.
- Adapters finos e fiéis ao backend R; **comparar R vs Python** a cada passo.
- Layout `src/pyharbinger/`, build hatchling, deps mínimas.
- 3 camadas: `pytest` (unit) + integração + **documentação executável**
  (notebooks jupytext → `examples/` via `tools/build_examples.py`; um exemplo por
  classe, **agrupados por família** num notebook).
- `develop/` excluída do sdist. Sempre verde (`ruff` + `pytest` + build de exemplos).
- Git: branch por milestone (cada um saindo do anterior) → push → Eduardo revisa no
  GitHub → ff-merge na main. `gh` não instalado.
- Commits terminam com `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Deploy: PyPI via `twine` (token `daldeploy` em `~/.pypirc`); beta até 1.0.0;
  Eduardo autoriza/roda o upload.

## Reuso do ecossistema

```python
from pydaltoolbox import DalBase, ClusterKmeans, ...        # base, clustering, classificação
from pytspredit import TsData, ts_projection, TsReg, ...    # janelas e regressores
```

- `ts_data(serie, sw)` → DataFrame de janelas (colunas `t{sw-1}..t0`, `t0` = alvo).
- `ts_projection(ts)` → `.input` (X) e `.output` (y).
- `HanrMl` é a **ponte** que prova pydaltoolbox→pytspredit→pyharbinger.

## Escopo da v1 — PROPOSTA (aguardando confirmação do Eduardo)

**Trilha enxuta** (wrappers finos de libs já disponíveis):

| Família | Entra na v1 | Adiado (0.2.0 / Bloco B) |
|---|---|---|
| Núcleo | `Harbinger` + `detect` + utils (desvios, filtros, seleção) | — |
| Avaliação | `HarEval`, `HarEvalSoft` (Hungarian scipy) | — |
| Anomalia | `HanrMl`, `HanrFft`, `HanrHist`, `HanrFbiad` | `arima`, `garch`, `emd`, `remd`, `wavelet`, `rtad` |
| Change point | `HcpAmoc`, `HcpBinseg`, `HcpPelt` (ruptures), `HcpKswin`, `HcpPageHinkley` | `bocpd`, `chow`, `garch`, `gft`, `joinpoint`, `cf_*`, `scp`, `waypoint` |
| Clustering | `HanctKmeans`, `HancMl` | `hanct_dtw` (DTW) |
| Motif/discord | — (adiado) | `hdis_mp`, `hmo_mp` (stumpy), `*_sax`, `*_xsax`, `hmu_pca` |
| Outros | — | ensembles, streaming/online, plots, `mas`, `united` |

**Deps v1:** `pydaltoolbox`, `pytspredit`, `numpy`, `pandas`, `scikit-learn`,
`scipy`, `ruptures`. (motif/`stumpy` adiado para 0.2.0.)

### Tolerâncias de paridade esperadas

- Detectores puros (fft/hist/fbiad/kswin/page_hinkley/eval) → determinísticos, ~1e-9.
- `HanrMl` → herda a tolerância do regressor pytspredit (knn/elm exatos; rf/mlp dentro do RNG).
- `HanctKmeans` → init/RNG do k-means difere de `stats::kmeans` → tolerância documentada.
- **Change points (ruptures)**: o R usa `changepoint::cpt.meanvar` (custo Normal
  mean+variance, penalidade MBIC); o `ruptures` não reproduz MBIC exatamente →
  paridade **aproximada** (comparar posições dos breakpoints, não igualdade exata).

## Milestones — PROPOSTA

- **M0 — Scaffolding.** `pyproject` (deps pydaltoolbox+pytspredit), `src/pyharbinger/`,
  `tools/build_examples.py`, `develop/`. Extrair datasets do R (`examples_anomalies`,
  `examples_changepoints`, `examples_motifs`, `examples_harbinger`) via `Rscript` →
  CSV em `src/pyharbinger/data/`, verificando numericamente.
- **M1 — Núcleo + Avaliação.** `Harbinger` base + `detect` + utils (desvios L1/L2/Huber,
  filtros none/boxplot/gaussian/mad/grubbs/ratio, seleção firstgroup/highgroup/
  referencedistribution) + `HarEval` + `HarEvalSoft` (Hungarian via scipy). Paridade R.
- **M2 — Anomalias.** Slice vertical `HanrMl` (ponte pytspredit) ponta-a-ponta +
  `HanrFft`, `HanrHist`, `HanrFbiad`. Paridade R.
- **M3 — Change points + clustering + closeout.** `HcpAmoc/Binseg/Pelt` (ruptures),
  `HcpKswin`, `HcpPageHinkley`, `HanctKmeans`, `HancMl` (+ motif se escolhido) →
  review de API, README, build + twine check + install em venv limpo → publicar 0.1.0.

*(4 milestones consolidadas; cada uma = implementar + testes + notebook + build_examples
+ ruff/pytest verdes + comparação R vs Python + commit/push + revisão do Eduardo.)*

## Decisões travadas (2026-11-18, Eduardo)

- **Escopo da v1 = Trilha enxuta (sem motif).** Motif/discord (`hdis_mp`/`hmo_mp`
  via stumpy) fica para 0.2.0, junto com arima/garch/emd/wavelet/DTW/SAX/ensembles/
  streaming/plots.
- **Cadência = pausar a cada milestone.** Branch por milestone (saindo do anterior),
  push, exemplos commitados + links do GitHub, Eduardo revisa, ff-merge na main.

## Status

- **2026-11-18** — Análise do R concluída; escopo e cadência confirmados com o Eduardo.
- **2026-11-18 — M0 DONE** (branch `milestone-0-scaffolding`). `pyproject` (deps
  pydaltoolbox+pytspredit+numpy/pandas/sklearn/scipy/ruptures), `src/pyharbinger/`,
  `tools/build_examples.py`, `develop/`. Datasets `examples_anomalies` (10 séries) e
  `examples_changepoints` (7 séries) extraídos do R → CSVs tidy em
  `src/pyharbinger/data/`, verificados numericamente (paridade exata com o R).
  `datasets.py` + `__init__`. 5 testes verdes, ruff limpo, 1 notebook de exemplo
  construído. **Nota de ambiente:** Python 3.14 → `ruptures` só tem wheel pura em
  1.0.6 (as 1.1.x exigem compilação C); pin relaxado para `ruptures>=1.0`.
  Datasets de motif/harbinger não embarcados na v1 (fora do escopo enxuto).
- **2026-11-18 — M1 DONE** (branch `milestone-1-core-eval`, sai da M0). `utils.py`
  (desvios L1/L2/Huber; filtros none/boxplot/gaussian/mad/grubbs/ratio → `FilterResult`;
  seleção firstgroup/highgroup/referencedistribution → `SelectionResult`), `harbinger.py`
  (`Harbinger(DalBase)` + `fit`/`detect`/`evaluate` + `_store_refs`/`_restore_refs`,
  hooks default = L2/gaussian/firstgroup), `evaluation.py` (`HarEval` + `HarEvalSoft`
  com Hungarian via `scipy.optimize.linear_sum_assignment`). **Paridade R EXATA**
  (devios, filtros c/ índices+thresholds, seleção, hard eval, soft eval — todos ≤1e-5;
  determinísticos). 23 testes verdes, ruff limpo, 2 notebooks. **Nota:** unifiquei as
  chaves de métrica em `TP/FP/FN/TN` para hard e soft (no R o soft usa `TPs/FPs/...`);
  valores idênticos. Confusão = DataFrame 2x2 (detection×event). `idx` 0-based.
  `fuzzify` (só p/ ensemble fuzzy) não portado — fora do escopo v1.
- **2026-11-18 — M2 DONE** (branch `milestone-2-anomalies`, sai da M1). `anomaly.py`:
  `HanrMl` (ponte: embrulha regressor do pytspredit, prevê 1-passo sobre janelas,
  marca resíduos grandes), `HanrFft` (filtro passa-alta via FFT), `HanrHistogram`
  (densidade de bins; `r_pretty` reproduz o `pretty()`/`hist()` do R), `HanrFbiad`
  (inercial forward+backward). **Paridade R:** FFT, Histogram, FBIAD e HanrMl(kNN,
  determinístico) batem **EXATAMENTE** com o R (eventos + resíduos ≤1e-4) sobre
  `examples_anomalies$simple` (anomalia em idx 50 1-based). **Tolerância documentada:**
  HanrMl com ELM `nhid<input_size` + `purelin` é linear de posto reduzido → depende do
  RNG dos pesos ocultos → difere R vs Python (igual ao pytspredit; use `nhid>=input_size`
  ou regressor determinístico p/ paridade exata). 30 testes verdes, ruff limpo, 3
  notebooks. **Nenhum bug encontrado no `harbinger` R.**
- **2026-11-18 — M3 DONE + closeout** (branch `milestone-3-changepoints-closeout`,
  sai da M2). `changepoint.py`: `HcpAmoc`/`HcpBinseg`/`HcpPelt` (ruptures, custo
  "normal"), `HcpKswin` (online, KS com subamostra RNG), `HcpPageHinkley` (online,
  determinístico). `clustering.py`: `HanctKmeans` (sklearn; anomalia seq=1 / discord
  seq>1). **Paridade R:** `HcpPageHinkley` **EXATO** (29/36/110/vazio nas 4 séries);
  `HanctKmeans` bate (event=50, centers=3); AMOC/BinSeg batem em 3/4 e ~off-by-1
  (aproximado, esperado). **Tolerâncias:** PELT usa penalidade `3·log(n)` (MBIC não
  reproduzível no ruptures); KSWIN e KMeans dependem de RNG (`random_state`). **Escopo:
  `HancMl` ADIADO para 0.2.0** — exige caminho multivariado (DataFrame) + manejo de
  rótulos/probabilidades do classificador pydaltoolbox; destoa dos detectores
  univariados. Closeout: README v1 (27 símbolos públicos), `develop/deploy.md`,
  build + `twine check` PASSOU, sdist exclui dev + embarca CSVs, **install em venv
  limpo PASSOU** (numpy/pandas/ruptures 1.0.6/pydaltoolbox/pytspredit do PyPI; smoke
  end-to-end dos 8 detectores OK). 43 testes verdes, ruff limpo, 5 notebooks.
  **Nota py3.14:** `ruptures` 1.1.x não tem wheel cp314 (sdist precisa compilador) →
  usar `pip install --prefer-binary` (pega a 1.0.6 pura); py3.9–3.13 instalam direto.

## Pronto para publicar

`pyharbinger 0.1.0` está validado e pronto para upload ao PyPI. **Aguardando
autorização explícita do Eduardo** para rodar o `twine upload` (publicar é
irreversível; versões imutáveis). Após publicar: ff-merge dos branches na main +
badge/status no README.

> **2026-11-18 — PUBLICADO.** `pyharbinger 0.1.0` no PyPI; branches ff-merge na main;
> badges no README. Família DAL completa em Python (pydaltoolbox + pytspredit + pyharbinger).

---

# Roadmap 0.2.0 → 1.0.0 (refinado 2026-11-19)

> Refinamento pedido pelo Eduardo: considerar **todas** as classes do R, introduzir
> os componentes **gradualmente** e decidir caso a caso **dependência pesada vs.
> implementação nativa**. Esta seção substitui o antigo "Bloco B" genérico.

## Inventário completo do R `harbinger` (o que falta portar)

Mapa R→Python de cada classe ainda não portada, com a dependência que ela exige.
Legenda da coluna **Dep**: ✅ = sem dep nova (reusa o que já temos); 🆕 = dep nova
(vira *extra* opcional); 🔨 = implementação nativa (não há lib boa/leve o bastante).

### Anomalia (`hanr_*`, `han_*`)

| R | Python (proposto) | Backend | Dep |
|---|---|---|---|
| `hanr_fft_amoc` | `HanrFftAmoc` | FFT + ruptures (AMOC) | ✅ |
| `hanr_fft_binseg` | `HanrFftBinseg` | FFT + ruptures (BinSeg) | ✅ |
| `hanr_fft_sma` | `HanrFftSma` | FFT + média móvel (`mas`) | ✅ |
| `hanr_fft_amoc_cusum` | `HanrFftAmocCusum` | FFT + AMOC + CUSUM | ✅ |
| `hanr_fft_binseg_cusum` | `HanrFftBinsegCusum` | FFT + BinSeg + CUSUM | ✅ |
| `han_autoencoder` | `HanAutoencoder` | `pydaltoolbox.AutoencBaseEd` (torch já transitivo) | ✅ |
| `hanr_garch` | `HanrGarch` | `arch` (sGARCH+ARMA) | 🆕 `[garch]` |
| `hanr_emd` | `HanrEmd` | `EMD-signal` (CEEMD) | 🆕 `[emd]` |
| `hanr_remd` | `HanrRemd` | `EMD-signal` + statsmodels ARIMA | 🆕 `[emd]` |
| `hanr_wavelet` | `HanrWavelet` | `PyWavelets` (SWT≈MODWT) | 🆕 `[wavelet]` |
| `hanr_rtad` | `HanrRtad` | `EMD-signal` + rolling (pandas) | 🆕 `[emd]` |

### Change point (`hcp_*`)

| R | Python (proposto) | Backend | Dep |
|---|---|---|---|
| `hcp_cf_lr` | `HcpCfLr` | ChangeFinder c/ regressão linear | 🔨 nativo |
| `hcp_cf_arima` | `HcpCfArima` | ARIMA (reusa `pytspredit`/statsmodels) | ✅ |
| `hcp_cf_ets` | `HcpCfEts` | ETS (`statsmodels` ExponentialSmoothing) | ✅ |
| `hcp_chow` | `HcpChow` | teste F de Chow (`statsmodels` OLS) | 🔨 nativo |
| `hcp_gft` | `HcpGft` | fluctuation/F-stats (`statsmodels`) | 🔨 nativo |
| `hcp_joinpoint` | `HcpJoinpoint` | joinpoint (lm/cor) | 🔨 nativo |
| `hcp_scp` | `HcpScp` | OLS segmentado (`lm`) | 🔨 nativo |
| `hcp_bocpd` | `HcpBocpd` | BOCPD Adams&MacKay (R usa `ocp`, só *Suggests*) | 🔨 nativo |
| `hcp_waypoint` | `HcpWaypoint` | `pydaltoolbox.AutoencBaseEd` | ✅ |
| `hcp_garch` | `HcpGarch` | `arch` + OLS 1º estágio | 🆕 `[garch]` |

### Clustering / classificação / multivariado

| R | Python (proposto) | Backend | Dep |
|---|---|---|---|
| `hanc_ml` | `HancMl` | `pydaltoolbox` classificadores (`ClaRf/Svm/...`) | ✅ |
| `hanct_dtw` | `HanctDtw` | DTW (`dtaidistance` ou `tslearn`) | 🆕 `[dtw]` |
| `hmu_pca` | `HmuPca` | PCA multivariado (`sklearn`/`DtPca`) | ✅ |

### Motif / discord (simbólico + matrix profile)

| R | Python (proposto) | Backend | Dep |
|---|---|---|---|
| `trans_sax` | `TransSax` | SAX (z-norm + binning por quantil) | 🔨 nativo |
| `trans_xsax` | `TransXsax` | xSAX | 🔨 nativo |
| `hmo_sax` | `HmoSax` | motif via SAX | 🔨 nativo |
| `hmo_xsax` | `HmoXsax` | motif via xSAX | 🔨 nativo |
| `hdis_sax` | `HdisSax` | discord via SAX | 🔨 nativo |
| `hmo_mp` | `HmoMp` | matrix profile (`stumpy`) | 🆕 `[motif]` |
| `hdis_mp` | `HdisMp` | matrix profile (`stumpy`) | 🆕 `[motif]` |

### Ensembles / visualização / streaming / infra

| R | Python (proposto) | Backend | Dep |
|---|---|---|---|
| `mas` | `mas` (util) | média móvel | ✅ |
| `har_ensemble` | `HarEnsemble` | combinação de detectores | ✅ |
| `har_ensemble_fuzzy` | `HarEnsembleFuzzy` (+`fuzzify`) | ensemble fuzzy | ✅ |
| `har_plot` | `har_plot` | `matplotlib` | 🆕 `[plot]` |
| `har_plot_ensemble` | `har_plot_ensemble` | `matplotlib` | 🆕 `[plot]` |
| `har_online_*` | runners online (memory/sources/executor/session) | puro | ✅ |
| `har_stream_eval` / `har_stream_experiment` | avaliação em streaming | puro | ✅ |
| `united` / `united_loadfulldata` | experimento "united" | puro | ✅ |
| datasets `examples_motifs`, `examples_harbinger` | extrair do R | — | ✅ |

> ## ⚠️ PENDÊNCIA CRÍTICA — datasets miniatura + `loadfulldata` (corrigir ao chegar na 0.7.0)
>
> **Este ponto está SUBESTIMADO acima e precisa ser acertado — igual ao que fizemos no
> pytspredit (0.7.0).** O `harbinger` R usa o MESMO padrão **miniatura-no-pacote +
> `loadfulldata` remoto** que o `tspredit`, e o plano atual não reflete isso:
>
> - `united_loadfulldata` **NÃO é "puro"** — é exatamente o `loadfulldata` do tspredit:
>   lê `attr(x, "url")` da miniatura e **baixa o `.RData` completo** de um repo remoto
>   (ver `harbinger/R/united_loadfulldata.R`; assinatura `loadfulldata(x, envir=)`,
>   aceita nome ou objeto, carrega no `envir`).
> - Os **datasets de benchmark de detecção estão FALTANDO no roadmap acima.** O
>   `harbinger/data/` embarca **miniaturas** (cada uma com `attr(url)`): `A1Benchmark`…
>   `A4Benchmark` (Yahoo), `gecco`, `mit_bih_MLII/V1/V2/V5`, `nab_*` (6), `oil_3w_*` (7),
>   `ucr_*` (ecg/int_bleeding/nasa/power_demand), além de `examples_*`. Verificar as urls
>   reais com `Rscript` (análogo ao que fiz no tspredit).
>
> **Correção exigida (fazer IGUAL ao pytspredit — ver `C:\Git\dal\pytspredit\develop\
> planejamento.md` §5.4 / release 0.7.0 e a tabela de deps):**
> 1. **Miniaturas embarcadas** no pacote, convertidas do `.RData` via `Rscript` para
>    formato neutro (CSV para data.frame; JSON para listas) + `manifest.json` com
>    `{nome: {url, kind}}`. Sem dependência nova; funciona offline.
> 2. **`datasets.loadfulldata(nome|obj)`** num **extra `[bench]`** usando um leitor RData
>    puro-Python (`rdata`, talvez `pyreadr`) que baixa o `.RData` completo das **MESMAS
>    urls remotas** que o R usa (nada de reprocessar/rehospedar). `examples_*` pequenos
>    seguem embarcados como estão (didáticos).
>
> **Seu eu do futuro: ao trabalhar na release de datasets/streaming, NÃO trate
> `loadfulldata` como trivial — replique o mecanismo do pytspredit aqui.**

**Resumo das deps NOVAS (só 5 libs pesadas), todas como *extras* opcionais:**
`arch` (`[garch]`), `EMD-signal` (`[emd]`), `PyWavelets` (`[wavelet]`),
`stumpy` (`[motif]`), DTW lib (`[dtw]`). **`matplotlib` é leve/ubíqua ⇒ dep
obrigatória** (plots sempre disponíveis; sem extra `[plot]`). Tudo o mais é ✅
(reuso) ou 🔨 (nativo). `torch` já entra transitivo via `pydaltoolbox` para
autoencoder/waypoint — **sem dep direta nova**.

## Política de dependências (native vs. dependency)

1. **Núcleo enxuto, imutável:** `pydaltoolbox`, `pytspredit`, `numpy`, `pandas`,
   `scikit-learn`, `scipy`, `ruptures`, `matplotlib`. `pip install pyharbinger` nunca
   puxa backend pesado. (`matplotlib` entra no núcleo por ser leve/ubíqua — decisão
   "misto"; `torch` só chega transitivo via pydaltoolbox, quando o autoencoder é usado.)
2. **Backend pesado/especializado ⇒ *extra* opcional + import preguiçoso** que,
   se faltar, levanta erro acionável nomeando o extra. Espelha o padrão do próprio R
   (`Suggests` + `requireNamespace`, como `hcp_bocpd` faz com `ocp`). Pesadas =
   `stumpy`, `arch`, `EMD-signal`, DTW, `PyWavelets`.
3. **Preferir NATIVO (🔨)** quando o algoritmo é pequeno e bem-especificado **ou** a
   única lib candidata é pesada/abandonada frente às ~50–150 linhas necessárias:
   SAX/xSAX, ChangeFinder-LR, Chow/GFT/joinpoint/SCP (OLS), BOCPD (Adams&MacKay),
   `mas`. (BOCPD: o R deixa `ocp` em *Suggests*; em Python não há port mantido → nativo.)
4. **Preferir DEPENDÊNCIA (🆕)** quando o algoritmo é grande/numericamente sutil e há
   lib canônica, mantida e com *wheel*: GARCH→`arch`, EMD→`EMD-signal`,
   wavelet→`PyWavelets`, matrix profile→`stumpy`, DTW→`dtaidistance`. Reimplementar
   isso à mão seria frágil e violaria o princípio "não reimplementar".
5. **Gate de release:** cada dep nova é validada quanto a *wheel* para a versão de
   Python alvo (lição do `ruptures` 1.0.6 + `--prefer-binary` no py3.14) antes de entrar.

## Releases graduais — DOIS CICLOS

Estrutura em **duas rodadas** sobre os mesmos temas:

- **Ciclo 1 (baixa dependência):** varre *todos* os temas, mas só as classes **sem dep
  pesada nova** (✅ reuso / 🔨 nativo). Entrega quase toda a cobertura do R mantendo o
  núcleo enxuto (nenhum `stumpy`/`arch`/`EMD-signal`/`PyWavelets`/DTW).
- **Ciclo 2 (dependências pesadas):** volta aos *mesmos temas* acrescentando as classes
  de maior dependência — **uma família de dep por release**, cada uma como *extra*
  opcional. **A ser refinado com o Eduardo tema a tema** (ordem/escopo abaixo é esboço).

Granularidade livre (pode ter muitos ciclos pequenos; sem medo de ~20 releases até a
1.0.0). Cada release = branch próprio, paridade R, testes + ruff + notebooks verdes,
revisão do Eduardo, ff-merge (mesma cadência da v1).

### Ciclo 1 — baixa dependência (sem dep pesada nova)

- **0.2.0 — Aprendizado & multivariado.** Infra: caminho **multivariado** em
  `_store_refs`/`_restore_refs` (entrada DataFrame). Detectores: `HancMl`,
  `HanAutoencoder`, `HmuPca` (todos via pydaltoolbox; torch já transitivo). Infra
  amortizada em três detectores.
- **0.3.0 — Espectral (anomalia).** `mas` + os 5 híbridos FFT
  (`HanrFftAmoc/Binseg/Sma/AmocCusum/BinsegCusum`).
- **0.4.0 — Change point estendido.** `HcpCfLr`, `HcpCfArima`, `HcpCfEts`, `HcpChow`,
  `HcpGft`, `HcpJoinpoint`, `HcpScp`, `HcpBocpd` (nativo Adams&MacKay), `HcpWaypoint`.
- **0.5.0 — Simbólico (motif/discord nativo).** SAX/xSAX: `TransSax`, `TransXsax`,
  `HmoSax`, `HmoXsax`, `HdisSax`. Embarca dataset `examples_motifs`.
- **0.6.0 — Ensembles + visualização.** `HarEnsemble`, `HarEnsembleFuzzy` (+`fuzzify`) +
  `har_plot`/`har_plot_ensemble` (`matplotlib`, dep do núcleo).
- **0.7.0 — Streaming / online (fecha o Ciclo 1).** Framework online
  (memory/sources/executor/session) + `har_stream_eval`/`har_stream_experiment` +
  `united`/`united_loadfulldata`. Embarca dataset `examples_harbinger`.

Ao fim do Ciclo 1: todo o R portado **exceto** as 5 famílias de dep pesada.

### Ciclo 2 — dependências pesadas (extras opcionais) — *esboço, a refinar*

Uma família de dep por release; cada detector faz import preguiçoso + erro acionável.
Ordem/escopo **a combinar com o Eduardo** (ele refina este ciclo em detalhe):

- **0.8.0 — Motif/discord por matrix profile.** `HmoMp`, `HdisMp` — extra `[motif]` (`stumpy`).
- **0.9.0 — GARCH.** `HanrGarch`, `HcpGarch` — extra `[garch]` (`arch`).
- **0.10.0 — EMD.** `HanrEmd`, `HanrRemd`, `HanrRtad` — extra `[emd]` (`EMD-signal`).
- **0.11.0 — Wavelet.** `HanrWavelet` — extra `[wavelet]` (`PyWavelets`).
- **0.12.0 — DTW.** `HanctDtw` — extra `[dtw]` (`dtaidistance`/`tslearn`).

### Fechamento

- **1.0.0 — Paridade + estabilização.** API pública congelada, um notebook executável
  por classe, **matriz completa de paridade R↔Python**, site de documentação, passada
  de depreciações. Marca paridade de funcionalidades com o R.

## Decisões travadas (2026-11-19, Eduardo)

- **(A) Empacotamento = misto.** Deps leves/ubíquas (`matplotlib`) obrigatórias no
  núcleo; só as pesadas (`stumpy`, `arch`, `EMD-signal`, DTW, `PyWavelets`) como
  *extras* opcionais com import preguiçoso + erro acionável (padrão `Suggests` do R).
- **(B) Granularidade = dois ciclos temáticos.** Ciclo 1 (0.2→0.7) varre todos os
  temas com baixa dependência; Ciclo 2 (0.8→0.12, *esboço*) volta aos temas com as
  classes de dep pesada, uma família por release — **Eduardo refina o Ciclo 2 em
  detalhe**. Granularidade livre (muitos ciclos pequenos OK; pode chegar a ~20 releases).
- **(C) ⚠️ Datasets miniatura + `loadfulldata` = replicar o padrão do pytspredit.**
  Ver a **PENDÊNCIA CRÍTICA** na seção de Ensembles/infra acima: o `harbinger` usa
  miniatura-no-pacote + `loadfulldata` que baixa o `.RData` completo de url remota
  (benchmark de detecção: Yahoo A1–A4, gecco, MIT-BIH, NAB, oil_3w, UCR). Corrigir na
  0.7.0 IGUAL ao pytspredit (miniatura embarcada + `[bench]` extra com leitor RData).
