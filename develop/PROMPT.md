# Kickoff prompt — portar `harbinger` (R) para Python (`pyharbinger`)

> Cole/aponte este arquivo no início de uma nova sessão. Ele te dá tudo para retomar
> o port no mesmo padrão com que já fizemos o **pydaltoolbox** (publicado no PyPI).
> **Pré-requisito:** o `pytspredit` precisa estar portado antes (o harbinger depende dele).

---

## 0. Missão

Portar o pacote R **`harbinger`** (detecção de eventos em séries temporais — anomalias,
change points, motifs/discords) para Python como **`pyharbinger`**. É o **último** da
família DAL: depende de **`daltoolbox` E `tspredit`** → ou seja, reutiliza
**`pydaltoolbox` + `pytspredit`**.

Princípio central (o mesmo de sempre): **não reimplementar algoritmos** — envolver libs
clássicas (`scikit-learn`, `numpy`, `pandas`, `statsmodels`, `ruptures`, `stumpy`, ...) com
a **interface uniforme da Linha de Experimento**. A contribuição é a interface e o
**framework unificado de detecção + avaliação de eventos**.

## 1. Leia primeiro (pastas de referência)

| Pasta | O que é |
|---|---|
| `C:\Git\dal\pydaltoolbox` | **O PADRÃO A COPIAR** (estrutura, convenções, testes, exemplos, deploy). |
| `C:\Git\dal\pytspredit` | O port de TS (dependência direta) — reutilize `TsData`, `ts_projection`, regressores. |
| `C:\Git\dal\daltoolbox` / `C:\Git\dal\tspredit` | Pacotes R fundacionais (origem). |
| `C:\Git\dal\harbinger` | **O código R a portar** (fonte desta missão). |
| Memória | `C:\Users\eduar\.claude\projects\c--Git-dal-pydaltoolbox\memory\pydaltoolbox-port.md` |

A pasta atual (`C:\Git\dal\pyharbinger`) é um scaffold vazio, como os outros começaram.

## 2. Convenções já estabelecidas (replique do pydaltoolbox)

Idênticas às do pydaltoolbox/pytspredit — resumo:
- **API:** métodos de instância, `fit` encadeável; **PascalCase** (ex.: `Harbinger`,
  `HanrMl`, `HcpPelt`, `HarEval`).
- **Dados:** pandas/numpy. Série = vetor/Series; detecção = DataFrame `idx/event/type`.
- **Adapters finos** fiéis ao backend R; **comparar R vs Python** nos pontos sensíveis.
- **Layout `src/pyharbinger/`**, `pyproject` hatchling, deps mínimas.
- **3 camadas:** `pytest` unit + integração + **documentação executável** (notebooks
  jupytext → `examples/` via `tools/build_examples.py`; **um exemplo didático por classe**,
  com objetivo + referência + `fit`/`detect`/`evaluate`).
- **`develop/`** para material de dev (este PROMPT, planejamento.md, deploy.md),
  excluída do sdist (copie a config de `[tool.hatch.build.targets.sdist]` do pydaltoolbox).
- **Sempre verde** (`ruff` + `pytest` + build dos exemplos).
- **Git:** branch → push → **Eduardo revisa no GitHub** → pausa por fase → ff-merge na main.
  `gh` não instalado. Commits terminam com `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Deploy:** PyPI via `twine` (`.pypirc` já tem o token `daldeploy`). Beta até `1.0.0`;
  versões imutáveis; Eduardo autoriza/roda. Guia: `pydaltoolbox/develop/deploy.md`.

## 3. Dependências — reutilize pydaltoolbox + pytspredit

```python
from pydaltoolbox import DalBase, ClusterKmeans, AutoencBaseEd, ...   # base, clustering, autoenc
from pytspredit import TsData, ts_projection, TsNormGminmax, TsReg, ...  # janelas e regressores
```
Do código R (confirmado por análise): o harbinger usa do daltoolbox **`dal_base`,
`regression`/`classification`/`predictor`, `evaluate`, `autoenc_base_ed`, `minmax`,
`adjust_data.frame`**; e do tspredit **`ts_data`/`ts_projection`, `ts_norm_*`, regressores**.
O `hanr_ml` (detector por regressão) é a **ponte** que prova o ecossistema: ele envolve um
regressor do pytspredit, prevê a série e marca resíduos grandes como anomalia.

## 4. Arquitetura do `harbinger` (64 arquivos em `R/`) — o que portar

- **Base + contrato** (`harbinger.R`): `harbinger()` estende `dal_base`; genérico
  **`detect(obj, serie)`** (análogo a `predict`/`cluster`/`discover`); guarda refs da série
  e restaura os índices das detecções. **Saída da detecção:** DataFrame com
  `idx` (índice), `event` (bool), `type` ∈ {`"anomaly"`, `"changepoint"`, `""`}.
  `evaluate.harbinger` delega para `har_eval`.
- **Utils** (`harbinger_utils*`): medidas de desvio (ex. L2), critérios de filtro
  (ex. gaussiano), seleção de candidatos.
- **Avaliação** (`har_eval.R`, `har_eval_soft.R`): compara eventos detectados vs
  ground truth → TP/FP/FN, precision/recall/F1, com janela de tolerância; a versão "soft"
  usa atribuição húngara (`RcppHungarian` → `scipy.optimize.linear_sum_assignment`).
- **Detectores de anomalia** (`hanr_*`): `hanr_ml` (regressão do pytspredit + resíduos —
  **faça este cedo**), `hanr_fft`, `hanr_hist`, `hanr_fbiad`, `hanr_arima`, `hanr_garch`,
  `hanr_rtad`, `hanr_emd`/`hanr_remd`/`hanr_wavelet` (estes 3 precisam de PyEMD/PyWavelets →
  Bloco B).
- **Por classificação/clustering:** `hanc_ml` (classificador do pydaltoolbox),
  `hanct_kmeans` (usa KMeans — `stats::kmeans` no R; use `sklearn`/`ClusterKmeans`),
  `hanct_dtw` (DTW → `dtaidistance`/`tslearn`, talvez Bloco B).
- **Change points** (`hcp_*`): `amoc`, `binseg`, `pelt` (→ `ruptures`), `bocpd`, `chow`,
  `kswin`, `page_hinkley`, `cf_arima`/`cf_ets`/`cf_lr`, `garch`, `gft`, `joinpoint`,
  `scp`, `waypoint`.
- **Motif/discord** (`hdis_mp`, `hmo_mp` via matrix profile → `stumpy`; `*_sax`/`xsax` via
  representação SAX — `trans_sax.R`/`trans_xsax.R`), `hmu_pca`.
- **Ensembles** (`har_ensemble`, `har_ensemble_fuzzy`), **streaming/online**
  (`har_online_*`, `har_stream_*`), **plots** (`har_plot*`), `mas`, `united*` →
  majoritariamente **Bloco B**.

## 5. Critério de escopo da v1 (mesmo princípio)

Entra na v1 o que é wrapper fino de libs já disponíveis:
- **Núcleo:** `Harbinger` base + `detect` + `HarEval`/`HarEvalSoft` (scipy Hungarian).
- **`HanrMl`** (ponte com pytspredit) + anomalias simples (`fft`, `hist`, `fbiad`).
- **Change points** via `ruptures` (`amoc`/`binseg`/`pelt`) + detectores de janela
  deslizante estatísticos simples (`kswin`, `page_hinkley`).
- **`HanctKmeans`** (sklearn) e, se `stumpy` for aceitável, motif/discord por matrix profile.

Adie para **Bloco B** (lib extra/complexo): EMD/wavelet (PyEMD/PyWavelets), GARCH (`arch`),
DTW pesado, SAX avançado, ensembles, streaming/online e gráficos.

**Deps da v1:** `pydaltoolbox`, `pytspredit`, `numpy`, `pandas`, `scikit-learn`, `scipy`,
`ruptures` (change points). `statsmodels`/`stumpy`/`arch` conforme o escopo escolhido —
avalie como extras opcionais.

## 6. Roadmap sugerido (em fases/sub-blocos, pause a cada um)

0. Scaffolding (deps `pydaltoolbox`+`pytspredit`), `src/`, `develop/`, `tools/build_examples.py`.
1. **Núcleo:** `Harbinger` base + genérico `detect` + utils (desvio/filtro/seleção) +
   datasets de exemplo (`examples_anomalies`, `examples_changepoints` do R `data.R`).
2. **Avaliação:** `HarEval` (TP/FP/FN, precision/recall/F1 com tolerância) + `HarEvalSoft`
   (Hungarian via scipy). Teste com detecção sintética.
3. **Slice vertical:** `HanrMl` (regressor do pytspredit + resíduos) ponta-a-ponta —
   prova que pydaltoolbox→pytspredit→pyharbinger encaixam.
4. **Anomalias estatísticas simples:** `HanrFft`, `HanrHist`, `HanrFbiad`.
5. **Change points (ruptures):** `HcpAmoc`, `HcpBinseg`, `HcpPelt` (+ `HcpKswin`, `HcpPageHinkley`).
6. **Clustering-based:** `HanctKmeans` (+ `HancMl`).
7. **Motif/discord** (se `stumpy`): `HdisMp`, `HmoMp`.
8. Fechamento + publicar `pyharbinger` 0.1.0 no PyPI.
Bloco B: EMD/wavelet/GARCH/DTW/SAX, ensembles, streaming, gráficos.

*(Confirme com o Eduardo e mantenha um `develop/planejamento.md` vivo.)*

## 7. Ritual a cada fase
Implementar + testes → notebook do tema → `build_examples` → `ruff`+`pytest` verdes →
commit no branch + push → **Eduardo revisa no GitHub** → ff-merge na main.

## 8. Primeiro passo concreto
Confirme que `pytspredit` já está pronto/instalável (`pip install pytspredit` ou editable).
Leia `pydaltoolbox` (padrão), `harbinger/R/harbinger.R`, `har_eval.R`, `hanr_ml.R`,
`hcp_pelt.R`, `harbinger_utils*.R`. Proponha o planejamento ao Eduardo e comece pela
Fase 0→2 (núcleo + avaliação), depois o slice `HanrMl`.
