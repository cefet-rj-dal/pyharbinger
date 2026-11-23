# Kickoff prompt — portar `harbinger` (R) para Python (`pyharbinger`)

> Cole/aponte este arquivo no início de uma nova sessão. Ele te dá tudo para retomar
> o port no mesmo padrão com que já fizemos o **pydaltoolbox** e o **pytspredit**
> (ambos publicados no PyPI).
> **Pré-requisito ATENDIDO:** `pytspredit 0.1.0` já está no PyPI (`pip install pytspredit`)
> e `pydaltoolbox 0.1.0` também. **Leia a §9 (lições da sessão pytspredit) antes de começar.**

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
  jupytext → `examples/` via `notebooks/conversor.py`; **um exemplo didático por classe**,
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

0. Scaffolding (deps `pydaltoolbox`+`pytspredit`), `src/`, `develop/`, `notebooks/conversor.py`.
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
Leia a **§9** (lições da sessão pytspredit). Confirme `pytspredit`/`pydaltoolbox`
instalados (`python -c "import pytspredit, pydaltoolbox"`). Leia `pydaltoolbox` (padrão),
`harbinger/R/harbinger.R`, `har_eval.R`, `hanr_ml.R`, `hcp_pelt.R`, `harbinger_utils*.R`.
**Proponha o planejamento ao Eduardo (em milestones, não 10 fases) e confirme com ele
o escopo da v1 e a cadência de revisão ANTES de codar.** Depois comece pela Fase 0→2.

---

## 9. Lições da sessão pytspredit (aplicar aqui) — 2026-11-18

O port do `pytspredit` foi feito e **publicado (0.1.0) no PyPI num único dia**. O que o
Eduardo pediu e o que descobrimos — replique:

### Workflow que o Eduardo fixou (preferências dele)
- **Planeje em MILESTONES, não em 10 fases com 10 pausas.** No começo, proponha 3–4
  milestones consolidadas com meta de release, e **pergunte a ele** (use AskUserQuestion):
  (a) **escopo da v1** (trilha enxuta vs completa — o que adia para 0.2.0/Bloco B) e
  (b) **cadência de revisão** (pausar a cada milestone vs uma revisão final antes do upload).
  No pytspredit ele escolheu **trilha enxuta + pausar a cada milestone**.
- **Branch por milestone**, cada um baseado no anterior (ex.: `milestone-2-...` sai do
  `milestone-1-...`), para o PR mostrar só o diff daquela milestone. Merge final: ff na `main`.
- **Exemplos SEMPRE commitados** e, ao pausar, **mande os links do GitHub de cada exemplo**
  (`https://github.com/cefet-rj-dal/pyharbinger/blob/<branch>/examples/...`). Ele revisa por ali.
  `gh` não está instalado → o PR ele abre pelo link de compare; você só empurra os branches.
- **Documento vivo** `develop/planejamento.md` atualizado a cada milestone (decisões, status).
- **Convenção de exemplos:** um exemplo por classe, mas **agrupados por família num
  notebook** (ex.: `anomaly/01-detectors.md` com uma seção por detector), não 1 arquivo por classe.

### Comparação R vs Python a CADA passo (exigência dele — dá segurança)
- **R 4.5.0 está disponível** em `/c/R/R-4.5.0` (`Rscript`). Para cada componente portado,
  rode o equivalente no `harbinger` (R) e no `pyharbinger` sobre o **mesmo dado** e
  **compare** (p/ detecção: `idx/event/type` e métricas TP/FP/FN/precision/recall/F1).
- Padrão que funcionou: um `*_ref.R` gera CSVs de referência; um `*_cmp.py` recomputa e
  compara com tolerância. Determinísticos devem bater ~1e-9; onde o backend/RNG difere
  (RF, MLP, k-means init, MLE do ARIMA/GARCH), **documente a tolerância** no planejamento.
- **Extraia datasets do R fielmente:** `Rscript` para ler o `.RData` → salve CSV em
  `src/pyharbinger/data/` e **verifique numericamente** (foi assim com o `tsd`).
- Para testar um **branch de patch no pacote R** sem reinstalar: `pkgload::load_all(".")`.

### Armadilhas técnicas concretas (vão reaparecer)
- **Teste de install em venv limpo é OBRIGATÓRIO antes de publicar.** Ele pegou um bug real:
  `df.to_numpy()` vem **read-only** no numpy novo (2.5) → escrita in-place quebra. Use
  `np.array(df.to_numpy())` quando for escrever. O dev env tinha numpy antigo e mascarou.
- **O ruff deste ambiente tem ruleset LARGO** (B/BLE/S/RUF/UP ativos, não só E/F). Mantenha
  limpo: `ClassVar` em dict de classe mutável; nada de `except Exception:` sem `# noqa: BLE001`;
  sem variável/desempacote não usado; sem aspas redundantes em anotação (há `from __future__`).
- **Paridade estatística exata com o R precisa do backend certo:** `statsmodels.acf(adjusted=False)`
  = `stats::acf` do R; `statsmodels.pacf(method="ywm")` = `stats::pacf`; **`np.quantile(method="median_unbiased")`
  = `quantile(type=8)` do R** (discretização/MI). Para k-means (`hanct_kmeans`), a init/RNG difere
  do `stats::kmeans` → tolerância.
- **Fidelidade a defaults do R:** replique defaults do backend (no pytspredit, os `TsNorm*`
  aplicam `OutliersBoxplot()` por padrão sobre os dados transformados; o SVM replica o
  `scale=TRUE` do e1071). Confira caso a caso.
- **Cheque herança/MV:** no tspredit, `ts_arimax` era **multivariado** (`ts_reg_mv`) → foi para
  o Bloco B com a família MV. No harbinger, veja quais detectores puxam dep adiada
  (EMD/wavelet/GARCH/DTW/SAX) e adie junto.
- **Bugs latentes no R:** achamos e corrigimos um (`ts_norm_none` ignorava o alvo no x-path;
  quebrava `ts_regsw`). Se achar algo parecido no `harbinger`, **conserte num branch do
  pacote R** (`c:\Git\dal\harbinger`), verifique com `load_all`, **não bumpe a versão do
  DESCRIPTION**, deixe na branch para o Eduardo revisar/mergear (release CRAN é dele). O fix do
  tspredit está no branch `fix-ts-norm-none-target-path`.

### Deploy (confirmado nesta sessão)
- `python -m build` + `twine check` + **install em venv limpo** → só então `twine upload`.
- O token **`daldeploy`** (conta inteira) está em `C:\Users\eduar\.pypirc` [pypi] e serve para
  todos os pacotes DAL. **O Eduardo autoriza explicitamente** o upload (publicar é irreversível;
  versões imutáveis; beta até 1.0.0). sdist exclui `develop/notebooks/examples/tests`.
- Depois de publicar: ff-merge dos branches na `main`, badge de PyPI + status "published" no README.

### Memória
- Leia também `C:\Users\eduar\.claude\projects\c--Git-dal-pytspredit\memory\` (notas do port de TS
  e o memo `workflow-r-vs-python-check`), além do memo do pydaltoolbox citado na §1.
