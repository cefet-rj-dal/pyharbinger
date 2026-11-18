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
