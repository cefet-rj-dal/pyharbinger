# Planejamento — o que falta no pyharbinger

> Documento vivo. **Contém só o que resta fazer.** A história do que já foi concluído está
> em `historico.md`; a correspondência de classes R↔Python em `mapeamento.md`; a metodologia
> e a matriz de paridade em `parity.md`; o kickoff em `PROMPT.md`.

Última atualização: 2026-11-26 (sessão da tarde)

---

## Onde estamos (resumo)

Port Python do R **harbinger** (detecção de eventos em TS), último da família DAL, depende de
**pydaltoolbox + pytspredit** (PyPI). Núcleo + avaliação + anomalias base + FFT-híbridos +
change points (ruptures/nativos/os 3 **++** strucchange/forecast/ocp) + clustering +
multivariado + ensemble fuzzy + `HanrArima` (ARIMA++) + `HanrFftSma` + **simbólico
(`TransSax`/`TransXsax`/`HmoSax`/`HmoXsax`/`HdisSax`)** já prontos. **PyPI latest = 0.5.0**
(`HanrFftSma`, publicada em 2026-11-25). História completa: `historico.md`.

---

## O que falta — roadmap até a 1.0.0

> **0.6.0 simbólico PUBLICADA** (PyPI, mesclada em `main` HEAD `47c01a2`). Detalhes em
> `historico.md`.

### Ciclo 1 — baixa dependência (fechar a cobertura, núcleo enxuto)
1. ✅ **Simbólico (motif/discord nativo)** — `TransSax`/`TransXsax`, `HmoSax`/`HmoXsax`/
   `HdisSax` (SAX/xSAX, nativo). Embarca `examples_motifs`. **Feito na 0.6.0** (paridade
   exata R↔Python nas 3 séries). R `hdis_sax` rotulava o resultado como `type=="motif"`
   (lapso de copy-paste) → **corrigido para `"discord"` nos dois lados** (paridade é sobre o
   conjunto de índices, não sobre `type`). Commit do R pendente de versão Boeing (ver abaixo).
2. ⬜ **Ensemble simples + visualização** — `HarEnsemble` (não-fuzzy) + `har_plot`/
   `har_plot_ensemble`/`har_ensemble_plot_models` (matplotlib, dep do núcleo).
3. ⬜ **Streaming / online + datasets (fecha o Ciclo 1)** — framework online (memory/sources/
   executor/session) + `har_stream_eval`/`har_stream_experiment` + `united`/`loadfulldata`.
   ⚠️ **`loadfulldata` NÃO é trivial** — replicar o mecanismo do pytspredit: **miniaturas
   embarcadas** (CSV/JSON + `manifest.json{url,kind}`) + **`datasets.loadfulldata(nome)`** que
   baixa o completo das MESMAS urls remotas. Benchmark de detecção: Yahoo A1–A4, gecco, MIT-BIH,
   NAB, oil_3w, UCR, + `examples_harbinger`.

### Ciclo 2 — dependências pesadas (extras opcionais) — *a refinar com o Eduardo*
Uma família de dep por release; import preguiçoso + erro acionável nomeando o extra. Para cada
uma, decidir **embrulhar vs. back-implementar ++** pelo gate (nicho/atrito/paridade):
4. ⬜ **Motif/discord por matrix profile** — `HmoMp`/`HdisMp`, extra `[motif]` (`stumpy`).
5. ⬜ **GARCH** — `HanrGarch`/`HcpGarch`, extra `[garch]` (`arch`).
6. ⬜ **EMD** — `HanrEmd`/`HanrRemd`/`HanrRtad`, extra `[emd]` (`EMD-signal`). *(Candidato a
   back-impl nativo 0-dep, como no pytspredit.)*
7. ⬜ **Wavelet** — `HanrWavelet`, extra `[wavelet]` (`PyWavelets`). *(Idem — MODWT nativo já
   existe no pytspredit; avaliar reuso.)*
8. ⬜ **DTW** — `HanctDtw`, extra `[dtw]` (`dtaidistance`/`tslearn`).
9. ⬜ **Deep learning** — `HanAutoencoder`/`HcpWaypoint`, extra `[deep]` (`torch`). *(O
   `AutoencBaseEd` do pydaltoolbox é stub identidade — precisa de torch de verdade.)*

### Fechamento
10. ⬜ **Maturidade (espelhar pydaltoolbox/pytspredit)** — promover as checagens asseridas para
    `develop/parity/<fam>.R` + `tests/test_quality_*` (R-free); auditoria de crédito Py+R
    (Teoria+References+Porting, frases proibidas=0); metadados (`Development Status 4 - Beta`,
    autores do port). → **release BETA** (próxima `0.x.0`, roda ~1 mês).
11. ⬜ **1.0.0 — estável** — API congelada, matriz de paridade completa. **Pós-1.0.0 adota
    numeração Boeing** (como tspredit/harbinger R).

## Pendências externas (Eduardo faz)
- Renderização do pipeline de docs do **R harbinger** (examples/*.md + docs/) e submissão ao CRAN
  (master 2.1.787; `strucchange`/`ocp`/`forecast` já removidos; resta `changepoint` + libs do
  Ciclo 2).

## Candidatos de back-implementation ainda abertos (caso a caso)
- `ruptures`→PELT/Dynp/BinSeg nativos (remove a dep do núcleo + o atrito de wheel no py3.14;
  revalida HcpAmoc/Binseg/Pelt + cutoff dos híbridos FFT) — P3, maior esforço.
- EMD/wavelet nativos no Ciclo 2 (reusar os de pytspredit) em vez de `EMD-signal`/`PyWavelets`.

---

## Princípios ainda em vigor (guiam o que falta)

- **Convenção de docstring:** toda classe de algoritmo traz **Teoria + `References:` (paper
  original) + `Porting:`** (`Mirrors <arquivo_r>.R`; fidelidade exact/approximate/adapted). Sem
  frases proibidas; redução de dep vai na **mensagem de commit**, não na doc. Um `++` cita o
  original **e** os artigos que inspiraram a inovação (autoria auditável).
- **Back-implementation ++ (Python = releitura canônica, R = backport):** trocar dep por nativo
  **só** em casos nicho/atípicos (lib falta/não mantida/atrito de wheel/auto-seleção quebra
  paridade). Libs canônicas seguem embrulhadas (sklearn/scipy/ruptures/stumpy/arch/EMD-signal/
  PyWavelets). **Não é regra cega — caso a caso com o Eduardo.**
- **Gate Ação 1++:** para um `++`, o teste é **"detecta a verdade tão bem ou melhor"** —
  **SoftED-F1 vs a coluna `event`** (ground truth), comparado ao F1 do pacote de referência,
  **não** paridade com ele. Instrumento: `develop/benchmark_backimpl.py`. Para um port fiel, o
  teste segue sendo paridade com o R (conjunto detectado + scores).
- **Paridade a cada passo:** rodar R e Python sobre o mesmo input congelado (`examples_*`),
  comparar o **conjunto de índices detectados** (não o vetor de score), e travar num guard.
  **Bug em qualquer lado corrige-se nos dois repos.** Reference não é verdade.
- **Gate de dep:** validar *wheel* da dep nova para o Python alvo antes de entrar (lição do
  `ruptures` 1.0.6 + `--prefer-binary` no py3.14).
- **Como trabalhamos:** incremental, verde sempre (`ruff` + `pytest` + `python
  notebooks/conversor.py`); **branch por milestone** → push → Eduardo revisa no GitHub →
  ff-merge (sem `gh`); nada na `main` sem o Eduardo pedir; cada `twine upload` e cada
  commit/push no R exigem o **"pode fazer"** dele.
