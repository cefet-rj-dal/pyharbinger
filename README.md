# pyharbinger

Python port of the R [harbinger](https://github.com/cefet-rj-dal/harbinger) — a **framework
for time series event detection** (anomalies, change points, and motifs/discords), built on
the DAL Toolbox *Experiment Line*.

`pyharbinger` unifies many detection methods under one interface (`fit` → `detect` →
`evaluate`) and a common evaluation protocol, so different detectors can be compared fairly —
reusing classic libraries (scikit-learn, numpy, pandas, ruptures, stumpy, scipy) through thin
adapters.

It is the capstone of the DAL Python family:

- Builds on **[pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox)** (foundation) and
  **pytspredit** (time series prediction) — its regressors power the forecasting-based
  anomaly detectors.

## Status

🚧 **Planned / in development** (after `pytspredit`, which it depends on). The port follows
the same method validated for `pydaltoolbox`. The full kickoff brief and roadmap are in
[develop/PROMPT.md](develop/PROMPT.md).

Once released:

```bash
pip install pyharbinger
```

## Related DAL projects

- [pydaltoolbox](https://github.com/cefet-rj-dal/pydaltoolbox) — foundation (Python)
- pytspredit — time series prediction (Python)
- [daltoolbox](https://github.com/cefet-rj-dal/daltoolbox) / [tspredit](https://github.com/cefet-rj-dal/tspredit) / [harbinger](https://github.com/cefet-rj-dal/harbinger) — R originals
- [Data Analytics Lab](https://eic.cefet-rj.br/~dal/)

## License

MIT
