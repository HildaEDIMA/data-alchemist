# Data Alchemist

**A flight recorder for AI data analysts.** It records every statistical test an agent runs, writes the report itself, and only keeps findings that survive confirmation on data the agent never saw.

Built for the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (Oct 26 – Nov 4, 2026).

## The problem

Ask an AI agent "what drives our sales?" and it will run dozens of tests, then show you the few that look impressive. You never see how many were run. At alpha = 0.05, 14 hidden tests give a 51% chance of at least one false positive; 45 give 90% (1 − 0.95^m).

We measured it on Muse Spark. On 5 synthetic datasets with **no real effect**, it made 56 claims: 39 correct denials and **17 asserting a link that does not exist** (0 to 7 per dataset). Nothing in its answer says which. After our engine's check, **0 of the 17 were kept**.

These are early numbers: 5 datasets, one prompt, synthetic data. See [Limits](#limits).

## How it works

```
Agent  ──►  Test API (registry)  ──►  Engine report  ──►  Confirmation on held-out half
              logs every test         written by the        run once, after the number of
              (variables, subsample,  engine, never by      promoted hypotheses is frozen
              code)                   the agent
```

- **Registry** (`core/registry.py`): the agent gets p-values only through our test API (correlation, chi-square, ANOVA, Mann-Whitney, regression with adjustment, nonlinear and interaction tests). Every call is logged, so the number of tests *m* is known. Unregistered or altered extractions are refused.
- **Engine-written report** (`core/report.py`): shows *m*, raw discoveries, and each rejected claim with its reason, plus a selection check comparing registry p-values with reported ones. The agent never writes conclusions.
- **Explore / confirm split** (`core/split.py`, after Cox 1975): the agent explores one half freely. The number of hypotheses to promote is fixed before the untouched half is opened, and it can be opened only once.
- **Claims** (`core/claims.py`): every claim must reference registered tests. A claim with no test behind it is rejected. Effects are reported as associations or "robust to adjustment", never as proven causes.
- **Multiplicity control** (`core/multiplicity.py`): Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli and e-BH, each with its assumptions stated.
- **Model adapter** (`core/agent_adapter.py`): Muse Spark, through the Meta Model API, is the first model audited. It explores only through the instrumented API and returns structured claims tied to registry ids. The engine itself is model-agnostic.

## Pre-registered protocol

`docs/PROTOCOL.md` fixes hypotheses, metrics and decision rules before the benchmark runs. The main rule:

> We say **p-hacking** only if the registry's p-values are uniform while the reported ones concentrate below 0.05. Otherwise the honest conclusion is **over-claiming**.

The benchmark covers: generator validity on null data, over-claiming on null data, p-hacking measured with the same API, prompt sensitivity, dose-response against test budget, planted effects (recall and false discovery proportion), and a comparison of correction methods on the same registries.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .
pytest -q                                          # automated tests, coverage on core/
python salmon_muse.py --mode null --datasets 5     # null benchmark with Muse Spark
```

Set your Meta Model API key in `.env` or in your environment before running the Muse benchmark. Results are written to `results/`.

## Status

| Part | Status |
|---|---|
| Test registry, report, split, claims, multiplicity corrections | built, with automated tests |
| Muse Spark adapter | built |
| Null-data benchmark (Muse Spark alone vs. engine) | run on 5 datasets, results in `results/` |
| Pre-registered protocol | written |
| Planted-effects benchmark (recall, false discovery proportion) | in progress |
| Scaling the null benchmark to 20+ datasets with intervals | planned |
| Dashboard and API | planned |
| Demo on real enterprise data (Olist) | planned |

## Limits

- Synthetic data, 5 datasets, one prompt so far: the 17-of-56 figure is an estimate, not a general rate.
- The column names in the first runs (`noise_num`, `noise_cat`) hinted at null data, so the figures are probably a lower bound. A rerun with neutral names is planned.
- Zero false claims kept on noise is the easy direction. The planted-effects benchmark is the real test, including the power cost of splitting the data.
- Row filters and recodings outside the logged tests are logged but cannot be fully eliminated.
- Guarantees hold only under their stated assumptions (BH under independence or positive dependence; BY and e-BH under arbitrary dependence).

## Related work

- Huang et al. (2025). *Automated Hypothesis Validation with Agentic Sequential Falsifications* (POPPER), ICML 2025, arXiv:2502.09858. Validates a given hypothesis with sequential falsification and e-values; we audit open-ended discovery and its test budget.
- Baumann et al. (2025). *Large Language Model Hacking*, arXiv:2509.08825.
- Cox (1975). A note on data-splitting for the evaluation of significance levels. *Biometrika* 62(2), 441–444.
- Benjamini & Hochberg (1995). *JRSS B* 57(1), 289–300. Benjamini & Yekutieli (2001). *Annals of Statistics* 29(4), 1165–1188.
- Vovk & Wang (2021). E-values. *Annals of Statistics* 49(3), 1736–1754. Wang & Ramdas (2022). *JRSS B* 84(3), 822–852.
- Dwork et al. (2015). The reusable holdout. *Science* 349(6248), 636–638.
- Bennett et al. (2010). Neural correlates of interspecies perspective taking in the post-mortem Atlantic salmon. *Journal of Serendipitous and Unexpected Results* 1(1), 1–5.
