# Data Alchemist

**An AI that discovers insights in your data, then tries to prove itself wrong.**

Built for the [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) (Oct 26 - Nov 4, 2026).

> Status: work in progress. Sections marked *planned* are not built yet.

## The problem

Analysts only find what they think to ask. Automated tools can explore a dataset on their own, but they run into a trap: test enough hypotheses and some will look significant by pure chance. An autonomous system that cannot tell a real effect from a lucky one is worse than no system at all.

## The idea

Data Alchemist explores a dataset on its own and reports only what survives rigorous checks.

**The LLM proposes, statistics decide.**

```
Dataset
   |
   v
1. Profile        schema, distributions, missing values
   |
   v
2. Propose        Meta Muse Spark suggests testable hypotheses   (planned)
   |
   v
3. Test           every hypothesis is tested in code; the model never computes a statistic
   |
   v
4. Control        false discovery rate (Benjamini-Hochberg) over ALL hypotheses tested
   |
   v
5. Validate       survivors are re-tested on a held-out half of the data
   |
   v
6. Screen         confounders are checked and effects re-estimated after adjustment
   |
   v
7. Rank           by effect size, robustness and novelty
```

Findings are reported as associations or "robust to adjustment", never as proven causes.

## How we will prove it works (planned)

We generate a synthetic dataset where we **plant the truth ourselves**: real effects, hidden subgroup effects, pure interactions, non-linear relationships, a Simpson's paradox, a confounder, and pure noise. Then we measure how many real effects the system recovers and how many false discoveries it reports, compared with the same Meta model running **without** the statistical engine.

A second demonstration runs on real enterprise data (Olist Brazilian e-commerce dataset).

## Meta models

- **Muse Spark** (via Meta Model API): hypothesis generation *(planned)*
- **Muse Code**: generation of test code *(planned)*

## Timeline

| Step | What | Status |
|---|---|---|
| 1 | Synthetic benchmark with planted ground truth | in progress |
| 2 | Statistical engine | in progress |
| 3 | Hypothesis generation with Meta Muse Spark | planned |
| 4 | Evaluation against an LLM-only baseline | planned |
| 5 | Real-data run (Olist) | planned |
| 6 | Demo interface and video | planned |

## Author

Hildegarde Edima Biyenda, statistics and data science engineering student at ENSAE Pierre Ndiaye, Dakar.
