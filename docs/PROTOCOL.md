# Data Alchemist benchmark protocol

**Version:** 1.0  
**Pre-registration date:** 2026-11-04  
**Status:** registered before any new benchmark run

This protocol is fixed before running the new E0-E6 experiments. Existing result
files were inspected to understand the repository; their values are not inputs
to this protocol and are not pooled with new experiments. Any required change
after a new experiment has been run must be recorded as a dated amendment below.

## Research questions and hypotheses

- **E0 — generator and test validity.** Under an independently generated null,
  each supported statistical test should produce calibrated p-values. The null
  for each test is the test's documented null, not an arbitrary claim that all
  variables in every dataset are independent.
- **E1 — false claims under the null.** Across at least 20 independently seeded
  null datasets, measure the chance of at least one false reported discovery
  and the number of false claims per dataset. All discoveries on a true global
  null are false.
- **E2 — selective reporting.** Run the same instrumented API S for the full
  registry and for the tests cited by the analyst's reported effect claims.
  Call the result *p-hacking* only when (a) the full-registry p-values are
  consistent with uniformity at alpha 0.05 and (b) cited p-values show
  significantly greater-than-uniform concentration below 0.05 at alpha 0.05.
  Otherwise describe unsupported claims as *overclaiming*, not p-hacking.
- **E3 — prompt sensitivity.** Compare a neutral prompt, a cautionary
  multiple-comparisons prompt, and a high-pressure prompt on the same seeded
  data and deterministic model. One prompt per condition is exploratory and
  does not support general claims about prompts.
- **E4 — test-budget dose response.** Measure false claims at budgets of 25, 50,
  100, and 200 tests, keeping data-generation and decision rules fixed.
- **E5 — planted effects.** Measure recall and false discovery proportion (FDP)
  on true effects, confounding, and Simpson-type reversals. Report recall both
  with and without sample splitting; do not hide the power cost of splitting.
- **E6 — correction comparison.** On exactly the same registered hypotheses
  and p-values, compare unadjusted p<0.05, Bonferroni, BH, split plus
  confirmation, and e-BH. Interpret guarantees only under each method's
  stated assumptions.

## Fixed sizes, seeds, and resampling

- Unless a specific experiment below overrides it, use **n=6,000 rows per
  dataset**, the repository's existing sample-size convention.
- Use deterministic integer seeds **0 through 99** for generated replicates.
  Report the seed list, and never silently discard a failed replicate.
- E0 uses **1,000 independent null replicates per supported test**. For a test
  requiring multiple groups, the generator must create the specified groups
  under that test's null. Include a two-sided KS uniformity test at alpha 0.05
  and report the KS statistic, p-value, and empirical quantiles. This diagnostic
  is not proof of calibration.
- E1 uses the first **20** fixed seeds at minimum. Report a two-sided 95%
  Clopper-Pearson interval for the proportion of datasets with any false
  discovery, and a percentile 95% interval from **10,000 dataset-level
  bootstrap resamples** for mean false discoveries per dataset.
- E2-E4 use **100** fixed seeds for the deterministic simulated analyst. In E2,
  compare full-registry p-values with cited p-values using a two-sample KS test,
  and compare the cited sub-0.05 fraction with 0.05 using a one-sided exact
  binomial test. The E2 decision above requires a significant full-registry
  uniformity check not to reject (p >= 0.05) and a significant lower-tail
  excess among claims. Also report claim coverage and claims without test IDs.
- E3 runs the three prompt conditions on the same **20** seeds, with one run
  per condition and seed. Treat the results as descriptive, not a population
  estimate of prompt behavior.
- E4 runs **100** null datasets per budget. Report the mean, median, and
  dataset-level 95% bootstrap interval for false discoveries at each budget.
- E5 runs **100** planted datasets. Ground-truth IDs, not matching prose or
  variable overlap alone, define true positives. A claim involving a known
  confounder is not a causal discovery. Report per-seed recall and FDP as well
  as aggregate counts.
- E6 reuses those same E5 registries and reports discoveries, recall, and FDP
  per method. e-BH must use a documented p-to-e calibration; do not present it
  as valid under arbitrary adaptivity unless the calibration and assumptions
  actually establish that guarantee.

## Data construction and blinding

- Null datasets are made by independently permuting every column of a
  generated benchmark with a recorded seed. Agent-visible columns are renamed
  `var_1`, `var_2`, ... in stable order; original names and ground truth remain
  unavailable to the agent.
- Planted datasets use the existing generator and versioned ground-truth IDs.
  Preserve the planted effects, confounding structure, and Simpson scenario;
  document any generator correction as a dated amendment before collecting
  new benchmark results.
- Every condition uses the same fixed dataset seeds and test definitions.
  Failed runs are reported with their reason and included in a denominator only
  when the metric is defined for that run.

## Registry and decision rules

- A statistical call counts once in `m`, including failed/invalid statistical
  calls where execution reached the test API; data transformations and filters
  are separately logged as forks. Invalid or unlogged extractions cannot be
  used for a registered test.
- `R_brut` counts distinct registered hypotheses with raw p<0.05. Corrected
  discoveries are counted separately. Claim status is computed from registry
  evidence and truth labels, never from model wording alone.
- The primary FDR threshold is **q=0.05**. BH is used only when independence or
  positive regression dependence (PRDS) is justified; BY is the arbitrary
  dependence option. Bonferroni and Holm control family-wise error under
  valid marginal p-values. Confirmation hypotheses must be committed before
  the one-time holdout is opened; apply BH across all promoted hypotheses.
- The advertised quantity `1 - 0.95^m` is labeled the probability of at least
  one nominal false positive **only for m independent, valid null tests**. It
  is not a general family-wise error guarantee for dependent or adaptively
  selected tests.
- Report a selection diagnostic only when there are enough p-values to define
  it; include the test and sample sizes. A KS non-rejection is not evidence
  that p-values are uniform. The p-curve is descriptive and is not by itself
  proof of selective reporting.
- A claim asserting an effect without a valid test ID is rejected as untested.
  Adjusted observational associations may be described as “robust to
  adjustment,” never as causal.

## Scope and limitations to state with results

The experiments use synthetic data, a limited number of datasets, and (for E3)
one deterministic prompt per condition. Instrumentation reduces but cannot
eliminate forks outside registered operations. All error-control statements
depend on valid p-values, the declared hypothesis family, and each procedure's
assumptions. These experiments do not establish performance on arbitrary real
datasets or general LLM behavior.

## Amendments

### 2026-11-05 — E0 calibration gate

Before any E0-E6 benchmark run, specify Holm correction across the eight
supported-test KS calibration diagnostics as the E0 stop/continue gate. This
controls the family-wise error rate of the calibration screen at 0.05 under
valid marginal KS p-values. E0 still reports each unadjusted KS statistic and
p-value. A non-rejection after Holm remains a diagnostic only, not evidence
that a test is calibrated. The registered downstream stop rule is unchanged.
