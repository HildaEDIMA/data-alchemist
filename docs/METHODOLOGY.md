# Methodology

## Research target

Data Alchemist addresses a reporting failure in automated data exploration:
the final claim often hides the set of tests, transformations, and choices that
preceded it. The system's primary contribution is auditability and an explicit
exploration-to-confirmation boundary. It does not make exploratory p-values
valid merely by logging them.

## Evidence lifecycle

### Admission and registry

A dataset must be registered before it can be extracted for inference. Tests
accept only registry-owned tracked extractions. Extraction hashes and
transformation records make data provenance inspectable; the registry records
the procedure, variables, direction, sample size, test statistic, raw
p-value, effect size, null hypothesis, and execution error for each call.
Failed statistical calls reached through API S remain in `m`.

### Exploration and multiplicity

The analyst can request tests only through API S. In the live API, the
exploration family consists of the tests performed by that run. BY is the
default because pairwise discovery tests commonly share variables and their
dependence structure is not generally established. Its FDR statement relies
on valid marginal p-values and a well-defined family; logging does not repair
invalid test assumptions or unregistered external analyses.

The report separately counts raw `p < 0.05` results and tests retained after
the selected correction. `R_brut` counts distinct test/extraction hypotheses
with raw `p < 0.05`; repeated calls still count separately in `m` and in the
multiplicity family. Claims cite registry IDs, and variable coverage is
checked. A claim with no test ID is rejected as untested. Failure to reject
does not establish the null; proving practical equivalence requires an
equivalence test and a predeclared margin.

### Confirmation

Promoted hypotheses are fixed in the registry before the confirmation
extraction is opened. The holdout can be opened once, and BH is applied across
the full promoted family. Confirmation p-values are meaningful only if
the hypotheses were fixed without using holdout information, the tests are
valid for the holdout data, and BH's independence or PRDS assumption holds.
The UI and report do not describe an observational association as causal.
Directional group comparisons use a stable category-level ordering so the
meaning of `greater` or `less` does not change when rows are split or shuffled.

### e-values

The benchmark uses the calibration `e = gamma * p**(gamma - 1)` with
`gamma = 0.5`. For a valid super-uniform null p-value, this decreasing
calibration has null expectation at most one. e-BH controls FDR for a fixed
family of valid e-values under arbitrary dependence. This does not itself
authorize adaptive reuse or selection of the data used to construct the
p-values; no such blanket guarantee is claimed by this implementation.

## Selection diagnostics

Reports include a KS uniformity diagnostic for the full registered
p-values, a two-sample KS comparison of registered and cited p-values when
both samples are available, and descriptive p-curve bins. Each diagnostic
includes its sample size. A KS non-rejection is not proof of uniformity.
Because p-values within a dataset can be dependent and cited values are a
subset of registry values, the KS p-values are diagnostics rather than
unqualified inferential evidence of selective reporting.

## Benchmark design

The registered plan and exact stopping rules live in
[PROTOCOL.md](PROTOCOL.md). E0 checks p-value behavior for the supported
procedures under independently generated nulls and blocks interpretation of
downstream experiments if the Holm-adjusted calibration screen rejects any
procedure. E1 estimates false discovery behavior; E2 applies the registered
two-part rule before using the term *p-hacking*; E3 is a paired descriptive
prompt comparison; E4 measures false-claim counts over increasing budgets;
E5 compares planted-effect recall and false discovery proportion with and
without a split; E6 compares decision procedures on the same deterministic
test suite.

The null generator independently permutes every source column and renames
columns in stable order before agent-visible analysis. Ground-truth IDs—not
claim wording—score planted effects. Failed replicates are recorded rather
than silently dropped. Bootstrap resampling is at the dataset level, not the
individual-test level.

The registered protocol is frozen before E0-E6 runs. Changes required after
results are observed must be recorded as dated amendments, and the original
results remain distinguishable from amended follow-up experiments.

## Interpretation limits

1. The data are synthetic, and the number of datasets is finite.
2. E3 uses one deterministic policy per prompt condition; it is not an
   estimate of general LLM prompt sensitivity.
3. Logging reduces invisible forking paths but cannot cover analysis performed
   outside the instrumented interface.
4. Validity depends on test-specific assumptions, model specification,
   missingness behavior, and the declared hypothesis family.
5. A selected association—even one robust to measured adjustment—is not a
   causal effect.
6. Reported risk `1 - 0.95^m` is only the probability of at least one nominal
   false positive when all `m` tests are independent and valid null tests.

## Relationship to POPPER

POPPER is the closest related agentic approach identified for this project.
It validates a specified hypothesis using sequential falsification. Data
Alchemist instead audits discovery across a test budget and separates
exploration from confirmation. The systems address related but distinct
stages; this comparison does not imply that one subsumes the other.
