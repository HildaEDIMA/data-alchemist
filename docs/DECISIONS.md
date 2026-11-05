# Design decisions

## 2026-11-05 — Conservative exploratory correction

Live exploratory claims use BY rather than BH by default. Pairwise tests
commonly share variables, and the required independence/PRDS conditions are
not established automatically by the registry. This trades power for a
defensible arbitrary-dependence FDR procedure under valid marginal p-values.
Confirmation remains BH, with its assumptions stated in the report.

## 2026-11-05 — Holdout is confirmation, not reusable holdout

The implementation permits one confirmation opening per run. It does not
implement differential privacy or the reusable-holdout construction, so it
does not claim the guarantees of Dwork et al. Repeated holdout use fails
closed.

## 2026-11-05 — Historical archive is not retroactively audited

The legacy archive lacks a complete test registry. The demo displays its
stored claims and provenance but does not infer a historical test count,
recreate p-values, or present the current engine's output as an evaluation of
the old results.

## 2026-11-05 — E0 diagnostic multiplicity

E0's eight KS calibration diagnostics are screened using Holm at family-wise
alpha 0.05 before downstream benchmark interpretation. This addition was
recorded in the protocol amendment before running E0-E6. Raw KS diagnostics
remain visible, and non-rejection is not interpreted as proof of calibration.

## 2026-11-05 — Reproducible offline demonstration

The current-engine demo generates its fixed synthetic dataset in process
instead of reading an ignored local CSV. This makes the demo reproducible from
a clean checkout and avoids treating generated working files as source data.

## 2026-11-05 — Synthetic null construction

Benchmark nulls independently permute all columns from the planted generator,
preserving each column's marginal distribution while severing the encoded
cross-column relationships. Agent-visible names are neutral and stable.
This follows the registered blinding rule; it does not model every possible
real-world null.

## 2026-11-05 — Test-time extraction integrity

The registry stores the full extraction hash and per-column hashes. Each
statistical call verifies the exact columns it consumes, while filters and
recodings verify the full extraction because their transformations may depend
on any column. This preserves test-input integrity without re-hashing every
unrelated column for every pairwise benchmark test.
