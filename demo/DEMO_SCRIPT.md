# Three-minute demo script

## Before the room

Run `make demo`, open `http://localhost:8080`, and keep the **Run en direct**
view selected. This path uses no external model key. The historical archive
and current-engine replay are distinct demonstrations; do not describe the
archive as a run of the current engine.

## Talk track

### 0:00–0:35 — The blind spot

“An AI analyst can test dozens of relationships and show only the most
interesting p-value. Without the search history, we cannot tell whether that
finding is evidence or selection. Data Alchemist makes the search itself part
of the result.”

Point to the run counter. In the archive view, explain that the stored legacy
results contain claims but not the complete test registry. Do not invent an
archive test count or a historical p-value distribution.

### 0:35–1:15 — Replay the archived failure case

In **Rapport d’audit**, show the archived claims and the provenance note. Say:
“These are historical outcomes and truth labels, not a fresh benchmark. The
archive is useful as a motivating example; it cannot tell us the old `m`.”
If quoting any archive totals, read them directly from the displayed metrics.

### 1:15–2:10 — Run the instrumented product

Return to **Run en direct** and click **Rejouer avec le moteur actuel**. The
offline deterministic analyst runs on a fixed, generated dataset. Point out
that the counter is populated from registered tests, not estimated from the
number of claims. The displayed `1 − 0.95^m` is explicitly an independence
illustration, not a general error guarantee.

### 2:10–2:40 — Inspect the audit trail

Open **Rapport d’audit** and show each claim's evidence and disposition. Then
open **Sélection**: the live run has registered p-values; the old archive does
not, so the interface refuses to fabricate a comparison. Mention that
exploration uses BY by default and promoted claims need pre-registered
holdout confirmation.

### 2:40–3:00 — Close with the claim boundary

“We do not claim that logging makes every p-value valid, or that a robust
association is causal. We make the hidden search visible, state the
assumptions, and keep exploratory discovery separate from confirmation.
E0-E6 results are shown only after the pre-registered calibration gate and
benchmark runs are actually completed.”
