# Data Alchemist

**The AI proposes. Statistics keep the receipts.**

Data Alchemist is an auditable statistical-discovery layer for AI data analysts.
It makes the search visible: every registered test is counted, claims must cite
their evidence, exploratory findings face multiplicity correction, and promoted
hypotheses are checked on a one-time holdout.

The product includes a pure Python core, a FastAPI service, a React interface,
and a reproducible synthetic benchmark. The deterministic demo runs offline;
external model credentials are optional.

## Why this exists

An analyst who tries enough relationships will eventually find apparently
significant results in noise. If only the selected result is shown, users cannot
see how many attempts preceded it. Data Alchemist makes that hidden search
auditable rather than treating an isolated p-value as proof.

## Product flow

1. **Register** a dataset and each extraction or transformation used.
2. **Explore** through API S; an agent requests tests but cannot execute
   generated Python or directly obtain an unregistered p-value.
3. **Record** the test, variables, sample hash, code, null, statistic,
   p-value, effect size, sample size, and test direction.
4. **Correct** the full exploratory family using BY by default in live runs.
5. **Promote** claims to explicit hypotheses before opening a single-use
   confirmation split; confirmation uses BH across the promoted family.
6. **Report** from the registry, including raw discoveries, confirmation
   status, rejected claims, reasons, and selection diagnostics.

An adjusted observational association is described as “robust to adjustment,”
not as causal. A non-significant result is not described as proof of no effect.
The benchmark's e-BH comparison uses the documented calibration
`e = 0.5 * p**(-0.5)` and only claims validity for a fixed family of valid
null p-values under arbitrary dependence; this is not a blanket guarantee for
adaptive exploration.

## Run locally

### One-command product demo

Requirements: Docker Compose and GNU Make (or run the equivalent Docker
commands on Windows). From the repository root:

```sh
make demo
```

Open [http://localhost:8080](http://localhost:8080). The API documentation is
at [http://localhost:8000/docs](http://localhost:8000/docs). The demo includes
the historical, pre-audit archive and a separate replay through the current
offline engine. The archive is clearly labeled: it does not contain the
original complete test registry, so no historical `m` or p-value distribution
is reconstructed from it.

Stop the containers with `docker compose down`.

### Development

Python 3.11 or newer and Node.js 20 or newer:

```sh
python -m pip install -e ".[api,agents,benchmark,dev]"
npm --prefix web ci
```

In separate terminals:

```sh
python -m uvicorn api.main:app --reload
npm --prefix web run dev
```

The web development server uses the API at `http://localhost:8000` by default.
Set `VITE_API_URL` before building if the API is hosted elsewhere.

## Verification and benchmarks

```sh
make test       # pytest + core coverage threshold
make lint       # Ruff and strict mypy
make web-build  # TypeScript check and production Vite build
make bench      # E0 first; stop if its pre-registered gate fails
make demo-script
```

The benchmark CLI supports selecting a subset and reducing sizes for
development smoke checks, but only the registered sample sizes and seeds are
confirmatory:

```sh
python -m benchmark.run --experiments E0
python -m benchmark.run --experiments E1,E5,E6 --n 6000
```

Full registered outputs are written to `results/e0.json` through
`results/e6.json`, with figures under `benchmark/figures/`. Each JSON includes
its seeds, metrics, and a result paragraph generated from that run's values.
No new E0-E6 result is claimed until the corresponding files exist and have
passed validation.

## Architecture

| Path | Responsibility |
|---|---|
| `core/` | Registry, test API, claims, multiplicity, split, and report; no web dependency |
| `api/` | FastAPI run lifecycle, registry/report endpoints, SSE, benchmark and demo data |
| `web/` | React/Vite user experience for live runs, audit reports, selection, and results |
| `benchmark/` | E0-E6 runner, generated figures, and versioned result output |
| `docs/PROTOCOL.md` | Pre-registered hypotheses, sample sizes, seeds, and decision rules |
| `docs/METHODOLOGY.md` | Statistical interpretation, assumptions, and limitations |
| `demo/` | Offline demo script and three-minute presentation track |

## API at a glance

- `POST /runs` — queue an analysis of a supplied dataset.
- `GET /runs/{id}` — retrieve state and current registered-test count.
- `GET /runs/{id}/events` — stream status updates using server-sent events.
- `GET /runs/{id}/registry` and `/report` — inspect audit evidence and findings.
- `POST /demo/run` — generate a fixed synthetic dataset and run the offline engine.
- `GET /demo` — load archived demo material with explicit provenance.
- `GET /benchmark/{experiment}` — read a versioned E0-E6 JSON result when available.

The current API keeps run state in process memory; it is intended for a local
demo and evaluation, not multi-worker or durable production deployment.
The configured test budget caps agent-requested exploration; system-run
confirmation tests are additionally counted in the registry's `m`.

## Statistical scope

- **BY** is the default exploratory FDR procedure because it handles arbitrary
  dependence, assuming valid marginal p-values and a declared family.
- **BH** is used for confirmation only with the usual independence or PRDS
  assumptions and valid holdout p-values; the report states these assumptions.
- **Bonferroni/Holm** control FWER under valid marginal p-values.
- **e-BH** is benchmarked for a fixed family using a p-to-e calibration; p-to-e
  conversion alone does not make adaptively selected p-values valid.
- `1 - 0.95^m` is shown only as an independence-based illustration, never as a
  general guarantee for correlated or adaptively selected tests.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and
[docs/PROTOCOL.md](docs/PROTOCOL.md) for the full definitions and stopping
rules. Reference metadata and verification status are in
[docs/REFERENCES.md](docs/REFERENCES.md) and [docs/references.bib](docs/references.bib).

## Limitations

The benchmark uses synthetic data and a bounded number of deterministic
replicates. Prompt comparisons are descriptive, not evidence about the
population behavior of language models. Instrumentation makes registered
forks visible but cannot eliminate uninstrumented exploration outside the
system. Every error-control statement depends on valid p-values, a correctly
declared test family, and the assumptions named in the report. Results on
synthetic data do not establish performance on arbitrary real datasets.

## Closest related system

POPPER (Huang et al.) is a closely related agentic system for **validating a
given hypothesis** through sequential falsification. Data Alchemist focuses on
the complementary problem of **auditing discovery**: it exposes the search
budget and controls how exploratory findings become confirmatory claims.
POPPER's arXiv record is cited in the references; its claimed conference
publication status is not asserted here.
