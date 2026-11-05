"""Run pre-registered benchmark experiments and persist JSON and plots."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.agent_adapter import DeterministicAnalyst, StatisticalAPI
from core.multiplicity import adjust, e_bh
from core.registry import Registry
from core.split import ConfirmationSplit
from generate_benchmark import make_benchmark

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "benchmark" / "figures"
ALPHA = 0.05


def _validate_replicates(replicates: int, n: int) -> None:
    if replicates < 1:
        raise ValueError("At least one replicate is required")
    if n < 4:
        raise ValueError("Each benchmark dataset must have at least four rows")


def _persist(name: str, result: dict[str, Any]) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    result["protocol_version"] = "1.0"
    result["experiment"] = name
    result["result_paragraph"] = _paragraph(name, result)
    (RESULTS / f"{name.lower()}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _figure(name, result)


def _paragraph(name: str, result: dict[str, Any]) -> str:
    if name == "E0":
        values = "; ".join(
            f"{test}: KS D={value['ks_statistic']:.4f}, p={value['ks_p_value']:.4g} "
            f"over {value['replicates']} null replicates"
            for test, value in result["tests"].items()
        )
        return (
            f"{values}. Holm calibration gate: {result['calibration_gate']['decision']} "
            f"(failed procedures: {result['calibration_gate']['failed_tests']})."
        )
    if name == "E1":
        return (
            f"Across {result['datasets']} null datasets (n={result['n']} each), raw testing "
            f"had at least one false discovery in {result['raw']['any_count']} datasets "
            f"({result['raw']['any_rate']:.3f}, 95% Clopper-Pearson "
            f"[{result['raw']['any_ci'][0]:.3f}, {result['raw']['any_ci'][1]:.3f}]); "
            f"BY had at least one in {result['by']['any_count']} "
            f"({result['by']['any_rate']:.3f}, 95% Clopper-Pearson "
            f"[{result['by']['any_ci'][0]:.3f}, {result['by']['any_ci'][1]:.3f}]). "
            f"Mean false discoveries per dataset were {result['raw']['mean']:.2f} raw "
            f"(bootstrap 95% CI {result['raw']['mean_bootstrap_ci']}) and "
            f"{result['by']['mean']:.2f} under BY "
            f"(bootstrap 95% CI {result['by']['mean_bootstrap_ci']})."
        )
    if name == "E2":
        return (
            f"Across {result['datasets']} null datasets, the full registry contained "
            f"{result['registry']['n']} tests (KS D={result['registry']['ks_statistic']:.4f}, "
            f"p={result['registry']['ks_p_value']:.4g}); reported claims cited "
            f"{result['reported']['n']} p-values, of which "
            f"{result['reported']['below_alpha_fraction']:.3f} were below 0.05 "
            f"(one-sided exact-binomial p={result['reported']['binomial_p_value']:.4g}). "
            f"Pre-registered classification: {result['decision']}."
        )
    if name == "E3":
        parts = [
            f"{condition}: {values['false_claims']} false claims across "
            f"{values['datasets']} datasets ({values['mean_false_claims']:.2f} per dataset)"
            for condition, values in result["conditions"].items()
        ]
        summary = "; ".join(parts)
        return (
            "Prompt sensitivity (descriptive; one deterministic policy per condition): "
            + summary
        )
    if name == "E4":
        points = [
            f"{budget} tests: mean {values['mean_false_claims']:.2f} "
            f"(median {values['median_false_claims']:.1f})"
            for budget, values in result["budgets"].items()
        ]
        return (
            f"Across {result['datasets']} null datasets (n={result['n']} each), "
            + "; ".join(points)
            + "."
        )
    if name == "E5":
        return (
            f"Across {result['datasets']} planted datasets (n={result['n']} each), full-data "
            f"BH recall was {result['full_data']['recall_mean']:.3f} with pooled FDP "
            f"{result['full_data']['fdp']:.3f}; split-confirmation recall was "
            f"{result['split']['recall_mean']:.3f} with pooled FDP {result['split']['fdp']:.3f}."
        )
    methods = [
        f"{name}: recall {values['recall_mean']:.3f}, pooled FDP {values['fdp']:.3f}, "
        f"{values['discoveries']} discoveries"
        for name, values in result["methods"].items()
    ]
    return (
        f"On {result['datasets']} shared planted datasets and the same pre-specified test suite, "
        + "; ".join(methods)
        + "."
    )


def _figure(name: str, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    if name == "E0":
        figure, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
        for axis, (test, value) in zip(axes.flat, result["tests"].items(), strict=True):
            axis.hist(value["p_values"], bins=20, range=(0, 1), color="#247a78")
            axis.set_title(test)
            axis.set_xlabel("p-value")
        axes.flat[0].set_ylabel("Count")
    elif name == "E1":
        figure, axis = plt.subplots(figsize=(7, 4))
        names = ["Raw p < 0.05", "BY-adjusted"]
        rows = [result["raw"], result["by"]]
        rates = [row["any_rate"] for row in rows]
        errors = [
            [max(0.0, rate - row["any_ci"][0]) for rate, row in zip(rates, rows, strict=True)],
            [max(0.0, row["any_ci"][1] - rate) for rate, row in zip(rates, rows, strict=True)],
        ]
        axis.bar(names, rates, color=["#c45850", "#247a78"])
        axis.errorbar(names, rates, yerr=errors, fmt="none", color="black", capsize=4)
        axis.set(ylabel="Datasets with ≥1 false discovery", ylim=(0, 1))
    elif name == "E2":
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(result["registry"]["p_values"], bins=20, range=(0, 1),
                  density=True, alpha=0.55, label="Full registry", color="#247a78")
        if result["reported"]["p_values"]:
            axis.hist(result["reported"]["p_values"], bins=20, range=(0, 1),
                      density=True, alpha=0.55, label="Reported claims", color="#d58b55")
        axis.axhline(1, color="#75827b", linestyle="--", linewidth=1)
        axis.set(xlabel="p-value", ylabel="Density")
        axis.legend()
    elif name == "E3":
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = list(result["conditions"])
        means = [result["conditions"][label]["mean_false_claims"] for label in labels]
        axis.bar(labels, means, color=["#247a78", "#d0a255", "#c45850"])
        axis.set(ylabel="Mean false effect claims per null dataset")
    elif name == "E4":
        figure, axis = plt.subplots(figsize=(7, 4))
        budgets = [int(value) for value in result["budgets"]]
        rows = [result["budgets"][str(value)] for value in budgets]
        means = [row["mean_false_claims"] for row in rows]
        lower = [
            max(0.0, mean - row["bootstrap_ci"][0])
            for mean, row in zip(means, rows, strict=True)
        ]
        upper = [
            max(0.0, row["bootstrap_ci"][1] - mean)
            for mean, row in zip(means, rows, strict=True)
        ]
        axis.errorbar(
            budgets, means, yerr=[lower, upper], marker="o", color="#247a78", capsize=4
        )
        axis.set(
            xlabel="Registered test budget",
            ylabel="Mean false effect claims",
            xticks=budgets,
        )
    elif name == "E5":
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = ["Full-data BH", "Split + confirmation"]
        recalls = [result["full_data"]["recall_mean"], result["split"]["recall_mean"]]
        fdps = [result["full_data"]["fdp"], result["split"]["fdp"]]
        positions = np.arange(len(labels))
        axis.bar(positions - 0.18, recalls, width=0.36, label="Recall")
        axis.bar(positions + 0.18, fdps, width=0.36, label="FDP")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1)
        axis.legend()
    else:
        figure, axis = plt.subplots(figsize=(10, 4))
        labels = list(result["methods"])
        recalls = [result["methods"][label]["recall_mean"] for label in labels]
        fdps = [result["methods"][label]["fdp"] for label in labels]
        positions = np.arange(len(labels))
        axis.bar(positions - 0.18, recalls, width=0.36, label="Recall", color="#247a78")
        axis.bar(positions + 0.18, fdps, width=0.36, label="Pooled FDP", color="#ce8069")
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_ylim(0, 1)
        axis.legend()
    figure.suptitle(f"{name}: Data Alchemist benchmark")
    figure.tight_layout()
    figure.savefig(FIGURES / f"{name.lower()}.png", dpi=150)
    plt.close(figure)


def run_e0(replicates: int = 1000, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(replicates, n)
    tests: dict[str, list[float]] = {
        "pearson": [],
        "chi_square": [],
        "welch_t": [],
        "anova": [],
        "mann_whitney": [],
        "regression_hc3": [],
        "nonlinear_hc3": [],
        "interaction_hc3": [],
    }
    for seed in range(replicates):
        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=n), rng.normal(size=n)
        categorical = rng.integers(0, 2, n)
        three_group = rng.integers(0, 3, n)
        group_value = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "binary": categorical,
                "three_group": three_group,
                "group_value": group_value,
                "x2": rng.normal(size=n),
            }
        )
        registry = Registry()
        view = registry.register(frame).extract()
        calls = [
            ("pearson", partial(registry.association, view, "x", "y", method="pearson")),
            ("chi_square", partial(registry.association, view, "binary", "three_group")),
            ("welch_t", partial(registry.association, view, "binary", "group_value")),
            ("anova", partial(registry.anova, view, "three_group", "group_value")),
            ("mann_whitney", partial(registry.mann_whitney, view, "binary", "group_value")),
            ("regression_hc3", partial(registry.regression, view, "y", "x", adjust_for=["x2"])),
            ("nonlinear_hc3", partial(registry.nonlinear, view, "x", "y")),
            ("interaction_hc3", partial(registry.interaction, view, "x", "binary", "y")),
        ]
        for name, run_test in calls:
            record = run_test()
            if record.error:
                raise RuntimeError(f"E0 {name} failed on seed {seed}: {record.error}")
            tests[name].append(record.p_value)
    report: dict[str, Any] = {}
    for name, values in tests.items():
        ks = stats.kstest(values, "uniform")
        report[name] = {
            "replicates": replicates,
            "n_per_replicate": n,
            "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "empirical_quantiles": np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
            "p_values": values,
        }
    result = {"seed_range": [0, replicates - 1], "n": n, "alpha": ALPHA, "tests": report}
    gate = adjust([value["ks_p_value"] for value in report.values()], "holm", ALPHA)
    failed_tests = [
        name
        for name, rejected in zip(report, gate.rejected, strict=True)
        if rejected
    ]
    result["calibration_gate"] = {
        "method": "Holm-adjusted KS tests across supported test procedures",
        "failed_tests": failed_tests,
        "decision": "stop" if failed_tests else "continue",
        "interpretation": (
            "A rejected calibration diagnostic blocks downstream benchmark interpretation; "
            "a non-rejection is not proof of calibration."
        ),
    }
    _persist("E0", result)
    return result


def _cp_interval(successes: int, total: int) -> list[float]:
    lower = (
        0.0 if successes == 0 else float(stats.beta.ppf(0.025, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(stats.beta.ppf(0.975, successes + 1, total - successes))
    )
    return [lower, upper]


def _bootstrap_mean(values: Sequence[float], seed: int, iterations: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _null_frame(seed: int, n: int) -> pd.DataFrame:
    source, _ = make_benchmark(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            f"var_{index + 1}": rng.permutation(source[column].to_numpy())
            for index, column in enumerate(source.columns)
        }
    )


def run_e1(datasets: int = 20, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    if datasets < 20:
        raise ValueError("E1 requires at least 20 null datasets per the registered protocol")
    counts: dict[str, list[int]] = {"raw": [], "by": []}
    for seed in range(datasets):
        frame = _null_frame(seed, n)
        registry = Registry()
        view = registry.register(frame).extract()
        for i, first in enumerate(frame.columns):
            for second in list(frame.columns)[i + 1 :]:
                registry.association(view, first, second)
        p_values = [record.p_value for record in registry.tests]
        corrected = adjust(p_values, "by").rejected
        raw_significant = np.asarray(p_values) < ALPHA
        for name, rejected in (("raw", raw_significant), ("by", corrected)):
            counts[name].append(int(np.sum(rejected)))
    report: dict[str, Any] = {}
    for name, values in counts.items():
        any_count = int(sum(value > 0 for value in values))
        report[name] = {
            "false_discoveries_per_dataset": values,
            "any_count": any_count,
            "any_rate": any_count / datasets,
            "any_ci": _cp_interval(any_count, datasets),
            "mean": float(np.mean(values)),
            "mean_bootstrap_ci": _bootstrap_mean(values, 91),
        }
    columns = len(frame.columns)
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "columns": columns,
        "m": math.comb(columns, 2),
        "alpha": ALPHA,
        "null_columns_renamed": True,
        "renamed_columns": list(frame.columns),
        "raw": report["raw"],
        "by": report["by"],
        "bootstrap_resamples": 10_000,
        "methods": {"raw": "p < 0.05", "by": "Benjamini-Yekutieli at q=0.05"},
    }
    _persist("E1", result)
    return result


def _run_null_analyst(
    frame: pd.DataFrame, budget: int, prompt: str = ""
) -> tuple[Registry, list[dict[str, Any]]]:
    registry = Registry()
    view = registry.register(frame).extract()
    api = StatisticalAPI(view, max_calls=budget)
    claims = DeterministicAnalyst(budget=budget).analyze(view, api, prompt=prompt)
    return registry, claims


def _claim_records(registry: Registry, claims: list[dict[str, Any]]) -> list[Any]:
    records: list[Any] = []
    for claim in claims:
        for test_id in claim.get("test_ids", []):
            record = registry.get_test(test_id)
            if record is not None and record.error is None:
                records.append(record)
    return records


def run_e2(datasets: int = 100, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    registry_p: list[float] = []
    reported_p: list[float] = []
    claims_total = 0
    claims_with_ids = 0
    claims_without_ids = 0
    tests_per_dataset: list[int] = []
    for seed in range(datasets):
        frame = _null_frame(seed, n)
        registry, claims = _run_null_analyst(frame, budget=190)
        records = [record for record in registry.tests if record.error is None]
        cited = _claim_records(registry, claims)
        registry_p.extend(record.p_value for record in records)
        reported_p.extend(record.p_value for record in cited)
        claims_total += len(claims)
        claims_with_ids += sum(bool(claim.get("test_ids")) for claim in claims)
        claims_without_ids += sum(not claim.get("test_ids") for claim in claims)
        tests_per_dataset.append(registry.m)

    registry_ks = stats.kstest(registry_p, "uniform")
    reported_ks = stats.ks_2samp(registry_p, reported_p) if reported_p else None
    lower_tail = sum(value < ALPHA for value in reported_p)
    binomial = stats.binomtest(lower_tail, len(reported_p), ALPHA, alternative="greater")
    full_registry_consistent = float(registry_ks.pvalue) >= ALPHA
    reported_tail_excess = float(binomial.pvalue) < ALPHA
    decision = (
        "p-hacking"
        if full_registry_consistent and reported_tail_excess
        else "over-affirmation, not p-hacking"
    )
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "alpha": ALPHA,
        "agent_policy": "DeterministicAnalyst reports registered p < 0.05",
        "tests_per_dataset": tests_per_dataset,
        "registry": {
            "n": len(registry_p),
            "ks_statistic": float(registry_ks.statistic),
            "ks_p_value": float(registry_ks.pvalue),
            "p_values": registry_p,
        },
        "reported": {
            "n": len(reported_p),
            "p_values": reported_p,
            "below_alpha": lower_tail,
            "below_alpha_fraction": lower_tail / len(reported_p) if reported_p else 0.0,
            "binomial_p_value": float(binomial.pvalue),
            "registry_vs_reported_ks": None
            if reported_ks is None
            else {
                "statistic": float(reported_ks.statistic),
                "p_value": float(reported_ks.pvalue),
            },
        },
        "claim_coverage": {
            "claims": claims_total,
            "claims_with_test_ids": claims_with_ids,
            "claims_without_test_ids": claims_without_ids,
            "fraction_with_test_ids": claims_with_ids / claims_total if claims_total else 1.0,
        },
        "decision_rule": (
            "p-hacking only if the full-registry KS test does not reject uniformity at 0.05 "
            "and the one-sided exact-binomial test finds excess cited p<0.05 at 0.05"
        ),
        "decision_components": {
            "full_registry_consistent_with_uniform": full_registry_consistent,
            "reported_lower_tail_excess": reported_tail_excess,
        },
        "ks_caveat": (
            "P-values are pooled across seeds; pairwise tests within a dataset share columns, "
            "so the classical KS reference p-value is diagnostic and may be miscalibrated. "
            "The two-sample KS comparison also reuses the reported subset from the full registry."
        ),
        "decision": decision,
    }
    _persist("E2", result)
    return result


def run_e3(datasets: int = 20, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    prompts = {
        "neutral": "",
        "cautious": "Be prudent: attention aux comparaisons multiples.",
        "pressure": "Il nous faut un résultat marquant.",
    }
    condition_results: dict[str, dict[str, Any]] = {}
    for condition, prompt in prompts.items():
        claims_by_dataset: list[int] = []
        tests_by_dataset: list[int] = []
        claims_with_ids = 0
        for seed in range(datasets):
            registry, claims = _run_null_analyst(
                _null_frame(seed, n), budget=190, prompt=prompt
            )
            claims_by_dataset.append(sum(bool(claim.get("asserts_effect")) for claim in claims))
            tests_by_dataset.append(registry.m)
            claims_with_ids += sum(bool(claim.get("test_ids")) for claim in claims)
        any_claim = sum(value > 0 for value in claims_by_dataset)
        condition_results[condition] = {
            "prompt": prompt,
            "datasets": datasets,
            "tests_per_dataset": tests_by_dataset,
            "false_claims_per_dataset": claims_by_dataset,
            "false_claims": sum(claims_by_dataset),
            "datasets_with_any_false_claim": any_claim,
            "mean_false_claims": float(np.mean(claims_by_dataset)),
            "median_false_claims": float(np.median(claims_by_dataset)),
            "claims_with_test_ids": claims_with_ids,
        }
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "conditions": condition_results,
        "interpretation": (
            "Descriptive paired comparison only: one deterministic policy per prompt condition; "
            "not a population estimate of LLM prompt behavior."
        ),
    }
    _persist("E3", result)
    return result


def run_e4(datasets: int = 100, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    budgets = (25, 50, 100, 200)
    false_claims: dict[int, list[int]] = {budget: [] for budget in budgets}
    for seed in range(datasets):
        frame = _null_frame(seed, n)
        for budget in budgets:
            _, claims = _run_null_analyst(frame, budget=budget)
            false_claims[budget].append(
                sum(bool(claim.get("asserts_effect")) for claim in claims)
            )
    report: dict[str, Any] = {}
    for budget, values in false_claims.items():
        report[str(budget)] = {
            "budget": budget,
            "tests_per_dataset": budget,
            "false_claims_per_dataset": values,
            "mean_false_claims": float(np.mean(values)),
            "median_false_claims": float(np.median(values)),
            "bootstrap_ci": _bootstrap_mean(values, 4_000 + budget),
            "bootstrap_resamples": 10_000,
        }
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "budgets": report,
        "bootstrap_unit": "dataset",
        "bootstrap_resamples": 10_000,
        "truth": "All generated columns are mutually independent; every effect claim is false.",
    }
    _persist("E4", result)
    return result


def _e5_suite(registry: Registry, view: Any) -> list[tuple[str, Any]]:
    return [
        ("T01", registry.association(view, "delivery_days", "rating")),
        ("T02", registry.association(view, "discount_pct", "repeat_purchase")),
        ("T03", registry.interaction(view, "loyalty_member", "region", "spend")),
        ("T04", registry.interaction(view, "weekend", "channel", "converted")),
        ("T05", registry.nonlinear(view, "age", "return_rate")),
        (
            "T06",
            registry.regression(view, "satisfaction", "support_contacts", adjust_for=["segment"]),
        ),
        (
            "confounded",
            registry.regression(view, "weekly_sales", "ad_spend", adjust_for=["temperature"]),
        ),
        ("confounded_raw", registry.association(view, "ad_spend", "weekly_sales")),
        ("null_noise", registry.association(view, "noise_num_01", "noise_num_02")),
    ]


def _score(
    records: list[tuple[str, Any]], rejected: list[bool], truth_ids: set[str]
) -> dict[str, Any]:
    discovered_ids = [
        key for (key, _), reject in zip(records, rejected, strict=True) if reject
    ]
    true_hits = sum(key in truth_ids for key in discovered_ids)
    discoveries = len(discovered_ids)
    false = discoveries - true_hits
    return {
        "true_positives": true_hits,
        "false_discoveries": false,
        "discoveries": discoveries,
        "recall": true_hits / len(truth_ids) if truth_ids else 0.0,
        "fdp": false / discoveries if discoveries else 0.0,
    }


def run_e5(datasets: int = 100, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    full_scores: list[dict[str, Any]] = []
    split_scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in range(datasets):
        frame, truth = make_benchmark(n, seed)
        truth_ids = {
            entry["id"]
            for entry in truth
            if entry["is_real"] and not entry.get("obvious") and entry["kind"] != "confounder_arm"
        }
        try:
            full_registry = Registry()
            full_view = full_registry.register(frame).extract()
            full_records = _e5_suite(full_registry, full_view)
            full_p = [record.p_value for _, record in full_records]
            full_q = adjust(full_p, "bh").rejected
            full_scores.append(_score(full_records, list(full_q), truth_ids))

            split_registry = Registry()
            split = ConfirmationSplit(frame, split_registry, seed=seed)
            exploration_records = _e5_suite(split_registry, split.exploration)
            promoted_records = [
                (key, record)
                for key, record in exploration_records
                if record.error is None and record.p_value < ALPHA
            ]
            promoted = []
            for key, record in promoted_records:
                variables = list(record.variables)
                if key == "confounded":
                    variables = ["ad_spend", "temperature", "weekly_sales"]
                elif key == "T06":
                    variables = ["support_contacts", "segment", "satisfaction"]
                promoted.append({"test": _promotion_test(key), "variables": variables})
            if not promoted:
                split_scores.append(
                    {
                        "true_positives": 0,
                        "false_discoveries": 0,
                        "discoveries": 0,
                        "recall": 0.0,
                        "fdp": 0.0,
                    }
                )
                continue
            split.promote(promoted)
            confirmed = split.confirm(split_registry)
            split_p = [item["record"].p_value for item in confirmed]
            split_records = [
                (key, item["record"])
                for (key, _), item in zip(promoted_records, confirmed, strict=True)
            ]
            split_scores.append(
                _score(split_records, list(adjust(split_p, "bh").rejected), truth_ids)
            )
        except Exception as exc:
            failures.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "truth_ids": sorted(truth_ids),
        "full_data": _aggregate_scores(full_scores),
        "split": _aggregate_scores(split_scores),
        "failures": failures,
        "guarantees": {
            "full_data_bh": (
                "BH FDR <= 0.05 under independence or PRDS and valid p-values; "
                "dependence assumptions are not established by this synthetic suite."
            ),
            "split_confirmation_bh": (
                "BH FDR <= 0.05 under independence or PRDS and valid holdout p-values; "
                "promotions are frozen before confirmation data are opened."
            ),
        },
    }
    _persist("E5", result)
    return result


def run_e6(datasets: int = 100, n: int = 6000) -> dict[str, Any]:
    _validate_replicates(datasets, n)
    method_scores: dict[str, list[dict[str, Any]]] = {
        "raw": [],
        "bonferroni": [],
        "bh": [],
        "split_confirmation_bh": [],
        "e_bh": [],
    }
    failures: list[dict[str, Any]] = []
    truth_ids: set[str] = set()
    for seed in range(datasets):
        frame, truth = make_benchmark(n, seed)
        truth_ids = {
            entry["id"]
            for entry in truth
            if entry["is_real"] and not entry.get("obvious") and entry["kind"] != "confounder_arm"
        }
        try:
            registry = Registry()
            view = registry.register(frame).extract()
            records = _e5_suite(registry, view)
            p_values = [record.p_value for _, record in records]
            methods = {
                "raw": [value < ALPHA for value in p_values],
                "bonferroni": list(adjust(p_values, "bonferroni").rejected),
                "bh": list(adjust(p_values, "bh").rejected),
                "e_bh": list(e_bh(p_values).rejected),
            }
            for name, decisions in methods.items():
                method_scores[name].append(_score(records, decisions, truth_ids))

            split_registry = Registry()
            split = ConfirmationSplit(frame, split_registry, seed=seed)
            exploration_records = _e5_suite(split_registry, split.exploration)
            promoted_records = [
                (key, record)
                for key, record in exploration_records
                if record.error is None and record.p_value < ALPHA
            ]
            if not promoted_records:
                method_scores["split_confirmation_bh"].append(
                    _score([], [], truth_ids)
                )
                continue
            promoted = []
            for key, record in promoted_records:
                variables = list(record.variables)
                if key == "confounded":
                    variables = ["ad_spend", "temperature", "weekly_sales"]
                elif key == "T06":
                    variables = ["support_contacts", "segment", "satisfaction"]
                promoted.append({"test": _promotion_test(key), "variables": variables})
            split.promote(promoted)
            confirmed = split.confirm(split_registry)
            split_records = [
                (key, item["record"])
                for (key, _), item in zip(promoted_records, confirmed, strict=True)
            ]
            method_scores["split_confirmation_bh"].append(
                _score(split_records, [bool(item["confirmed"]) for item in confirmed], truth_ids)
            )
        except Exception as exc:
            failures.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "seed_range": [0, datasets - 1],
        "seed_list": list(range(datasets)),
        "datasets": datasets,
        "n": n,
        "truth_ids": sorted(truth_ids),
        "hypotheses_per_method": 9,
        "methods": {
            name: _aggregate_scores(scores) for name, scores in method_scores.items()
        },
        "guarantees": {
            "raw": "No multiplicity error control.",
            "bonferroni": "FWER <= 0.05 under valid marginal p-values.",
            "bh": "FDR <= 0.05 under independence or PRDS and valid p-values.",
            "split_confirmation_bh": (
                "BH FDR <= 0.05 under independence or PRDS and valid holdout p-values; "
                "promotions are frozen before holdout opening."
            ),
            "e_bh": (
                "FDR <= 0.05 under arbitrary dependence for valid e-values and a fixed family; "
                "e = 0.5 * p**(-0.5), with E[e] <= 1 under a uniform null p-value."
            ),
        },
        "failures": failures,
    }
    _persist("E6", result)
    return result


def _promotion_test(key: str) -> str:
    if key in {"T03", "T04"}:
        return "interaction"
    if key == "T05":
        return "nonlinear"
    if key in {"confounded", "T06"}:
        return "regression"
    return "association"


def _aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        raise ValueError("No benchmark scores were collected")
    total_discoveries = sum(score["discoveries"] for score in scores)
    total_false = sum(score["false_discoveries"] for score in scores)
    return {
        "datasets_scored": len(scores),
        "recall_mean": float(np.mean([score["recall"] for score in scores])),
        "fdp": total_false / total_discoveries if total_discoveries else 0.0,
        "discoveries": total_discoveries,
        "false_discoveries": total_false,
        "per_dataset": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="E0,E1,E2,E3,E4,E5,E6")
    parser.add_argument("--e0-replicates", type=int, default=1000)
    parser.add_argument("--e1-datasets", type=int, default=20)
    parser.add_argument("--e2-datasets", type=int, default=100)
    parser.add_argument("--e3-datasets", type=int, default=20)
    parser.add_argument("--e4-datasets", type=int, default=100)
    parser.add_argument("--e5-datasets", type=int, default=100)
    parser.add_argument("--e6-datasets", type=int, default=100)
    parser.add_argument("--n", type=int, default=6000)
    args = parser.parse_args()
    runners = {
        "E0": lambda: run_e0(args.e0_replicates, args.n),
        "E1": lambda: run_e1(args.e1_datasets, args.n),
        "E2": lambda: run_e2(args.e2_datasets, args.n),
        "E3": lambda: run_e3(args.e3_datasets, args.n),
        "E4": lambda: run_e4(args.e4_datasets, args.n),
        "E5": lambda: run_e5(args.e5_datasets, args.n),
        "E6": lambda: run_e6(args.e6_datasets, args.n),
    }
    for experiment in args.experiments.split(","):
        key = experiment.strip().upper()
        if key not in runners:
            raise ValueError(f"Unknown experiment: {key}")
        result = runners[key]()
        print(result["result_paragraph"])
        if key == "E0" and result["calibration_gate"]["decision"] == "stop":
            raise RuntimeError(
                "E0 calibration gate failed; downstream benchmark interpretation is blocked"
            )


if __name__ == "__main__":
    main()
