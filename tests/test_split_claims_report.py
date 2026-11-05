import numpy as np
import pandas as pd
import pytest

from core.claims import validate_claims
from core.registry import Registry
from core.report import build_report
from core.split import ConfirmationSplit


def sample() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(size=1000)
    return pd.DataFrame({"x": x, "y": 2 * x + rng.normal(size=1000)})


def test_confirmation_requires_preregistration_and_is_single_use() -> None:
    registry = Registry()
    split = ConfirmationSplit(sample(), registry, seed=3)
    with pytest.raises(RuntimeError, match="before opening"):
        split.open_confirmation()
    split.promote([{"test": "association", "variables": ["x", "y"], "direction": "greater"}])
    results = split.confirm(registry)
    assert results[0]["confirmed"]
    with pytest.raises(RuntimeError, match="single-use"):
        split.open_confirmation()
    with pytest.raises(RuntimeError, match="frozen"):
        split.promote([{"test": "association", "variables": ["x", "y"]}])


def test_split_applies_bh_over_promotions_and_rejects_invalid() -> None:
    registry = Registry()
    data = sample()
    data["z"] = np.random.default_rng(42).normal(size=len(data))
    split = ConfirmationSplit(data, registry)
    with pytest.raises(ValueError, match="(?i)at least one"):
        split.promote([])
    with pytest.raises(ValueError, match="two variable"):
        split.promote([{"test": "association", "variables": ["x"]}])
    split.promote(
        [
            {"test": "association", "variables": ["x", "y"]},
            {"test": "association", "variables": ["x", "z"]},
        ]
    )
    results = split.confirm(registry)
    assert len(results) == 2
    assert all("adjusted_p" in result for result in results)


def test_claims_require_ids_and_registered_variable_coverage() -> None:
    registry = Registry()
    view = registry.register(sample()).extract()
    record = registry.association(view, "x", "y")
    outcomes = validate_claims(
        [
            {"claim": "x relates to y", "variables": ["x", "y"], "asserts_effect": True},
            {
                "claim": "other relates",
                "variables": ["z"],
                "test_ids": [record.id],
                "asserts_effect": True,
            },
            {
                "claim": "supported",
                "variables": ["x", "y"],
                "test_ids": [record.id],
                "asserts_effect": True,
            },
        ],
        registry,
    )
    assert outcomes[0].reason == "effect claim has no registered test"
    assert outcomes[1].reason == "test does not cover all claimed variables"
    assert outcomes[2].status == "supported"


def test_report_includes_registry_counts_selection_and_rejections() -> None:
    registry = Registry()
    view = registry.register(sample()).extract()
    record = registry.association(view, "x", "y")
    report = build_report(
        registry,
        [
            {
                "claim": "x relates to y",
                "variables": ["x", "y"],
                "test_ids": [record.id],
                "asserts_effect": True,
            },
            {"claim": "unregistered", "variables": ["x", "y"], "asserts_effect": True},
        ],
        {record.id: {"confirmed": True}},
    )
    assert report["m"] == 1
    assert report["R_brut"] == 1
    assert report["confirmed_count"] == 1
    assert report["rejected"][0]["reason"] == "effect claim has no registered test"
    assert report["selection_diagnostic"]["reported_p_curve"]["n"] == 1


def test_report_counts_duplicate_raw_hypotheses_once_but_tests_twice() -> None:
    registry = Registry()
    view = registry.register(sample()).extract()
    registry.association(view, "x", "y")
    registry.association(view, "x", "y")

    report = build_report(registry, [])

    assert report["m"] == 2
    assert report["R_brut"] == 1


def test_claim_rejections_for_blank_unknown_failed_and_null() -> None:
    registry = Registry()
    rng = np.random.default_rng(84)
    view = registry.register(
        pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    ).extract()
    failed = registry.association(view, "absent", "y")
    flat = registry.association(view, "x", "y")
    outcomes = validate_claims(
        [
            {"claim": "  ", "asserts_effect": True},
            {"claim": "unknown", "test_ids": ["unknown"], "asserts_effect": True},
            {
                "claim": "failed",
                "variables": ["x", "y"],
                "test_ids": [failed.id],
                "asserts_effect": True,
            },
            {
                "claim": "not significant",
                "variables": ["x", "y"],
                "test_ids": [flat.id],
                "asserts_effect": True,
            },
            {"claim": "no effect is asserted", "asserts_effect": False},
        ],
        registry,
    )
    assert outcomes[0].reason == "empty claim text"
    assert outcomes[1].reason == "unknown registry test ID"
    assert outcomes[2].reason == "registered test failed"
    assert outcomes[3].reason == "all referenced tests have p >= 0.05"
    assert outcomes[4].status == "not_established"


def test_adjusted_claim_is_qualified_and_holdout_status_is_reported() -> None:
    registry = Registry()
    data = sample()
    data["z"] = np.random.default_rng(9).normal(size=len(data))
    view = registry.register(data).extract()
    adjusted = registry.regression(view, "y", "x", adjust_for=["z"])
    adjusted_report = validate_claims(
        [
            {
                "claim": "adjusted estimate",
                "variables": ["x", "y"],
                "test_ids": [adjusted.id],
                "asserts_effect": True,
            }
        ],
        registry,
    )
    assert (
        adjusted_report[0].reason
        == "association robust to adjustment; causality is not established"
    )
    report = build_report(
        registry,
        [
            {
                "claim": "adjusted estimate",
                "variables": ["x", "y"],
                "test_ids": [adjusted.id],
                "asserts_effect": True,
            }
        ],
        {adjusted.id: {"confirmed": False}},
    )
    assert report["claims"][0]["status"] == "rejected"
    assert report["selection_diagnostic"]["full_registry_ks_uniform"] is None


def test_split_rejects_bad_frame_and_unavailable_promotions() -> None:
    registry = Registry()
    with pytest.raises(ValueError, match="four rows"):
        ConfirmationSplit(sample().iloc[:3], registry)
    split = ConfirmationSplit(sample(), registry)
    with pytest.raises(ValueError, match="Unsupported"):
        split.promote([{"test": "unknown", "variables": ["x", "y"]}])
    with pytest.raises(ValueError, match="distinct"):
        split.promote([{"test": "association", "variables": ["x", "x"]}])
    with pytest.raises(ValueError, match="two-sided"):
        split.promote([{"test": "anova", "variables": ["x", "y"], "direction": "greater"}])


def test_split_regression_and_anova_confirmation() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(size=400), "group": np.where(x > 0, "a", "b")})
    for spec in [
        {"test": "regression", "variables": ["x", "y"]},
        {"test": "anova", "variables": ["group", "y"]},
    ]:
        registry = Registry()
        split = ConfirmationSplit(df, registry, seed=4)
        split.promote([spec])
        assert len(split.confirm(registry)) == 1
        assert registry.promotions


def test_split_nonlinear_and_interaction_confirmation() -> None:
    rng = np.random.default_rng(16)
    x = rng.uniform(-2, 2, 500)
    group = rng.binomial(1, 0.5, 500)
    data = pd.DataFrame(
        {
            "x": x,
            "group": group,
            "quadratic": x**2 + rng.normal(scale=0.1, size=500),
            "product": x * group + rng.normal(scale=0.1, size=500),
        }
    )
    for spec in [
        {"test": "nonlinear", "variables": ["x", "quadratic"]},
        {"test": "interaction", "variables": ["x", "group", "product"]},
    ]:
        registry = Registry()
        split = ConfirmationSplit(data, registry, seed=3)
        split.promote([spec])
        assert len(split.confirm(registry)) == 1
