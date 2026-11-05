"""Deterministic audit reports derived from the registry and claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy import stats

from core.claims import validate_claims
from core.multiplicity import adjust
from core.registry import Registry


def build_report(
    registry: Registry,
    claims: Sequence[dict[str, Any]],
    confirmation: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    adjustment_method: str = "by",
    alpha: float = 0.05,
    correction_test_ids: set[str] | None = None,
) -> dict[str, Any]:
    outcomes = validate_claims(claims, registry)
    correction_family = [
        record
        for record in registry.tests
        if record.error is None
        and (correction_test_ids is None or record.id in correction_test_ids)
    ]
    correction = adjust(
        [record.p_value for record in correction_family], adjustment_method, alpha
    )
    adjusted_p = {
        record.id: correction.adjusted_p[index]
        for index, record in enumerate(correction_family)
    }
    rejected_ids = {
        record.id
        for record, rejected in zip(correction_family, correction.rejected, strict=True)
        if rejected
    }
    rendered: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for outcome in outcomes:
        result = asdict(outcome)
        result["evidence"] = [asdict(record) for record in outcome.evidence]
        if outcome.status == "supported":
            result["adjusted_p_values"] = [
                adjusted_p.get(record.id, 1.0)
                for record in outcome.evidence
            ]
            if not all(record.id in rejected_ids for record in outcome.evidence):
                result["status"] = "rejected"
                result["reason"] = f"not retained after {adjustment_method} correction"
            elif confirmation is None:
                result["status"] = "rejected"
                result["reason"] = "not confirmed on a pre-registered holdout"
                result["confirmed"] = False
            else:
                result["confirmed"] = all(
                    bool(confirmation.get(record.id, {}).get("confirmed"))
                    for record in outcome.evidence
                )
                if not result["confirmed"]:
                    result["status"] = "rejected"
                    result["reason"] = (
                        "not confirmed after BH correction on the pre-registered holdout"
                    )
        rendered.append(result)
        if result["status"] in {"rejected", "not_established"}:
            rejected.append({"claim": outcome.claim, "reason": result["reason"] or "rejected"})
    p_values = [record.p_value for record in registry.tests if record.error is None]
    reported_p = [
        record.p_value
        for outcome in outcomes
        if outcome.asserts_effect
        for record in outcome.evidence
        if record.error is None
    ]
    raw_discoveries = {
        (
            record.extraction_id,
            record.test,
            record.variables,
            record.direction,
            record.adjusted_for,
        )
        for record in registry.tests
        if record.error is None and record.p_value < 0.05
    }
    ks_uniform = stats.kstest(p_values, "uniform") if len(p_values) >= 2 else None
    ks_selection = stats.ks_2samp(p_values, reported_p) if len(p_values) and reported_p else None
    return {
        "m": registry.m,
        "R_brut": len(raw_discoveries),
        "confirmed_count": sum(item.get("confirmed", False) for item in rendered),
        "discoveries_after_correction": len(rejected_ids),
        "multiplicity": {
            "method": adjustment_method,
            "alpha": alpha,
            "family_size": len(correction_family),
            "rejected_test_ids": sorted(rejected_ids),
            "guarantee": correction.guarantee,
        },
        "claims": rendered,
        "rejected": rejected,
        "selection_diagnostic": {
            "full_registry_ks_uniform": None
            if ks_uniform is None
            else {
                "statistic": float(ks_uniform.statistic),
                "p_value": float(ks_uniform.pvalue),
                "n": len(p_values),
            },
            "registry_vs_reported_ks": None
            if ks_selection is None
            else {
                "statistic": float(ks_selection.statistic),
                "p_value": float(ks_selection.pvalue),
                "n_registry": len(p_values),
                "n_reported": len(reported_p),
            },
            "reported_p_curve": _p_curve(reported_p),
            "warning": "KS non-rejection is not proof of uniformity; p-curve is descriptive.",
        },
        "error_control": {
            "exploration": correction.guarantee,
            "confirmation": (
                "BH FDR at most 5% under independence or PRDS and valid holdout p-values; "
                "holdout hypotheses were fixed before opening confirmation data"
                if confirmation is not None
                else "No holdout confirmation was supplied; exploratory claims are not confirmed"
            ),
        },
    }


def _p_curve(p_values: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(p_values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0) & (values <= 1)]
    bins = [0.01, 0.05, 0.1, 0.5, 1.0]
    counts = [
        int(((values >= lower) & (values < upper)).sum())
        for lower, upper in zip([0.0, *bins[:-1]], bins, strict=True)
    ]
    return {
        "bins": bins,
        "counts": counts,
        "n": int(len(values)),
        "interpretation": "Descriptive only; not a test of publication bias.",
    }
