"""Validation of model claims against registered evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.registry import Registry, TestRecord


@dataclass(frozen=True)
class ClaimResult:
    claim: str
    variables: tuple[str, ...]
    test_ids: tuple[str, ...]
    asserts_effect: bool
    status: str
    reason: str | None
    evidence: tuple[TestRecord, ...]


def validate_claims(claims: Sequence[dict[str, Any]], registry: Registry) -> list[ClaimResult]:
    validated: list[ClaimResult] = []
    for claim in claims:
        text = str(claim.get("claim", ""))
        variables = tuple(claim.get("variables", ()))
        test_ids = tuple(claim.get("test_ids", claim.get("registry_ids", ())))
        asserts_effect = bool(claim.get("asserts_effect", False))
        if not text.strip():
            validated.append(
                ClaimResult(
                    text, variables, test_ids, asserts_effect, "rejected", "empty claim text", ()
                )
            )
            continue
        if asserts_effect and not test_ids:
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    True,
                    "rejected",
                    "effect claim has no registered test",
                    (),
                )
            )
            continue
        records = tuple(registry.get_test(test_id) for test_id in test_ids)
        if any(record is None for record in records):
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    asserts_effect,
                    "rejected",
                    "unknown registry test ID",
                    (),
                )
            )
            continue
        evidence = tuple(record for record in records if record is not None)
        if any(record.error is not None for record in evidence):
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    asserts_effect,
                    "rejected",
                    "registered test failed",
                    evidence,
                )
            )
            continue
        if variables and any(not set(variables).issubset(record.variables) for record in evidence):
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    asserts_effect,
                    "rejected",
                    "test does not cover all claimed variables",
                    evidence,
                )
            )
            continue
        if asserts_effect and all(record.p_value >= 0.05 for record in evidence):
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    True,
                    "rejected",
                    "all referenced tests have p >= 0.05",
                    evidence,
                )
            )
            continue
        if asserts_effect and any(record.p_value >= 0.05 for record in evidence):
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    True,
                    "rejected",
                    "one or more referenced tests have p >= 0.05",
                    evidence,
                )
            )
            continue
        if not asserts_effect:
            validated.append(
                ClaimResult(
                    text,
                    variables,
                    test_ids,
                    False,
                    "not_established",
                    "failure to reject is not evidence for the null; "
                    "an equivalence test is required",
                    evidence,
                )
            )
            continue
        if any(record.adjusted_for for record in evidence):
            reason = "association robust to adjustment; causality is not established"
        else:
            reason = None
        validated.append(
            ClaimResult(
                text,
                variables,
                test_ids,
                asserts_effect,
                "supported",
                reason,
                evidence,
            )
        )
    return validated
