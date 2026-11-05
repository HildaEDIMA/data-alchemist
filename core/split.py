"""One-shot exploration/confirmation split with pre-registered promotions."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.multiplicity import adjust
from core.registry import Registry, TestRecord, TrackedFrame


@dataclass(frozen=True)
class PromotedHypothesis:
    id: str
    test: str
    variables: tuple[str, ...]
    direction: str = "two-sided"


class ConfirmationSplit:
    def __init__(self, frame: pd.DataFrame, registry: Registry, seed: int = 0) -> None:
        if len(frame) < 4:
            raise ValueError("At least four rows are required for a split")
        self.registry = registry
        registered = registry.register(frame)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(registered.frame))
        middle = len(order) // 2
        self._explore = registered.frame.iloc[order[:middle]].reset_index(drop=True)
        self._confirm = registered.frame.iloc[order[middle:]].reset_index(drop=True)
        self._dataset_id = registered.dataset_id
        self._promoted: tuple[PromotedHypothesis, ...] | None = None
        self._opened = False
        self.exploration = self._register_part(self._explore, f"seed={seed}: exploration half")

    def _register_part(self, frame: pd.DataFrame, expression: str) -> TrackedFrame:
        extraction_id = self.registry._record_extraction(self._dataset_id, frame, expression)
        return TrackedFrame(self.registry, extraction_id, frame.copy(deep=True), expression)

    def promote(self, hypotheses: Sequence[dict[str, Any]]) -> tuple[PromotedHypothesis, ...]:
        if self._promoted is not None or self._opened:
            raise RuntimeError("Promotions are frozen; hypotheses can only be promoted once")
        if not hypotheses:
            raise ValueError("At least one hypothesis must be promoted")
        promoted: list[PromotedHypothesis] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for item in hypotheses:
            test = str(item.get("test", ""))
            variables = tuple(item.get("variables", ()))
            direction = str(item.get("direction", "two-sided"))
            if not test or len(variables) < 2 or not all(isinstance(v, str) for v in variables):
                raise ValueError("Each promotion needs a test and at least two variable names")
            if test not in {
                "association",
                "mann_whitney",
                "anova",
                "regression",
                "nonlinear",
                "interaction",
            }:
                raise ValueError(f"Unsupported confirmation test: {test}")
            if len(set(variables)) != len(variables):
                raise ValueError("Promoted variables must be distinct")
            if direction not in ("two-sided", "greater", "less"):
                raise ValueError("direction must be two-sided, greater, or less")
            if test in {"anova", "regression"} and direction != "two-sided":
                raise ValueError(f"{test} confirmation is two-sided")
            expected_count = 3 if test == "interaction" else 2
            if test != "regression" and len(variables) != expected_count:
                raise ValueError(f"{test} confirmation requires {expected_count} variables")
            if test == "regression" and len(variables) < 2:
                raise ValueError("regression confirmation requires predictor and outcome")
            key = test, variables
            if key in seen:
                raise ValueError("Duplicate hypotheses cannot be promoted")
            seen.add(key)
            promoted.append(
                PromotedHypothesis(f"P{len(promoted) + 1:04d}", test, variables, direction)
            )
        self._promoted = tuple(promoted)
        self.registry.promotions.append(
            {
                "timestamp": time.time(),
                "hypotheses": [
                    {
                        "id": item.id,
                        "test": item.test,
                        "variables": item.variables,
                        "direction": item.direction,
                    }
                    for item in self._promoted
                ],
            }
        )
        return self._promoted

    def open_confirmation(self) -> TrackedFrame:
        if self._promoted is None:
            raise RuntimeError("Pre-register promoted hypotheses before opening confirmation")
        if self._opened:
            raise RuntimeError("Confirmation data are single-use")
        self._opened = True
        extraction_id = self.registry._record_extraction(
            self._dataset_id, self._confirm, "one-time holdout (confirmation)"
        )
        return TrackedFrame(self.registry, extraction_id, self._confirm.copy(deep=True))

    def confirm(self, test_api: Any) -> list[dict[str, Any]]:
        if self._promoted is None:
            raise RuntimeError("Pre-register promoted hypotheses before confirmation")
        view = self.open_confirmation()
        records: list[TestRecord] = []
        for hypothesis in self._promoted:
            if hypothesis.test == "association" and len(hypothesis.variables) == 2:
                record = test_api.association(
                    view, *hypothesis.variables, direction=hypothesis.direction
                )
            elif hypothesis.test == "mann_whitney" and len(hypothesis.variables) == 2:
                record = test_api.mann_whitney(
                    view, *hypothesis.variables, direction=hypothesis.direction
                )
            elif hypothesis.test == "anova" and len(hypothesis.variables) == 2:
                if hypothesis.direction != "two-sided":
                    raise ValueError("ANOVA confirmation is two-sided")
                record = test_api.anova(view, *hypothesis.variables)
            elif hypothesis.test == "regression" and len(hypothesis.variables) >= 2:
                if hypothesis.direction != "two-sided":
                    raise ValueError("Regression confirmation is two-sided")
                record = test_api.regression(
                    view,
                    hypothesis.variables[-1],
                    hypothesis.variables[0],
                    adjust_for=hypothesis.variables[1:-1],
                )
            elif hypothesis.test == "nonlinear" and len(hypothesis.variables) == 2:
                if hypothesis.direction != "two-sided":
                    raise ValueError("Nonlinearity confirmation is two-sided")
                record = test_api.nonlinear(view, *hypothesis.variables)
            elif hypothesis.test == "interaction" and len(hypothesis.variables) == 3:
                if hypothesis.direction != "two-sided":
                    raise ValueError("Interaction confirmation is two-sided")
                record = test_api.interaction(view, *hypothesis.variables)
            else:
                raise ValueError(f"Unsupported confirmation test: {hypothesis.test}")
            records.append(record)
        result = adjust([record.p_value for record in records], "bh")
        return [
            {
                "promotion_id": hypothesis.id,
                "record": record,
                "adjusted_p": result.adjusted_p[index],
                "confirmed": result.rejected[index] and record.error is None,
            }
            for index, (hypothesis, record) in enumerate(
                zip(self._promoted, records, strict=True)
            )
        ]
