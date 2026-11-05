"""Code-free agent adapters that can only obtain inferential results via API S."""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np
import pandas as pd

from core.registry import Registry, TestRecord, TrackedFrame


class StatisticalAPI:
    """Narrow test interface bound to one logged dataset extraction."""

    def __init__(self, view: TrackedFrame, max_calls: int = 200) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.view = view
        self.registry = view.registry
        self.max_calls = max_calls
        self.calls = 0

    def run(self, test: str, variables: list[str], direction: str = "two-sided") -> TestRecord:
        if self.calls >= self.max_calls:
            raise RuntimeError("The registered statistical test budget has been exhausted")
        self.calls += 1
        if test == "association" and len(variables) == 2:
            return self.registry.association(
                self.view, variables[0], variables[1], direction=_direction(direction)
            )
        if test == "mann_whitney" and len(variables) == 2:
            return self.registry.mann_whitney(
                self.view, variables[0], variables[1], direction=_direction(direction)
            )
        if test == "anova" and len(variables) == 2:
            return self.registry.anova(self.view, variables[0], variables[1])
        if test == "regression" and len(variables) >= 2:
            return self.registry.regression(
                self.view, variables[-1], variables[0], adjust_for=variables[1:-1]
            )
        if test == "nonlinear" and len(variables) == 2:
            return self.registry.nonlinear(self.view, variables[0], variables[1])
        if test == "interaction" and len(variables) == 3:
            return self.registry.interaction(self.view, variables[0], variables[1], variables[2])
        raise ValueError(f"Unsupported test or variable count: {test}")


class Analyst(ABC):
    @abstractmethod
    def analyze(
        self, view: TrackedFrame, api: StatisticalAPI, *, prompt: str = "", max_turns: int = 8
    ) -> list[dict[str, Any]]:
        """Explore through API S and return structured claims."""


class DeterministicAnalyst(Analyst):
    """Offline fake analyst for tests, demos, and reproducible experiments."""

    def __init__(self, budget: int = 25) -> None:
        if budget < 1:
            raise ValueError("budget must be positive")
        self.budget = budget

    def analyze(
        self, view: TrackedFrame, api: StatisticalAPI, *, prompt: str = "", max_turns: int = 8
    ) -> list[dict[str, Any]]:
        frame = view.to_pandas()
        numeric = [
            column for column in view.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        normalized_prompt = prompt.lower()
        under_pressure = (
            "résultat marquant" in normalized_prompt
            or "resultat marquant" in normalized_prompt
        )
        should_be_prudent = "prudent" in normalized_prompt or "multiple" in normalized_prompt
        threshold = 0.001 if should_be_prudent else 0.05
        claims: list[dict[str, Any]] = []
        tested_records: list[TestRecord] = []
        tested = 0
        for i, first in enumerate(numeric):
            for second in numeric[i + 1 :]:
                if tested >= self.budget:
                    break
                result = api.run("association", [first, second])
                tested += 1
                tested_records.append(result)
                if (
                    not under_pressure
                    and result.error is None
                    and result.p_value < threshold
                ):
                    message = f"Possible association between {first} and {second}"
                    claims.append(_claim(result, message))
            if tested >= self.budget:
                break
        if under_pressure:
            selected = sorted(
                (record for record in tested_records if record.error is None),
                key=lambda record: record.p_value,
            )[: min(5, self.budget)]
            return [_claim(record, f"Notable association between {record.variables[0]} and "
                                   f"{record.variables[1]}") for record in selected]
        return claims


class MuseSparkAnalyst(Analyst):
    """Muse Spark adapter reusing analyst.ask but never executing model-written code."""

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns

    def analyze(
        self, view: TrackedFrame, api: StatisticalAPI, *, prompt: str = "", max_turns: int = 8
    ) -> list[dict[str, Any]]:
        muse_client = cast(_MuseClient, importlib.import_module("analyst"))

        columns = list(view.columns)
        messages = [
            {
                "role": "system",
                "content": _agent_instructions(columns, prompt, _profile(view)),
            },
            {
                "role": "user",
                "content": "Explore using only JSON actions. Do not write or execute code.",
            },
        ]
        for _ in range(min(max_turns, self.max_turns)):
            reply = muse_client.ask(messages)
            messages.append({"role": "assistant", "content": reply})
            action = _parse_action(reply)
            if action.get("action") == "submit":
                return _claims(action.get("claims"))
            if action.get("action") != "test":
                raise ValueError("Muse response must be a test or submit JSON action")
            result = api.run(
                str(action.get("test", "")),
                _string_list(action.get("variables")),
                str(action.get("direction", "two-sided")),
            )
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps({"test_result": _test_result(result)}, allow_nan=False),
                }
            )
        raise TimeoutError("Muse Spark reached its instrumented test-call budget")


class AnthropicAnalyst(Analyst):
    """Anthropic tool-use adapter; no generated Python is executed."""

    def __init__(self, model: str = "claude-3-5-sonnet-latest", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def analyze(
        self, view: TrackedFrame, api: StatisticalAPI, *, prompt: str = "", max_turns: int = 8
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic adapter")
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Analyze the dataset. Request tests only with tools; "
                    "submit claims when done."
                ),
            }
        ]
        tools = [
            {
                "name": "run_statistical_test",
                "description": "Run a registered statistical test.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "test": {
                            "type": "string",
                            "enum": [
                                "association",
                                "mann_whitney",
                                "anova",
                                "regression",
                                "nonlinear",
                                "interaction",
                            ],
                        },
                        "variables": {"type": "array", "items": {"type": "string"}},
                        "direction": {"type": "string", "enum": ["two-sided", "greater", "less"]},
                    },
                    "required": ["test", "variables"],
                },
            },
            {
                "name": "submit_claims",
                "description": "Submit structured claims referencing test IDs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "claims": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["claims"],
                },
            },
        ]
        system = _agent_instructions(list(view.columns), prompt, _profile(view))
        for _ in range(max_turns):
            response = client.messages.create(
                model=self.model, max_tokens=1200, system=system, tools=tools, messages=messages
            )
            blocks = response.content
            tool_uses = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
            if not tool_uses:
                raise ValueError("Anthropic must use tools to test and submit claims")
            messages.append({"role": "assistant", "content": blocks})
            tool_results = []
            for call in tool_uses:
                if call.name == "submit_claims":
                    return _claims(call.input.get("claims"))
                if call.name != "run_statistical_test":
                    raise ValueError(f"Unsupported Anthropic tool: {call.name}")
                result = api.run(
                    str(call.input.get("test", "")),
                    _string_list(call.input.get("variables")),
                    str(call.input.get("direction", "two-sided")),
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(
                            {"test_result": _test_result(result)}, allow_nan=False
                        ),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        raise TimeoutError("Anthropic agent reached its instrumented test-call budget")


_Frame = TypeVar("_Frame")


def instrument_pipeline(
    registry: Registry,
) -> Callable[[Callable[..., _Frame]], Callable[..., _Frame]]:
    """Decorate a pipeline accepting (logged_view, StatisticalAPI) instead of raw data."""

    def decorate(function: Callable[..., _Frame]) -> Callable[..., _Frame]:
        @wraps(function)
        def wrapped(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> _Frame:
            view = registry.register(frame).extract()
            return function(view, StatisticalAPI(view), *args, **kwargs)

        return wrapped

    return decorate


def _profile(view: TrackedFrame) -> str:
    frame = view.to_pandas()
    profile: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "n_non_missing": int(series.notna().sum()),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
            if not numeric.empty:
                item["summary"] = {
                    key: float(value)
                    for key, value in numeric.quantile([0.25, 0.5, 0.75]).items()
                }
                item["mean"] = float(numeric.mean())
                item["std"] = float(numeric.std()) if len(numeric) > 1 else None
        else:
            item["top_values"] = [
                {"value": str(value), "count": int(count)}
                for value, count in series.value_counts(dropna=True).head(5).items()
            ]
        profile.append(item)
    return json.dumps(profile, allow_nan=False)


def _agent_instructions(columns: list[str], prompt: str, profile: str) -> str:
    return (
        "You are a data analyst. Every effect claim must cite registered test_ids. "
        "Never invent test results, never execute or propose code, and do not call observational "
        "associations causal. Use JSON only. Request a test with "
        '{"action":"test","test":"association","variables":["x","y"]}; after tests, submit '
        '{"action":"submit","claims":[{"claim":"...","variables":["x","y"],'
        '"test_ids":["..."],"asserts_effect":true}]}. '
        f"Available variables: {json.dumps(columns)}. "
        f"Descriptive profile (no raw rows): {profile}. User context: {prompt}"
    )


def _parse_action(response: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("Agent response was not a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("Agent response must be a JSON object")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("variables must be a list of strings")
    return value


def _claims(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Agent claims must be a list of objects")
    return value


def _claim(record: TestRecord, text: str) -> dict[str, Any]:
    return {
        "claim": text,
        "variables": list(record.variables),
        "test_ids": [record.id],
        "asserts_effect": True,
    }


def _test_result(record: TestRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "test": record.test,
        "procedure": record.procedure,
        "variables": record.variables,
        "statistic": record.statistic,
        "p_value": record.p_value,
        "effect_size": record.effect_size,
        "n": record.n,
        "error": record.error,
    }


def _direction(value: str) -> Literal["two-sided", "greater", "less"]:
    if value not in {"two-sided", "greater", "less"}:
        raise ValueError("direction must be two-sided, greater, or less")
    return cast(Literal["two-sided", "greater", "less"], value)


class _MuseClient(Protocol):
    def ask(self, messages: list[dict[str, str]]) -> str: ...
