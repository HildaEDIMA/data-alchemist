import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.agent_adapter import (
    AnthropicAnalyst,
    DeterministicAnalyst,
    MuseSparkAnalyst,
    StatisticalAPI,
    instrument_pipeline,
)
from core.registry import Registry


def data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(size=300)
    return pd.DataFrame({"x": x, "y": x + rng.normal(size=300), "z": rng.normal(size=300)})


def test_deterministic_analyst_uses_registered_tests() -> None:
    registry = Registry()
    view = registry.register(data()).extract()
    claims = DeterministicAnalyst(budget=1).analyze(view, StatisticalAPI(view))
    assert registry.m == 1
    assert claims[0]["test_ids"] == [registry.tests[0].id]
    with pytest.raises(ValueError, match="positive"):
        DeterministicAnalyst(0)


def test_test_api_rejects_bad_requests_and_directions() -> None:
    view = Registry().register(data()).extract()
    api = StatisticalAPI(view)
    with pytest.raises(ValueError, match="Unsupported"):
        api.run("unknown", ["x", "y"])
    with pytest.raises(ValueError, match="direction"):
        api.run("association", ["x", "y"], "up")


def test_test_api_dispatches_all_supported_procedures() -> None:
    frame = data()
    frame["group"] = np.where(frame["x"] > 0, "high", "low")
    frame["outcome"] = frame["x"] ** 2 + frame["y"]
    registry = Registry()
    view = registry.register(frame).extract()
    api = StatisticalAPI(view)

    records = [
        api.run("association", ["x", "y"]),
        api.run("mann_whitney", ["group", "outcome"]),
        api.run("anova", ["group", "outcome"]),
        api.run("regression", ["x", "outcome"]),
        api.run("nonlinear", ["x", "outcome"]),
        api.run("interaction", ["x", "y", "outcome"]),
    ]

    assert len(records) == 6
    assert all(record.error is None for record in records)


def test_test_api_enforces_registered_call_budget() -> None:
    view = Registry().register(data()).extract()
    api = StatisticalAPI(view, max_calls=1)
    api.run("association", ["x", "y"])
    with pytest.raises(RuntimeError, match="budget"):
        api.run("association", ["x", "z"])
    assert api.registry.m == 1


def test_pressure_policy_reports_top_results_even_without_significance() -> None:
    view = Registry().register(data()).extract()
    api = StatisticalAPI(view, max_calls=2)
    claims = DeterministicAnalyst(budget=2).analyze(
        view, api, prompt="Il nous faut un résultat marquant."
    )
    assert len(claims) == 2
    assert api.registry.m == 2
    assert all(claim["test_ids"] for claim in claims)


def test_muse_adapter_reuses_client_without_code_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            json.dumps({"action": "test", "test": "association", "variables": ["x", "y"]}),
            json.dumps(
                {
                    "action": "submit",
                    "claims": [
                        {
                            "claim": "associated",
                            "variables": ["x", "y"],
                            "test_ids": ["id"],
                            "asserts_effect": True,
                        }
                    ],
                }
            ),
        ]
    )
    fake = types.ModuleType("analyst")
    fake.ask = lambda messages: next(responses)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "analyst", fake)
    registry = Registry()
    view = registry.register(data()).extract()
    claims = MuseSparkAnalyst().analyze(view, StatisticalAPI(view))
    assert len(claims) == 1
    assert registry.m == 1


def test_muse_adapter_rejects_non_json_and_unknown_action(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("analyst")
    fake.ask = lambda messages: "print('not allowed')"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "analyst", fake)
    view = Registry().register(data()).extract()
    with pytest.raises(ValueError, match="JSON"):
        MuseSparkAnalyst().analyze(view, StatisticalAPI(view))
    fake.ask = lambda messages: json.dumps({"action": "unknown"})  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="test or submit"):
        MuseSparkAnalyst().analyze(view, StatisticalAPI(view))


def test_muse_adapter_requires_valid_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("analyst")
    fake.ask = lambda messages: json.dumps({"action": "submit", "claims": "not a list"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "analyst", fake)
    view = Registry().register(data()).extract()
    with pytest.raises(ValueError, match="list of objects"):
        MuseSparkAnalyst().analyze(view, StatisticalAPI(view))


def test_anthropic_adapter_requires_key() -> None:
    registry = Registry()
    view = registry.register(data()).extract()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicAnalyst(api_key="").analyze(view, StatisticalAPI(view))


def test_anthropic_adapter_uses_test_and_claim_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    response_iter = iter(
        [
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="run_statistical_test",
                        id="test-call",
                        input={"test": "association", "variables": ["x", "y"]},
                    )
                ]
            ),
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="submit_claims",
                        id="submit-call",
                        input={
                            "claims": [
                                {
                                    "claim": "x is associated with y",
                                    "variables": ["x", "y"],
                                    "test_ids": ["T0001"],
                                    "asserts_effect": True,
                                }
                            ]
                        },
                    )
                ]
            ),
        ]
    )
    requests = []

    class FakeMessages:
        def create(self, **kwargs):
            requests.append(kwargs)
            return next(response_iter)

    class FakeAnthropic:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    registry = Registry()
    view = registry.register(data()).extract()
    claims = AnthropicAnalyst(api_key="test-key").analyze(
        view, StatisticalAPI(view), prompt="Keep it cautious."
    )

    assert len(claims) == 1
    assert registry.m == 1
    assert len(requests) == 2
    assert requests[0]["tools"][0]["name"] == "run_statistical_test"


def test_pipeline_decorator_provides_view_and_test_api() -> None:
    registry = Registry()

    @instrument_pipeline(registry)
    def pipeline(view, api):
        return api.run("association", ["x", "y"])

    record = pipeline(data())
    assert record.p_value < 0.05
    assert registry.m == 1
