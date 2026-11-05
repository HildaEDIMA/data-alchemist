"""Asynchronous HTTP API for instrumented analysis runs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent_adapter import (
    Analyst,
    AnthropicAnalyst,
    DeterministicAnalyst,
    MuseSparkAnalyst,
    StatisticalAPI,
)
from core.registry import Registry
from core.report import build_report
from core.split import ConfirmationSplit
from generate_benchmark import make_benchmark


class RunConfig(BaseModel):
    prompt: str = ""
    max_tests: int = Field(default=25, ge=1, le=10_000)


class RunRequest(BaseModel):
    dataset: list[dict[str, Any]] = Field(min_length=1)
    config: RunConfig = Field(default_factory=RunConfig)
    agent: Literal["deterministic", "muse", "anthropic"] = "deterministic"


class RunResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    agent: str
    m: int
    created_at: float
    error: str | None = None


@dataclass
class RunState:
    id: str
    agent: str
    created_at: float
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    registry: Registry | None = None
    report: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def response(self) -> RunResponse:
        with self.lock:
            return RunResponse(
                id=self.id,
                status=self.status,
                agent=self.agent,
                m=0 if self.registry is None else self.registry.m,
                created_at=self.created_at,
                error=self.error,
            )


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()

    def create(self, agent: str) -> RunState:
        run = RunState(str(uuid.uuid4()), agent, time.time())
        with self._lock:
            self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> RunState:
        with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    def execute(
        self, run_id: str, dataset: list[dict[str, Any]], config: RunConfig
    ) -> None:
        run = self.get(run_id)
        with run.lock:
            run.status = "running"
        try:
            frame = pd.DataFrame(dataset)
            if frame.empty or frame.shape[1] == 0:
                raise ValueError("Dataset must contain at least one column and one row")
            registry = Registry()
            split = ConfirmationSplit(frame, registry)
            view = split.exploration
            with run.lock:
                run.registry = registry
            exploration_claims_start = len(registry.tests)
            analyst = _make_analyst(run.agent, config.max_tests)
            claims = analyst.analyze(
                view,
                StatisticalAPI(view, max_calls=config.max_tests),
                prompt=config.prompt,
                max_turns=min(8, config.max_tests),
            )
            exploration_records = registry.tests[exploration_claims_start:]
            correction_test_ids = {record.id for record in exploration_records}
            confirmation: dict[str, dict[str, Any]] = {}
            promotion_specs: list[dict[str, Any]] = []
            source_test_ids: list[str] = []
            seen_test_ids: set[str] = set()
            for claim in claims:
                if not claim.get("asserts_effect"):
                    continue
                test_ids = claim.get("test_ids", claim.get("registry_ids", []))
                if not isinstance(test_ids, list):
                    continue
                for test_id in test_ids:
                    if not isinstance(test_id, str) or test_id in seen_test_ids:
                        continue
                    record = registry.get_test(test_id)
                    if record is None or record.error is not None:
                        continue
                    promotion = _promotion(
                        record.procedure,
                        record.variables,
                        record.adjusted_for,
                        record.direction,
                    )
                    if promotion is None:
                        continue
                    promotion_specs.append(promotion)
                    source_test_ids.append(record.id)
                    seen_test_ids.add(record.id)
            if promotion_specs:
                split.promote(promotion_specs)
                results = split.confirm(registry)
                if len(results) != len(source_test_ids):
                    raise RuntimeError("Confirmation results do not match promoted hypotheses")
                confirmation = {
                    source_id: {
                        "confirmed": item["confirmed"],
                        "adjusted_p": item["adjusted_p"],
                        "record": asdict(item["record"]),
                    }
                    for source_id, item in zip(source_test_ids, results, strict=True)
                }
            report = build_report(
                registry,
                claims,
                confirmation,
                adjustment_method="by",
                correction_test_ids=correction_test_ids,
            )
            with run.lock:
                run.report = report
                run.status = "completed"
        except Exception as error:
            with run.lock:
                run.error = f"{type(error).__name__}: {error}"
                run.status = "failed"


def _make_analyst(agent: str, max_tests: int) -> Analyst:
    if agent == "deterministic":
        return DeterministicAnalyst(budget=max_tests)
    if agent == "muse":
        return MuseSparkAnalyst(max_turns=min(8, max_tests))
    if agent == "anthropic":
        return AnthropicAnalyst()
    raise ValueError(f"Unsupported agent: {agent}")


def _promotion(
    procedure: str,
    variables: tuple[str, ...],
    adjusted_for: tuple[str, ...],
    direction: str = "two-sided",
) -> dict[str, Any] | None:
    if procedure in {"association", "mann_whitney", "anova"}:
        test = procedure
        names = variables
    elif procedure == "ols_hc3_wald":
        test = "regression"
        names = (variables[0], *adjusted_for, variables[-1])
    elif procedure == "quadratic_hc3_wald":
        test = "nonlinear"
        names = variables
    elif procedure == "interaction_hc3_wald":
        test = "interaction"
        names = variables
    else:
        return None
    return {"test": test, "variables": list(names), "direction": direction}


manager = RunManager()
app = FastAPI(title="Data Alchemist API", version="0.1.0")
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/runs", response_model=RunResponse, status_code=202)
def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> RunResponse:
    run = manager.create(request.agent)
    background_tasks.add_task(manager.execute, run.id, request.dataset, request.config)
    return run.response()


@app.post("/demo/run", response_model=RunResponse, status_code=202)
def create_demo_run(background_tasks: BackgroundTasks) -> RunResponse:
    frame, _ = make_benchmark(n=1200, seed=42)
    dataset = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    run = manager.create("deterministic")
    background_tasks.add_task(manager.execute, run.id, dataset, RunConfig(max_tests=25))
    return run.response()


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    return manager.get(run_id).response()


@app.get("/runs/{run_id}/registry")
def get_registry(run_id: str) -> dict[str, Any]:
    run = manager.get(run_id)
    with run.lock:
        if run.registry is None:
            if run.status == "failed":
                raise HTTPException(status_code=409, detail=run.error)
            raise HTTPException(status_code=409, detail="Run has not started")
        registry = run.registry
        payload = jsonable_encoder(
            {
                "m": registry.m,
                "extractions": [asdict(item) for item in registry.extractions],
                "transformations": [asdict(item) for item in registry.transformations],
                "tests": [asdict(item) for item in registry.tests],
                "promotions": registry.promotions,
            }
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Registry serialization did not produce an object")
        return payload


@app.get("/runs/{run_id}/report")
def get_report(run_id: str) -> dict[str, Any]:
    run = manager.get(run_id)
    with run.lock:
        if run.status == "failed":
            raise HTTPException(status_code=409, detail=run.error)
        if run.report is None:
            raise HTTPException(status_code=409, detail="Run is not complete")
        return run.report


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = manager.get(run_id)

    async def events() -> AsyncIterator[str]:
        last_payload = ""
        while True:
            payload = run.response().model_dump_json()
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if run.status in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/benchmark/{experiment}")
def get_benchmark(experiment: str) -> dict[str, Any]:
    if len(experiment) != 2 or experiment[0].upper() != "E" or experiment[1] not in "0123456":
        raise HTTPException(status_code=404, detail="Unknown benchmark experiment")
    path = RESULTS_DIR / f"{experiment.lower()}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Benchmark result is not available")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Benchmark result is not a JSON object")
        return result
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Benchmark result could not be read") from error


@app.get("/demo")
def get_demo() -> dict[str, Any]:
    files = {
        "claims": "muse_null.json",
        "null_benchmark": "evaluation_null.json",
        "planted_benchmark": "evaluation_planted.json",
    }
    demo: dict[str, Any] = {
        "provenance": (
            "Historical, pre-audit-engine results from the repository. "
            "These are not results generated by the current registry."
        )
    }
    try:
        for key, filename in files.items():
            path = RESULTS_DIR / filename
            result = json.loads(path.read_text(encoding="utf-8"))
            if key == "claims":
                if (
                    not isinstance(result, list)
                    or not result
                    or any(not isinstance(seed_claims, list) for seed_claims in result)
                ):
                    raise ValueError(f"{filename} has an unexpected shape")
                claims = [
                    {**claim, "dataset_seed": seed}
                    for seed, seed_claims in enumerate(result)
                    for claim in seed_claims
                    if isinstance(claim, dict)
                ]
                if len(claims) != sum(map(len, result)):
                    raise ValueError(f"{filename} contains a malformed claim")
                demo["datasets"] = len(result)
                demo["archive_summary"] = {
                    "claims": len(claims),
                    "effect_claims": sum(bool(claim.get("asserts_effect")) for claim in claims),
                    "false_effect_claims": sum(
                        bool(claim.get("asserts_effect") and claim.get("false"))
                        for claim in claims
                    ),
                    "retained_by_legacy_engine": sum(
                        bool(claim.get("supported_by_engine")) for claim in claims
                    ),
                }
                demo[key] = claims
            elif not isinstance(result, dict):
                raise ValueError(f"{filename} has an unexpected shape")
            else:
                demo[key] = result
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=500, detail="Recorded demo data could not be read"
        ) from error
    return demo
