from fastapi.testclient import TestClient

from api.main import _promotion, app

client = TestClient(app)


def test_deterministic_run_exposes_registry_and_report() -> None:
    response = client.post(
        "/runs",
        json={
            "dataset": [
                {"x": float(index), "y": float(index * 2), "z": float(index % 3)}
                for index in range(30)
            ],
            "config": {"max_tests": 2},
            "agent": "deterministic",
        },
    )
    assert response.status_code == 202
    run_id = response.json()["id"]

    status = client.get(f"/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["m"] >= 2

    registry = client.get(f"/runs/{run_id}/registry")
    assert registry.status_code == 200
    assert registry.json()["m"] == status.json()["m"]
    assert len(registry.json()["tests"]) == registry.json()["m"]

    report = client.get(f"/runs/{run_id}/report")
    assert report.status_code == 200
    assert report.json()["m"] == status.json()["m"]
    assert report.json()["multiplicity"]["method"] == "by"


def test_run_events_emit_final_state() -> None:
    response = client.post(
        "/runs",
        json={
            "dataset": [{"x": index, "y": index} for index in range(5)],
            "config": {"max_tests": 1},
        },
    )
    run_id = response.json()["id"]

    events = client.get(f"/runs/{run_id}/events")
    assert events.status_code == 200
    assert '"status":"completed"' in events.text


def test_unknown_run_and_benchmark_return_not_found() -> None:
    assert client.get("/runs/not-a-run").status_code == 404
    assert client.get("/benchmark/E6").status_code == 404


def test_demo_uses_recorded_legacy_results() -> None:
    response = client.get("/demo")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["claims"]) == 56
    assert payload["archive_summary"] == {
        "claims": 56,
        "effect_claims": 17,
        "false_effect_claims": 17,
        "retained_by_legacy_engine": 0,
    }
    assert payload["null_benchmark"]["seeds"] == 5
    assert "pre-audit-engine" in payload["provenance"]


def test_offline_demo_replays_the_current_audit_engine() -> None:
    response = client.post("/demo/run")
    assert response.status_code == 202
    run_id = response.json()["id"]
    status = client.get(f"/runs/{run_id}").json()
    assert status["status"] == "completed"
    assert status["m"] >= 25
    report = client.get(f"/runs/{run_id}/report").json()
    assert report["multiplicity"]["family_size"] == 25


def test_invalid_agent_is_rejected() -> None:
    response = client.post(
        "/runs",
        json={"dataset": [{"x": 1}], "agent": "unknown"},
    )
    assert response.status_code == 422


def test_promotion_preserves_direction_for_confirmation() -> None:
    promotion = _promotion("association", ("x", "y"), (), "greater")
    assert promotion == {
        "test": "association",
        "variables": ["x", "y"],
        "direction": "greater",
    }
