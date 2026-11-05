"""Replay archived evidence and the current offline audit engine."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    with TestClient(app) as client:
        archive_response = client.get("/demo")
        archive_response.raise_for_status()
        archive = archive_response.json()
        print("Historical archive (pre-audit engine)")
        print(f"  Provenance: {archive['provenance']}")
        print(f"  Datasets: {archive['datasets']}")
        print(f"  Claims: {archive['archive_summary']['claims']}")
        print(f"  False effect claims: {archive['archive_summary']['false_effect_claims']}")
        print(
            "  Historical test count: unavailable "
            "(the original complete registry was not archived)"
        )

        run_response = client.post("/demo/run")
        run_response.raise_for_status()
        run_id = run_response.json()["id"]
        status_response = client.get(f"/runs/{run_id}")
        status_response.raise_for_status()
        status = status_response.json()
        if status["status"] != "completed":
            raise RuntimeError(f"Offline demo run did not complete: {status}")

        report_response = client.get(f"/runs/{run_id}/report")
        report_response.raise_for_status()
        report = report_response.json()
        print("\nCurrent engine (new offline run)")
        print(f"  Run status: {status['status']}")
        print(f"  Registered tests, m: {report['m']}")
        print(f"  Raw discoveries, R_brut: {report['R_brut']}")
        print(f"  Confirmed claims: {report['confirmed_count']}")
        print(f"  Claims rejected or not established: {len(report['rejected'])}")


if __name__ == "__main__":
    main()
