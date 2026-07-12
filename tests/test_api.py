"""API smoke test using FastAPI TestClient."""
from fastapi.testclient import TestClient

from oran_fed_robust.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_aggregate_endpoint():
    payload = {
        "updates": [[1.0, 1.0, 1.0], [1.1, 0.9, 1.0], [50.0, 50.0, 50.0]],
        "client_ids": [0, 1, 2],
        "aggregator": "median",
        "n_byzantine": 1,
    }
    r = client.post("/aggregate", json=payload)
    assert r.status_code == 200
    out = r.json()["aggregated"]
    assert len(out) == 3
    assert abs(out[0] - 1.1) < 1.0  # median resists the 50.0 outlier
