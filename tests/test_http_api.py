import numpy as np
from fastapi.testclient import TestClient

from resolutive.http_api import create_app


def test_http_session_lifecycle() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    created = client.post(
        "/sessions",
        json={
            "dimension": 3,
            "bounds": [-5.0, 5.0],
            "budget": 8,
            "seed": 7,
            "mode": "prototype",
            "batch_size": 4,
        },
    )
    assert created.status_code == 201
    sid = created.json()["session_id"]

    asked = client.post(f"/sessions/{sid}/ask")
    assert asked.status_code == 200
    points = np.asarray(asked.json()["points"], dtype=float)
    assert points.shape == (4, 3)

    duplicate_ask = client.post(f"/sessions/{sid}/ask")
    assert duplicate_ask.status_code == 409

    values = np.sum(points * points, axis=1).tolist()
    told = client.post(f"/sessions/{sid}/tell", json={"values": values})
    assert told.status_code == 200
    assert told.json()["evaluations"] == 4
    assert told.json()["remaining"] == 4

    current = client.get(f"/sessions/{sid}/result")
    assert current.status_code == 200
    assert current.json()["status"] == "running"

    checkpoint = client.get(f"/sessions/{sid}/checkpoint")
    assert checkpoint.status_code == 200
    assert "resolutive-session" in checkpoint.json()["checkpoint"]

    asked2 = client.post(f"/sessions/{sid}/ask")
    points2 = np.asarray(asked2.json()["points"], dtype=float)
    values2 = np.sum(points2 * points2, axis=1).tolist()
    finished = client.post(f"/sessions/{sid}/tell", json={"values": values2})
    assert finished.status_code == 200
    assert finished.json()["done"] is True

    result = client.get(f"/sessions/{sid}/result")
    assert result.status_code == 200
    assert result.json()["status"] == "success"
    assert result.json()["evaluations"] == 8

    deleted = client.delete(f"/sessions/{sid}")
    assert deleted.status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_http_unknown_session_is_404() -> None:
    client = TestClient(create_app())
    assert client.get("/sessions/missing").status_code == 404
    assert client.post("/sessions/missing/ask").status_code == 404
