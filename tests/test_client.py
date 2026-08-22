from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from resolutive.client import ResolutiveClient, ResolutiveHTTPError
from resolutive.http_api import create_app


def test_python_client_full_lifecycle() -> None:
    transport = TestClient(create_app())
    client = ResolutiveClient(http_client=transport)

    assert client.health() is True

    session = client.create(
        dimension=3,
        bounds=(-5.0, 5.0),
        budget=8,
        seed=7,
        mode="prototype",
        batch_size=4,
    )

    info = session.info()
    assert info.session_id == session.session_id
    assert info.evaluations == 0
    assert info.remaining == 8

    batch = session.ask()
    assert batch.points.shape == (4, 3)
    values = np.sum(batch.points * batch.points, axis=1)

    updated = session.tell(values)
    assert updated.evaluations == 4
    assert updated.remaining == 4

    result = session.result()
    assert result.status == "running"
    assert result.evaluations == 4
    assert result.x.shape == (3,)

    checkpoint = session.checkpoint()
    assert "resolutive-session" in checkpoint

    attached = client.attach(session.session_id)
    assert attached.session_id == session.session_id

    batch2 = attached.ask()
    attached.tell(np.sum(batch2.points * batch2.points, axis=1))
    final = attached.result()
    assert final.status == "success"
    assert final.evaluations == 8

    attached.delete()
    with pytest.raises(ResolutiveHTTPError) as exc_info:
        client.attach(session.session_id)
    assert exc_info.value.status_code == 404


def test_python_client_surfaces_protocol_conflicts() -> None:
    transport = TestClient(create_app())
    client = ResolutiveClient(http_client=transport)
    session = client.create(
        dimension=2,
        bounds=(-1.0, 1.0),
        budget=4,
        seed=3,
        mode="prototype",
        batch_size=2,
    )

    session.ask()
    with pytest.raises(ResolutiveHTTPError) as exc_info:
        session.ask()
    assert exc_info.value.status_code == 409
    assert "tell() must be called" in exc_info.value.detail
