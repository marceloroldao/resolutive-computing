"""Regression guards for the documented Resolutive Computing API v1 contract."""
from __future__ import annotations

from resolutive.client import (
    RemoteAskBatch,
    RemoteResult,
    RemoteSession,
    RemoteSessionInfo,
    ResolutiveClient,
    ResolutiveHTTPError,
)
from resolutive.http_api import create_app


DOCUMENTED_ROUTES = {
    ("GET", "/health"),
    ("POST", "/sessions"),
    ("GET", "/sessions/{session_id}"),
    ("POST", "/sessions/{session_id}/ask"),
    ("POST", "/sessions/{session_id}/tell"),
    ("GET", "/sessions/{session_id}/result"),
    ("GET", "/sessions/{session_id}/checkpoint"),
    ("DELETE", "/sessions/{session_id}"),
}


def test_documented_v1_http_routes_exist() -> None:
    app = create_app()
    observed = {
        (method, route.path)
        for route in app.routes
        for method in (route.methods or set())
    }
    missing = DOCUMENTED_ROUTES - observed
    assert not missing, f"missing documented v1 routes: {sorted(missing)}"


def test_documented_v1_client_symbols_import() -> None:
    symbols = (
        ResolutiveClient,
        RemoteSession,
        RemoteAskBatch,
        RemoteSessionInfo,
        RemoteResult,
        ResolutiveHTTPError,
    )
    assert all(symbol is not None for symbol in symbols)
