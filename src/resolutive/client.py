"""Python client for the Resolutive Computing HTTP API.

The client targets PC/server deployments and keeps HTTP concerns out of the
optimization core.  ``httpx`` is an optional dependency exposed through the
``client`` extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised only without client extra
    raise ImportError(
        "ResolutiveClient requires the optional client dependencies; "
        "install resolutive-computing[client]"
    ) from exc


class ResolutiveHTTPError(RuntimeError):
    """Error returned by the Resolutive HTTP service."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = int(status_code)
        self.detail = str(detail)
        super().__init__(f"HTTP {self.status_code}: {self.detail}")


@dataclass(frozen=True)
class RemoteAskBatch:
    points: np.ndarray
    generation: int


@dataclass(frozen=True)
class RemoteSessionInfo:
    session_id: str
    done: bool
    evaluations: int
    remaining: int | None
    session_type: str


@dataclass(frozen=True)
class RemoteResult:
    x: np.ndarray
    fun: float
    evaluations: int
    seed: int
    optimizer: str
    status: str
    diagnostics: dict[str, Any]


class RemoteSession:
    """Handle bound to one remote optimization session."""

    def __init__(self, client: "ResolutiveClient", session_id: str) -> None:
        self._client = client
        self.session_id = str(session_id)

    def info(self) -> RemoteSessionInfo:
        return self._client.session_info(self.session_id)

    def ask(self) -> RemoteAskBatch:
        return self._client.ask(self.session_id)

    def tell(self, values: Iterable[float]) -> RemoteSessionInfo:
        return self._client.tell(self.session_id, values)

    def result(self) -> RemoteResult:
        return self._client.result(self.session_id)

    def checkpoint(self) -> str:
        return self._client.checkpoint(self.session_id)

    def delete(self) -> None:
        self._client.delete(self.session_id)


class ResolutiveClient:
    """Synchronous Python client for a Resolutive Computing server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 30.0,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "ResolutiveClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _detail(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            return response.text or "request failed"
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
        return str(payload)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise ResolutiveHTTPError(response.status_code, self._detail(response))
        return response

    @staticmethod
    def _info(payload: dict[str, Any]) -> RemoteSessionInfo:
        return RemoteSessionInfo(
            session_id=str(payload["session_id"]),
            done=bool(payload["done"]),
            evaluations=int(payload["evaluations"]),
            remaining=(None if payload.get("remaining") is None else int(payload["remaining"])),
            session_type=str(payload["session_type"]),
        )

    def health(self) -> bool:
        payload = self._request("GET", "/health").json()
        return payload.get("status") == "ok"

    def create(
        self,
        *,
        dimension: int,
        bounds: tuple[float, float],
        budget: int,
        seed: int = 0,
        mode: str = "multires",
        batch_size: int = 16,
    ) -> RemoteSession:
        payload = {
            "dimension": int(dimension),
            "bounds": [float(bounds[0]), float(bounds[1])],
            "budget": int(budget),
            "seed": int(seed),
            "mode": str(mode),
            "batch_size": int(batch_size),
        }
        data = self._request("POST", "/sessions", json=payload).json()
        return RemoteSession(self, str(data["session_id"]))

    def attach(self, session_id: str) -> RemoteSession:
        # Validate existence now so stale IDs fail early.
        self.session_info(session_id)
        return RemoteSession(self, session_id)

    def session_info(self, session_id: str) -> RemoteSessionInfo:
        data = self._request("GET", f"/sessions/{session_id}").json()
        return self._info(data)

    def ask(self, session_id: str) -> RemoteAskBatch:
        data = self._request("POST", f"/sessions/{session_id}/ask").json()
        points = np.asarray(data["points"], dtype=float)
        return RemoteAskBatch(points=points, generation=int(data["generation"]))

    def tell(self, session_id: str, values: Iterable[float]) -> RemoteSessionInfo:
        body = {"values": [float(value) for value in values]}
        data = self._request("POST", f"/sessions/{session_id}/tell", json=body).json()
        return self._info(data)

    def result(self, session_id: str) -> RemoteResult:
        data = self._request("GET", f"/sessions/{session_id}/result").json()
        return RemoteResult(
            x=np.asarray(data["x"], dtype=float),
            fun=float(data["fun"]),
            evaluations=int(data["evaluations"]),
            seed=int(data["seed"]),
            optimizer=str(data["optimizer"]),
            status=str(data["status"]),
            diagnostics=dict(data.get("diagnostics") or {}),
        )

    def checkpoint(self, session_id: str) -> str:
        data = self._request("GET", f"/sessions/{session_id}/checkpoint").json()
        return str(data["checkpoint"])

    def delete(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}")
