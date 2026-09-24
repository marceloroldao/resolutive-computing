"""Session registry for stateful Resolutive ask/tell optimization.

The registry is intentionally transport-agnostic. It is the state-management
layer that a future HTTP API can delegate to without coupling optimization
sessions to FastAPI, databases, or process-global request objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from .api import create_session
from .checkpoint import checkpoint_json, restore_json


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    done: bool
    evaluations: int
    remaining: int | None
    session_type: str


class SessionRegistry:
    """Thread-safe in-memory registry of independent optimization sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._lock = RLock()

    def create(self, *, session_id: str | None = None, **session_kwargs: Any) -> str:
        sid = session_id or uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session id already exists: {sid}")
            self._sessions[sid] = create_session(**session_kwargs)
        return sid

    def put(self, session: Any, *, session_id: str | None = None) -> str:
        sid = session_id or uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session id already exists: {sid}")
            self._sessions[sid] = session
        return sid

    def get(self, session_id: str) -> Any:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session id: {session_id}") from exc

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session id: {session_id}")
            del self._sessions[session_id]

    def ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sessions.keys())

    def info(self, session_id: str) -> SessionInfo:
        session = self.get(session_id)
        return SessionInfo(
            session_id=session_id,
            done=bool(session.done),
            evaluations=int(getattr(session, "evaluations", 0)),
            remaining=(int(session.remaining) if hasattr(session, "remaining") else None),
            session_type=session.__class__.__name__,
        )

    def ask(self, session_id: str):
        session = self.get(session_id)
        return session.ask()

    def tell(self, session_id: str, values) -> None:
        session = self.get(session_id)
        session.tell(values)

    def result(self, session_id: str):
        session = self.get(session_id)
        return session.result()

    def checkpoint(self, session_id: str) -> str:
        return checkpoint_json(self.get(session_id))

    def restore(self, payload: str, *, session_id: str | None = None) -> str:
        return self.put(restore_json(payload), session_id=session_id)
