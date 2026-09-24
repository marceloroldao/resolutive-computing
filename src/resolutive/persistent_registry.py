"""Atomic disk-backed session registry for PC/server deployments.

Each session is stored as a versioned JSON checkpoint in its own file. Writes
use fsync + os.replace so a process crash cannot expose a partially-written
checkpoint as the current session state.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .api import create_session
from .checkpoint import checkpoint_json, restore_json
from .registry import SessionRegistry


class PersistentSessionRegistry(SessionRegistry):
    """SessionRegistry that automatically persists mutations to a directory."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load_all()

    @staticmethod
    def _digest(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{self._digest(session_id)}.json"

    def _fsync_directory(self) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _persist_locked(self, session_id: str) -> None:
        session = self._sessions[session_id]
        envelope = {
            "format": "resolutive-registry-entry",
            "version": 1,
            "session_id": session_id,
            "checkpoint": json.loads(checkpoint_json(session)),
        }
        target = self._path(session_id)
        tmp = self.directory / f".{target.name}.{uuid4().hex}.tmp"
        data = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            self._fsync_directory()
        finally:
            if tmp.exists():
                tmp.unlink()

    def _load_all(self) -> None:
        with self._lock:
            for path in sorted(self.directory.glob("*.json")):
                try:
                    envelope = json.loads(path.read_text(encoding="utf-8"))
                    if envelope.get("format") != "resolutive-registry-entry":
                        raise ValueError("unsupported registry entry format")
                    if envelope.get("version") != 1:
                        raise ValueError("unsupported registry entry version")
                    session_id = envelope.get("session_id")
                    checkpoint = envelope.get("checkpoint")
                    if not isinstance(session_id, str) or not session_id:
                        raise ValueError("invalid persisted session id")
                    if not isinstance(checkpoint, dict):
                        raise ValueError("invalid persisted checkpoint")
                    if self._path(session_id) != path:
                        raise ValueError("registry entry filename does not match session id")
                    if session_id in self._sessions:
                        raise ValueError(f"duplicate persisted session id: {session_id}")
                    self._sessions[session_id] = restore_json(
                        json.dumps(checkpoint, sort_keys=True, separators=(",", ":"))
                    )
                except Exception as exc:
                    raise ValueError(f"failed to load persisted session {path.name}: {exc}") from exc

    def create(self, *, session_id: str | None = None, **session_kwargs: Any) -> str:
        sid = session_id or uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session id already exists: {sid}")
            self._sessions[sid] = create_session(**session_kwargs)
            try:
                self._persist_locked(sid)
            except Exception:
                del self._sessions[sid]
                raise
        return sid

    def put(self, session: Any, *, session_id: str | None = None) -> str:
        sid = session_id or uuid4().hex
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"session id already exists: {sid}")
            self._sessions[sid] = session
            try:
                self._persist_locked(sid)
            except Exception:
                del self._sessions[sid]
                raise
        return sid

    def ask(self, session_id: str):
        with self._lock:
            session = self.get(session_id)
            batch = session.ask()
            self._persist_locked(session_id)
            return batch

    def tell(self, session_id: str, values) -> None:
        with self._lock:
            session = self.get(session_id)
            session.tell(values)
            self._persist_locked(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session id: {session_id}")
            path = self._path(session_id)
            if path.exists():
                path.unlink()
                self._fsync_directory()
            del self._sessions[session_id]
