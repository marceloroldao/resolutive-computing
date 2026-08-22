"""Optional HTTP transport for Resolutive Computing sessions.

This module targets PC/server deployments. The optimization core remains
transport-agnostic; FastAPI is an optional dependency exposed through the
``server`` extra.
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .registry import SessionRegistry


class CreateSessionRequest(BaseModel):
    dimension: int = Field(ge=1)
    bounds: tuple[float, float]
    budget: int = Field(ge=1)
    seed: int = 0
    mode: Literal["prototype", "multires", "hybrid"] = "multires"
    batch_size: int = Field(default=16, ge=1)


class TellRequest(BaseModel):
    values: list[float]


def _info_dict(registry: SessionRegistry, session_id: str) -> dict[str, object]:
    info = registry.info(session_id)
    return {
        "session_id": info.session_id,
        "done": info.done,
        "evaluations": info.evaluations,
        "remaining": info.remaining,
        "session_type": info.session_type,
    }


def create_app(registry: SessionRegistry | None = None) -> FastAPI:
    registry = registry or SessionRegistry()
    app = FastAPI(title="Resolutive Computing API", version="0.2-dev")
    app.state.registry = registry

    def missing(exc: KeyError) -> HTTPException:
        return HTTPException(status_code=404, detail=str(exc))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    def create_session(request: CreateSessionRequest) -> dict[str, object]:
        try:
            sid = registry.create(**request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _info_dict(registry, sid)

    @app.get("/sessions/{session_id}")
    def session_info(session_id: str) -> dict[str, object]:
        try:
            return _info_dict(registry, session_id)
        except KeyError as exc:
            raise missing(exc) from exc

    @app.post("/sessions/{session_id}/ask")
    def ask(session_id: str) -> dict[str, object]:
        try:
            batch = registry.ask(session_id)
        except KeyError as exc:
            raise missing(exc) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "generation": int(batch.generation),
            "points": batch.points.tolist(),
        }

    @app.post("/sessions/{session_id}/tell")
    def tell(session_id: str, request: TellRequest) -> dict[str, object]:
        try:
            registry.tell(session_id, request.values)
            return _info_dict(registry, session_id)
        except KeyError as exc:
            raise missing(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/result")
    def result(session_id: str) -> dict[str, object]:
        try:
            value = registry.result(session_id)
        except KeyError as exc:
            raise missing(exc) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "x": value.x.tolist(),
            "fun": float(value.fun),
            "evaluations": int(value.evaluations),
            "seed": int(value.seed),
            "optimizer": value.optimizer,
            "status": value.status,
            "diagnostics": value.diagnostics,
        }

    @app.get("/sessions/{session_id}/checkpoint")
    def checkpoint(session_id: str) -> dict[str, str]:
        try:
            return {"checkpoint": registry.checkpoint(session_id)}
        except KeyError as exc:
            raise missing(exc) from exc

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        try:
            registry.delete(session_id)
        except KeyError as exc:
            raise missing(exc) from exc

    return app
