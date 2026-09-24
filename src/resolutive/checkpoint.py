"""Versioned JSON-safe checkpoints for stateful optimization sessions.

The checkpoint format serializes only a strict whitelist of Resolutive session
classes plus JSON-compatible primitives, NumPy arrays/scalars and NumPy RNG
state.  It deliberately avoids pickle so checkpoint data can be validated
before objects are reconstructed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .hybrid_multires_sequential_session import HybridMultiResolutionSequentialSession
from .hybrid_sequential_session import HybridRegimeSequentialSession
from .session import OptimizationSession
from .specialist_sessions import V5Session
from .spiral_session import SpiralRefineSession
from .v2_sequential_session import V2SequentialSession

CHECKPOINT_FORMAT = "resolutive-session"
CHECKPOINT_VERSION = 1

_ALLOWED_CLASSES = {
    cls.__name__: cls
    for cls in (
        OptimizationSession,
        HybridMultiResolutionSequentialSession,
        HybridRegimeSequentialSession,
        V2SequentialSession,
        V5Session,
        SpiralRefineSession,
    )
}


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.random.Generator):
        return {
            "__type__": "rng",
            "bit_generator": value.bit_generator.__class__.__name__,
            "state": _encode(value.bit_generator.state),
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("checkpoint dictionaries must use string keys")
        return {k: _encode(v) for k, v in value.items()}

    cls_name = value.__class__.__name__
    if cls_name in _ALLOWED_CLASSES and isinstance(value, _ALLOWED_CLASSES[cls_name]):
        return {
            "__type__": "session_object",
            "class": cls_name,
            "state": {k: _encode(v) for k, v in value.__dict__.items()},
        }
    raise TypeError(f"unsupported checkpoint value: {type(value)!r}")


def _decode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        raise ValueError("invalid checkpoint value")

    tag = value.get("__type__")
    if tag is None:
        return {str(k): _decode(v) for k, v in value.items()}
    if tag == "ndarray":
        arr = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        expected = tuple(int(x) for x in value["shape"])
        if arr.shape != expected:
            raise ValueError("checkpoint array shape mismatch")
        return arr
    if tag == "tuple":
        return tuple(_decode(v) for v in value["items"])
    if tag == "rng":
        bitgen_name = value.get("bit_generator")
        bitgen_cls = getattr(np.random, str(bitgen_name), None)
        if bitgen_cls is None or not isinstance(bitgen_cls, type):
            raise ValueError(f"unsupported NumPy bit generator: {bitgen_name!r}")
        bitgen = bitgen_cls()
        bitgen.state = _decode(value["state"])
        return np.random.Generator(bitgen)
    if tag == "session_object":
        cls_name = value.get("class")
        cls = _ALLOWED_CLASSES.get(str(cls_name))
        if cls is None:
            raise ValueError(f"unsupported checkpoint class: {cls_name!r}")
        state = value.get("state")
        if not isinstance(state, dict):
            raise ValueError("invalid checkpoint object state")
        obj = cls.__new__(cls)
        obj.__dict__.update({str(k): _decode(v) for k, v in state.items()})
        return obj
    raise ValueError(f"unknown checkpoint type tag: {tag!r}")


def checkpoint_session(session: Any) -> dict[str, Any]:
    """Return a JSON-safe, versioned snapshot of a supported session."""
    root = _encode(session)
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "root": root,
    }


def restore_session(snapshot: dict[str, Any]) -> Any:
    """Restore a supported session from a validated checkpoint dictionary."""
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a dictionary")
    if snapshot.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if snapshot.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    session = _decode(snapshot.get("root"))
    if session.__class__.__name__ not in _ALLOWED_CLASSES:
        raise ValueError("checkpoint root is not a supported session")
    return session


def checkpoint_json(session: Any) -> str:
    """Serialize a supported session to canonical JSON text."""
    return json.dumps(checkpoint_session(session), sort_keys=True, separators=(",", ":"))


def restore_json(payload: str) -> Any:
    """Restore a supported session from checkpoint JSON text."""
    return restore_session(json.loads(payload))
