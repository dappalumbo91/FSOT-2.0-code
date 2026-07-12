"""Load the authoritative FSOT compute engine from the bundled package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MOD: Any = None


def load_compute():
    """Return the bundled fsot.compute module (singleton)."""
    global _MOD
    if _MOD is None:
        _MOD = import_module("fsot.compute")
    return _MOD


def domain_scalar(name: str) -> float:
    return float(load_compute().domain_scalar(name))