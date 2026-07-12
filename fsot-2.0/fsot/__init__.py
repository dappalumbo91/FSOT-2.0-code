"""FSOT 2.0 — monolithic interpretable runtime (Python + mpmath)."""

from __future__ import annotations

__version__ = "2.0.1"

from fsot.adapter import load_compute

__all__ = ["__version__", "load_compute"]