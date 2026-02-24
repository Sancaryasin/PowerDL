"""Torch high-level API (thin wrapper).

This module is kept for backward compatibility with earlier examples:

    from powerdl.highlevel_torch import profile_torch

The actual implementation lives in :mod:`powerdl.highlevel`.
"""

from __future__ import annotations

from .highlevel import PowerDLTorchProfiler as TorchProfiler
from .highlevel import profile_torch

__all__ = ["TorchProfiler", "profile_torch"]
