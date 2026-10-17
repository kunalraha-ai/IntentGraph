"""Shared IntentGraph test scenario — re-exports the canonical package corpus.

The scenario lives in ``intentgraph/scenario.py`` (product code: the
benchmark runs it) so tests and the benchmark share one corpus with zero
drift. This module keeps the historical test import path working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intentgraph.scenario import seed  # noqa: F401
from intentgraph.sdk import IntentGraph


def build(tmp_path, **overrides: Any) -> tuple[IntentGraph, dict[str, str]]:
    """Fresh IntentGraph over the seeded scenario (temp data dir)."""
    ig = IntentGraph(data_dir=str(Path(tmp_path) / "ig-data"), **overrides)
    ids = seed(ig)
    return ig, ids
