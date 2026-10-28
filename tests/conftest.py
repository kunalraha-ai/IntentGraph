"""Test-suite fixtures.

One autouse fixture carries the correctness load for the whole suite:
provider hermeticity. The ordinary suite must never read the developer's
.env nor inherit provider credentials, even on a machine with real
Nebius/NVIDIA/Tavily keys configured — tests stay offline and green
everywhere. Live runs (INTENTGRAPH_LIVE_TESTS=1) are the explicit opt-in
that may use real credentials; the live test file loads .env itself.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_providers(monkeypatch):
    """Disable .env loading and strip provider keys for every test.

    Tests that deliberately set a key do so with their own monkeypatch,
    which runs after this one.
    """
    if os.environ.get("INTENTGRAPH_LIVE_TESTS", "").strip() == "1":
        return
    monkeypatch.setenv("INTENTGRAPH_DISABLE_DOTENV", "1")
    for key in list(os.environ):
        is_provider_config = (
            key.startswith("INTENTGRAPH_")
            and any(s in key for s in ("EMBEDDING", "LLM", "RERANKER",
                                       "WEB_SEARCH", "ENV_FILE")))
        is_credential = key in ("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY",
                                "NVIDIA_API_KEY", "TAVILY_API_KEY",
                                "OPENAI_API_KEY")
        if is_provider_config or is_credential:
            monkeypatch.delenv(key, raising=False)
