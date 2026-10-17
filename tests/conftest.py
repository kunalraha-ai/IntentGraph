from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def _can_symlink() -> bool:
    """Whether this machine can create symlinks at all (#2642).

    Probed rather than inferred from ``sys.platform``: Windows *can* create
    symlinks from an elevated shell or with Developer Mode enabled, and those
    runs should still get the coverage. A plain non-elevated Windows shell
    raises ``OSError: [WinError 1314] A required privilege is not held by the
    client``, which pytest reports as a FAILURE — 15 of them, drowning out real
    defects — when what it means is "unsupported here".

    One file symlink is enough to probe: Windows gates file and directory
    symlinks behind the same ``SeCreateSymbolicLinkPrivilege`` check.
    """
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe-src"
        src.write_text("x", encoding="utf-8")
        try:
            (Path(d) / "probe-link").symlink_to(src)
        except (OSError, NotImplementedError):
            return False
        return True


@pytest.fixture
def requires_symlinks(_can_symlink) -> None:
    """Skip a test that must create symlinks when the platform won't allow it.

    Take this as a parameter rather than wrapping each ``symlink_to()`` call in
    try/except: the guard then sits in the signature where it is visible, and
    an OSError from the code UNDER test is still a real failure instead of
    being swallowed into a skip.
    """
    if not _can_symlink:
        pytest.skip(
            "symlink creation unavailable on this machine "
            "(Windows requires an elevated shell or Developer Mode)"
        )


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path_factory, monkeypatch):
    """Every test gets a throwaway HOME so installers/uninstallers can never
    touch the developer's real ~/.claude, ~/.gemini, ~/.codebuddy, ~/.copilot,
    ~/.config, ~/.agents (issue #2168).

    Also the credentials/credentials-hermeticity guard for the IntentGraph
    provider surface: the ordinary suite must never read the developer's
    .env or carry provider keys into ambient configuration, even on a
    machine with real Nebius/Tavily credentials configured (Phase 3).
    Live-mode runs (INTENTGRAPH_LIVE_TESTS=1) are exempt: they are the
    explicit opt-in that is allowed to use real credentials.

    Allocated via tmp_path_factory (not inside tmp_path) so tests that assert
    the exact contents of their own tmp_path are unaffected."""
    home = tmp_path_factory.mktemp("sandbox-home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))              # Windows ntpath.expanduser
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)     # escape hatch that bypasses Path.home
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    # --- IntentGraph provider hermeticity (Phase 3) --------------------------
    # Live mode is exempt: it is the explicit opt-in for real credentials.
    if os.environ.get("INTENTGRAPH_LIVE_TESTS", "").strip() == "1":
        return home
    # Otherwise: the ordinary suite must neither read the developer's .env
    # nor inherit provider keys — it stays offline and credential-free no
    # matter what this machine has configured. Tests that deliberately set
    # a key do so with their own monkeypatch, which runs after this.
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
    return home

@pytest.fixture(autouse=True)
def _isolate_backend_env(monkeypatch):
    """detect_backend() probes the developer's real shell: with GOOGLE_API_KEY (or
    any of a dozen others) exported, the backend-detection tests picked that
    backend instead of the one the test set up (#3481). Clear every variable it
    reads before each test; a test that wants one sets it with monkeypatch."""
    from graphify.llm import backend_detection_env_vars

    for key in backend_detection_env_vars():
        monkeypatch.delenv(key, raising=False)


_ANALYZE_WARNING_FILTERS = (
    "ignore:Tensorflow not installed; ParametricUMAP will be unavailable:ImportWarning:umap",
    "ignore:Please import `random` from the `scipy\\.sparse` namespace.*:"
    "DeprecationWarning:hyppo\\.independence\\.hhg",
    "ignore:The keyword argument 'nopython=False' was supplied.*:Warning:numba\\.core\\.decorators",
)


def pytest_collection_modifyitems(items: list[Any]) -> None:
    for item in items:
        if item.path.name != "test_analyze.py":
            continue
        for warning_filter in _ANALYZE_WARNING_FILTERS:
            item.add_marker(pytest.mark.filterwarnings(warning_filter))
