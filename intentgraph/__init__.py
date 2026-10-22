"""IntentGraph — neural memory and retrieval intelligence for AI systems.

IntentGraph gives an AI system a persistent, connected, evolving memory.
It stores original evidence (records) with provenance, connects them with a
typed relationship graph, retrieves them through a hybrid lexical /
semantic / graph pipeline, iterates until the evidence covers the question,
and compiles a token-budgeted, source-referenced context packet with
explicit limitations.

Layering:

    intentgraph.models     data model: records, edges, relations, provenance
    intentgraph.store      SQLite persistence, versioning, export/import
    intentgraph.ingest     safe ingestion, type inference, auto-links, dedup
    intentgraph.lexindex   pure-Python BM25 lexical index
    intentgraph.embeddings embedding-provider interface + local/hash/remote impls
    intentgraph.retrieve   retrievers, RRF fusion, ranking transparency
    intentgraph.verify     evidence verification: supersession, conflicts, gaps
    intentgraph.controller adaptive memory-activation engine (iterative loop)
    intentgraph.context    token-budgeted evidence-packet compiler
    intentgraph.codegraph  native codebase-graph engine (scan → parse → resolve → export)
    intentgraph.graphbridge expansion into a code-graph graph.json via anchors
    intentgraph.config     env + file configuration, privacy controls
    intentgraph.sdk        IntentGraph facade (the programmatic entry point)
    intentgraph.api        local-first JSON HTTP API (stdlib, no new deps)
    intentgraph.cli        `intentgraph` command line
    intentgraph.eval       reproducible benchmark with known-evidence fixtures

Everything deterministic (storage, lexical retrieval, graph traversal, evidence
checks, context compilation, codegraph builds) works with no configured model.
Embeddings, reranking and LLM-assisted planning are opt-in through provider
interfaces.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy imports so `import intentgraph` stays
    # cheap for callers that only need one subsystem.
    _map = {
        "MemoryRecord": ("intentgraph.models", "MemoryRecord"),
        "MemoryEdge": ("intentgraph.models", "MemoryEdge"),
        "MemoryStore": ("intentgraph.store", "MemoryStore"),
        "ingest": ("intentgraph.ingest", "ingest"),
        "BM25Index": ("intentgraph.lexindex", "BM25Index"),
        "IntentGraph": ("intentgraph.sdk", "IntentGraph"),
        "AdaptiveMemoryEngine": ("intentgraph.controller", "AdaptiveMemoryEngine"),
        "Config": ("intentgraph.config", "Config"),
    }
    if name in _map:
        import importlib

        mod_name, attr = _map[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'intentgraph' has no attribute {name!r}")
