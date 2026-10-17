# Architecture

graphify is a Claude Code skill backed by a Python library. The skill orchestrates the library; the library can be used standalone.

## Pipeline

```
detect()  →  extract()  →  build()  →  cluster()  →  analyze helpers  →  report.generate()  →  export.to_*()
```

Each stage lives in its own module and they communicate through plain Python dicts and NetworkX graphs. While designed to minimize shared state, some stages (like extractors) rely on ambient module-level state around the active scan root. Most stages are a single function; `analyze.py` and `export.py` are sets of sibling functions rather than one entry point.

## Module responsibilities

Signatures below are the real ones - `tests/test_architecture_doc.py` imports every symbol named here, so this table cannot drift from the code.

| Module | Entry point(s) | Input → Output |
|--------|----------------|----------------|
| `detect.py` | `detect(root)` | directory → scan summary dict: `files` grouped by category, plus `total_files`, `total_words`, `warning`, `scan_root`, … |
| `extract.py` | `extract(paths, *, root=None, ...)`, `collect_files(target)` | **list** of file paths → `{nodes, edges}` dict. `collect_files` expands a directory into that list, and lives here, not in `detect.py` |
| `build.py` | `build(extractions)`, `build_from_json(extraction)` | extraction dict(s) → `nx.Graph` |
| `cluster.py` | `cluster(G)` | graph → `{community_id: [node_id, ...]}` (the graph is not mutated) |
| `analyze.py` | `god_nodes(G)`, `surprising_connections(G)`, `suggest_questions(G, communities, community_labels)`, `find_import_cycles(G)`, `graph_diff(G_old, G_new)` | graph → one list/dict per analysis. There is no single `analyze()` entry point |
| `report.py` | `generate(G, communities, cohesion_scores, community_labels, ...)` | graph + analysis → GRAPH_REPORT.md string |
| `export.py` | `to_json`, `to_html`, `to_obsidian`, `to_svg`, `to_graphml`, `to_canvas`, `to_cypher` | graph → graph.json, graph.html, Obsidian vault, graph.svg, … one function per format |
| `wiki.py` | `to_wiki(G, communities, output_dir, ...)` | graph → one markdown article per community + `index.md` |
| `callflow_html.py` | `write_callflow_html(...)` | graphify-out files → Mermaid architecture/call-flow HTML |
| `ingest.py` | `ingest(url, target_dir, ...)` | URL → file saved to corpus dir |
| `cache.py` | `check_semantic_cache(files, root)`, `save_semantic_cache(nodes, edges, ...)` | files → cached nodes / edges / hyperedges + the list of files still needing extraction |
| `security.py` | `validate_url`, `safe_fetch`, `validate_graph_path`, `sanitize_label` | URL / path / label → validated value, or raises |
| `validate.py` | `validate_extraction(data)`, `assert_valid(data)` | extraction dict → **list of schema error strings** (`validate_extraction` returns them; `assert_valid` raises) |
| `serve.py` | `serve(graph_path)`, `serve_http(graph_path, *, host, port, ...)` | graph file path → MCP stdio server / HTTP server |
| `watch.py` | `watch(watch_path, debounce=3.0)`, `check_update(watch_path)` | directory → rebuild on change; `check_update` reports whether a re-extraction is pending |
| `benchmark.py` | `run_benchmark(graph_path)` | graph file → corpus vs subgraph token comparison |

### Calling `extract()` from your own code

`extract()` takes a **list** of paths, and `root` is keyword-only and optional:

```python
from pathlib import Path
from graphify.extract import extract

paths = [Path("src/lib/content.ts"), Path("src/pages/index.astro")]
result = extract(paths, root=Path(".").resolve())   # pass root explicitly
```

Always pass `root`. Node ids and `source_file` values are derived relative to it; when it is omitted, `extract()` infers one from the paths you passed, which is the common parent of *that list* rather than your project root. A single-file call therefore anchors to that file's own directory, and ids can end up carrying path segments from the machine they were extracted on.

## Extraction output schema

Every extractor returns:

```json
{
  "nodes": [
    {"id": "unique_string", "label": "human name", "source_file": "path", "source_location": "L42"}
  ],
  "edges": [
    {"source": "id_a", "target": "id_b", "relation": "calls|imports|uses|...", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS"}
  ]
}
```

`validate.py` enforces this schema before `build()` consumes it.

## Confidence labels

| Label | Meaning |
|-------|---------|
| `EXTRACTED` | Relationship is explicitly stated in the source (e.g., an import statement, a direct call) |
| `INFERRED` | Relationship is a reasonable deduction (e.g., call-graph second pass, co-occurrence in context) |
| `AMBIGUOUS` | Relationship is uncertain; flagged for human review in GRAPH_REPORT.md |

## Adding a new language extractor

1. Add an `extract_<lang>(path: Path) -> dict` function following the existing pattern (tree-sitter parse → walk nodes → collect `nodes` and `edges` → call-graph second pass for INFERRED `calls` edges). New languages go in their own module under `graphify/extractors/` - see `graphify/extractors/MIGRATION.md`; `extract.py` re-exports them while the existing ones are ported out of it.
2. Register the file suffix in `extract()`'s dispatch table and in `collect_files()` (both in `extract.py`).
3. Add the suffix to `CODE_EXTENSIONS` in `detect.py` and `_WATCHED_EXTENSIONS` in `watch.py`.
4. Add the tree-sitter package to `pyproject.toml` dependencies.
5. Add a fixture file to `tests/fixtures/` and tests to `tests/test_languages.py`.

## Security

All external input passes through `graphify/security.py` before use:

- URLs → `validate_url()` (http/https only) + `_NoFileRedirectHandler` (blocks file:// redirects)
- Fetched content → `safe_fetch()` / `safe_fetch_text()` (size cap, timeout)
- Graph file paths → `validate_graph_path()` (must resolve inside `graphify-out/`)
- Node labels → `sanitize_label()` (strips control chars, caps 256 chars)

See `SECURITY.md` for the full threat model.

## Testing

One test file per module under `tests/`. Run with:

```bash
pytest tests/ -q
```

The test suite is designed to avoid network access and uncontrolled filesystem effects; most tests are isolated with temporary directories and environment fixtures.

## IntentGraph layer (`intentgraph/`)

IntentGraph — persistent memory and retrieval intelligence for AI systems — is a
separate top-level package in this same repo. One-way dependency:
`intentgraph` imports `graphify` (security helpers, MinHash near-dup machinery,
the tiktoken tokenizer helper, node-link graph-loading conventions); `graphify`
never imports `intentgraph`. Full guide: `docs/intentgraph.md`.

Pipeline: `ingest → store → retrieve (lexical/semantic/graph, RRF) → verify →
adaptive controller (gap-targeted iterations) → compile_context (token-budgeted
evidence packet)`.

| Module | Entry point(s) | Input → Output |
|--------|----------------|----------------|
| `models.py` | `MemoryRecord`, `MemoryEdge`, `Feedback` | raw fields → validated memory records/edges (stable ids, content hash, provenance, validity) |
| `store.py` | `MemoryStore` | records/edges → SQLite archive with append-only version history, soft delete, tombstones |
| `ingest.py` | `ingest`, `ingest_file`, `ingest_dict`, `Ingestor` | untrusted text → stored record + auto-links (mentions, MinHash near-duplicates) |
| `lexindex.py` | `tokenize`, `BM25Index` | text → stemmed postings; query → ranked candidates with per-term transparency |
| `embeddings.py` | `provider_from_config`, `SemanticIndex` | config → embedding provider (none/hash/nebius/openai-compatible); text → stored vectors → cosine hits |
| `websearch.py` | `provider_from_config`, `TavilySearchProvider`, `parse_provider_date`, `store_external_result` | config → web-search provider (none/tavily); query → ExternalEvidence with full provenance (never a user memory) |
| `llmassist.py` | `llm_policy_from_config`, `LLMPolicy` | question/coverage report → schema-validated LLM plans (requirements, action selection) with deterministic fallback |
| `rerank.py` | `reranker_from_config`, `NebiusReranker` | (query, excerpts) → relevance order over Token Factory's /v1/rerank; ordering-only, classified failures |
| `retrieve.py` | `fuse_rrf`, `RetrievalEngine` | query + indexes → fused candidates (RRF over exact/lexical/semantic + bounded graph expansion) |
| `verify.py` | `analyze_requirements`, `topical_terms`, `verify_evidence` | selected records → checks (provenance, validity, temporal), conflicts, coverage, verdict |
| `controller.py` | `AdaptiveMemoryEngine` (`.run`) | question → iterative retrieve→expand→verify loop with gap-targeted follow-ups and stopping conditions |
| `context.py` | `compile_context`, `estimate_tokens` | verified evidence → token-budgeted packet (dict + text) with untrusted-evidence fencing |
| `graphbridge.py` | `GraphifyBridge` | `graph.json` + records with `graph_anchors` → code-adjacent record candidates |
| `config.py` | `load_config` | defaults + JSON file + env + args → resolved `Config` (privacy hard-stop included) |
| `sdk.py` | `IntentGraph` | everything above behind one facade (`ingest`, `retrieve`, `remember`, `export`, `status`, ...) |
| `scenario.py` | `seed` | empty archive → canonical benchmark corpus (shared with tests) |
| `eval.py` | `run_benchmark` | — → deterministic benchmark report (recall/precision/abstention/tokens) |
| `api.py` | `serve`, `make_server` | resolved Config → local JSON HTTP API (`/v1/memories`, `/v1/retrieve`, `/v1/context`, `/v1/feedback`, `/health`) |
| `cli.py` | `main`, `build_parser` | argv → subcommands (also reachable as `graphify intent ...`) |
| `mcp.py` | `tool_schemas`, `tool_remember`, `tool_recall`, `tool_context`, `tool_feedback`, `IntentGraphSessions` | validated agent-tool args (from the graphify MCP server's registered `remember`/`recall`/`context`/`feedback` tools) → SDK calls → compact JSON / evidence-packet text |

The IntentionGraph table follows the same no-drift rule: every lowercase symbol
it names is importable (guarded by `tests/test_architecture_doc.py`).
