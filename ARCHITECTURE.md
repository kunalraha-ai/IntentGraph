# Architecture

IntentGraph is a Python library with four surfaces over one core: the SDK (`intentgraph.sdk.IntentGraph`), a CLI (`intentgraph`), a local HTTP API (`intentgraph.api`), and MCP agent tools (`intentgraph.mcp` + `intentgraph.serve`). One package (`intentgraph/`), one storage engine (SQLite). There is no dependency on any external code-graph engine: IntentGraph *builds* code graphs natively (`intentgraph/codegraph/`), and the optional code-graph bridge (`intentgraph.graphbridge`) also accepts externally produced `graph.json` **data files** via networkx — nothing more.

## Pipeline

```
ingest → store → retrieve (exact/lexical/semantic/graph, RRF) → verify → adaptive controller → compile_context
```

Stages communicate through plain dataclasses, dicts, and SQLite rows. Local-first: every stage is deterministic and offline unless a remote provider is explicitly configured.

## IntentGraph layer

Signatures below are the real ones — `tests/test_architecture_doc.py` imports every lowercase symbol named here, so this table cannot drift from the code.

| Module | Entry point(s) | Input → Output |
|--------|----------------|----------------|
| `models.py` | `MemoryRecord`, `MemoryEdge`, `Feedback` | raw fields → validated memory records/edges (stable ids, content hash, provenance, validity) |
| `store.py` | `MemoryStore` | records/edges → SQLite archive with append-only version history, soft delete, tombstones |
| `ingest.py` | `ingest`, `ingest_file`, `ingest_dict`, `Ingestor` | untrusted text → stored record + auto-links (mentions, MinHash near-duplicates) |
| `lexindex.py` | `tokenize`, `BM25Index` | text → stemmed postings; query → ranked candidates with per-term transparency |
| `embeddings.py` | `provider_from_config`, `SemanticIndex` | config → embedding provider (none/hash/nebius/nvidia/openai-compatible); text → stored vectors → cosine hits |
| `websearch.py` | `provider_from_config`, `TavilySearchProvider`, `parse_provider_date`, `store_external_result` | config → web-search provider (none/tavily); query → ExternalEvidence with full provenance (never a user memory) |
| `llmassist.py` | `llm_policy_from_config`, `LLMPolicy` | question/coverage report → schema-validated LLM plans (requirements, action selection) with deterministic fallback |
| `rerank.py` | `reranker_from_config`, `NebiusReranker` | (query, excerpts) → relevance order over Token Factory's /v1/rerank; ordering-only, classified failures |
| `retrieve.py` | `fuse_rrf`, `RetrievalEngine` | query + indexes → fused candidates (RRF over exact/lexical/semantic + bounded graph expansion) |
| `verify.py` | `analyze_requirements`, `topical_terms`, `verify_evidence` | selected records → checks (provenance, validity, temporal), conflicts, coverage, verdict |
| `controller.py` | `AdaptiveMemoryEngine` (`.run`) | question → iterative retrieve→expand→verify loop with gap-targeted follow-ups and stopping conditions |
| `context.py` | `compile_context`, `estimate_tokens` | verified evidence → token-budgeted packet (dict + text) with untrusted-evidence fencing |
| `graphbridge.py` | `GraphifyBridge` | `graph.json` + records with `graph_anchors` → code-adjacent record candidates |
| `codelink.py` | `add_explicit_link`, `extract_links_for_record`, `refresh_links`, `suggest_links` | record text + code graph → verified/candidate links (explicit · extracted · candidate tiers; candidates are labelled suggestions, never evidence) |
| `codequery.py` | `CodeIndex`, `combined_retrieve` | one query → memories (existing hybrid) + code entities (deterministic symbol/path search) + verified-link traversal with full provenance |
| `replay.py` | `generate_replay` | query → ordered evidence-replay script (focal node, supporting trail, code links, verdict — each step with narration for the voice guide) |
| `viz.py` | `build_memory_graph`, `retrieval_overlay`, `write_memory_html` | real archive records/edges + a REAL retrieve() call → nx.Graph + highlight overlay consumed by the IntentGraph renderer |
| `renderer.py` | `to_html` | memory nx.Graph + overlay → self-contained interactive graph.html (viz-network, evidence panel, retrieval highlight/dim/reset, honest empty states) |
| `minhash.py` | `MinHash`, `_jaccard` | shingles → deterministic near-duplicate sketches (sha-256 slot constants, splitmix64 mixing — no process RNG) |
| `tokens.py` | `estimate_tokens`, `token_method`, `neutralise_injection_sentinels`, `get_tokenizer` | text → token estimates (tiktoken cl100k_base or documented chars/4) and defanged chat-template sentinels |
| `serve.py` | `main`, `serve`, `_build_server` | MCP stdio server: the four memory tools + usage instructions, mcp 1.x/2.x dual-compat, per-call errors never kill the process |
| `config.py` | `load_config` | defaults + JSON file + env + args → resolved `Config` (privacy hard-stop included) |
| `sdk.py` | `IntentGraph` | everything above behind one facade (`ingest`, `retrieve`, `remember`, `export`, `status`, ...) |
| `scenario.py` | `seed` | empty archive → canonical benchmark corpus (shared with tests) |
| `eval.py` | `run_benchmark` | — → deterministic benchmark report (recall/precision/abstention/tokens) |
| `api.py` | `serve`, `make_server` | resolved Config → local JSON HTTP API (`/v1/memories`, `/v1/retrieve`, `/v1/context`, `/v1/feedback`, `/health`) |
| `cli.py` | `main`, `build_parser` | argv → subcommands (console script `intentgraph` / `python -m intentgraph`) |
| `mcp.py` | `tool_schemas`, `tool_remember`, `tool_recall`, `tool_context`, `tool_feedback`, `IntentGraphSessions` | validated agent-tool args (registered by `intentgraph.serve` as `remember`/`recall`/`context`/`feedback`) → SDK calls → compact JSON / evidence-packet text |

The table follows a no-drift rule: every lowercase symbol it names is importable (guarded by `tests/test_architecture_doc.py`).

## Codegraph engine (`intentgraph/codegraph/`)

A native, independently written codebase-graph engine: scan a repository, parse its source (Python, via the standard library `ast` — analyzed code is never executed), resolve imports/calls/inheritance statically, and emit a typed, evidence-backed graph. No external code-graph package is installed or imported.

Pipeline: `scan → parse → index symbols → resolve references → build graph → analyze → export`.

| Module | Entry point(s) | Input → Output |
|--------|----------------|----------------|
| `schema.py` | `file_id`, `directory_id`, `symbol_id_part`, `IdRegistry` | repository-relative path / symbol qualname → stable flattened node ids (with deterministic collision suffixing) |
| `scan.py` | `scan_repository`, `ScanOptions` | repository root → deterministic file list + honest skip records (gitignore subset, built-in excludes, secret-name skip, size/count limits, symlinks never followed) |
| `python_parser.py` | `PythonParser`, `PARSERS`, `module_name_for`, `is_test_path` | source text → ParsedModule (definitions, imports, call sites, name uses, entry-point guard); `LanguageParser` protocol is the extension point for other languages |
| `build.py` | `build_graph`, `BuildOptions` | scan + parses → typed nx.MultiDiGraph (EXTRACTED vs INFERRED evidence; unresolved/external imports recorded, never edged) |
| `analyze.py` | `load_graph`, `graph_stats`, `import_cycles`, `find_paths`, `neighborhood`, `node_report`, `orphan_candidates`, `components` | graph.json → stats, bounded cycles/hubs/orphan-candidates, deterministic paths, bounded neighborhoods |
| `export.py` | `to_node_link`, `write_json`, `write_graphml`, `communities_by_directory` | build → deterministic bridge/renderer-compatible node-link JSON (`graph.schema: intentgraph-codegraph` v1) or GraphML |
| `cli.py` | `register`, `run` | argv → `intentgraph codegraph build/refresh/stats/show/path/render` |

**Node types:** `repository`, `directory`, `module` (non-test `.py`), `test` (test module), `class`, `function`, `method` (`file` is reserved for future parsers whose files are not importable modules). **Edge types:** `contains` (repo→dir→module), `defines` (module→symbol), `imports` (module→module, resolved in-repo only), `calls` / `references` / `inherits` (symbol→symbol, statically proven), `tests` (test module→tested module; EXTRACTED via imports, INFERRED via naming convention). Everything the resolver cannot prove is metadata (per-node `imports_unresolved`, `imports_external` lists) and counters — never an invented edge.

**Id scheme** (the bridge's documented anchor contract): repository-relative path, lowercased, extension stripped, `/` and `.` flattened to `_` — `intentgraph/models.py` → `intentgraph_models`; symbols append the lowercased dotted qualname — `MemoryStore.add_edge` in `intentgraph/store.py` → `intentgraph_store_memorystore_add_edge`. Ids are content-addressed by the tree, never by machine paths or scan order; they change when the path/qualname changes, and rare flatten-collisions get deterministic `__2`-style suffixes.

**Cross-graph linkage** (the memory side): the archive's `code_links` table (store schema v2) wires memories to code nodes in three verification tiers — `explicit` (user-asserted via `codelink add`), `extracted` (a deterministic matcher found a real backticked/dotted/file-path reference in the record's own text; runs automatically at ingest and on `codelink refresh`), and `candidate` (semantic suggestion via `codelink suggest`, labelled and never followed as retrieval evidence). Links are separate from record content, so code-graph rebuilds can never rewrite historical evidence. `intentgraph query` / the `recall` tool's `include_code` flag retrieve both graphs at once: memories through the unchanged hybrid path, code entities through `CodeIndex` (exact/structural symbol-path search), verified links followed in both directions, every entry carrying source paths, line numbers, memory ids, and link provenance. `viz --with-code` renders the combined graph (verified links green, candidates dashed gray, retrieval highlighting only what the query actually returned).

## Confidence labels

| Label | Meaning |
|-------|---------|
| `EXTRACTED` | The relationship is explicitly declared (user-stated link, a `supersedes` operation, an id mentioned in content) |
| `INFERRED` | The relationship was deduced (MinHash near-duplicate, chronology, type inference); it carries the recorded `basis` and renders dashed |

## Security

Memory content is untrusted input, end to end:

- **Ingest**: strict id validation, size caps, type checks — malformed records are rejected before they land in the store.
- **Packets**: evidence bodies render inside untrusted-evidence fences; known chat-template injection sentinels are defanged (`tokens.neutralise_injection_sentinels`).
- **API/MCP**: request body caps, path containment, project-scoped archives, no stack traces in errors.
- **Credentials**: `.env` only (never config files, never logs); `privacy.local_only: true` refuses every remote provider class at configuration load — before any request can be made.
- **Renderer**: embedded JSON is `</`-escaped (no script-tag breakout), all labels/fields sanitized and rendered through an esc() boundary.

## Testing

One test file per module under `tests/`. Run with:

```bash
pytest tests/ -q
```

The suite is credential-hermetic and offline: `tests/conftest.py` disables `.env` loading and strips provider keys for every test (live provider smokes are the explicit `INTENTGRAPH_LIVE_TESTS=1` opt-in). Docs are guarded too — the module table above is enforced by `tests/test_architecture_doc.py`.
