# IntentGraph

**Neural memory and retrieval intelligence for AI systems — built on Graphify.**

IntentGraph gives an AI system a persistent, connected, evolving memory. Instead of
merely retrieving documents, it activates related experiences, follows meaningful
relationships, detects missing evidence, surfaces contradictions, and compiles a
compact, evidence-backed context packet — with the source of every remembered fact.

Local-first: the default configuration makes **zero network calls**. Lexical
retrieval, graph traversal, evidence verification, and adaptive activation are fully
deterministic and need no model, no API key, and no cloud service. Embeddings and
LLM assistance are opt-in through replaceable provider interfaces.

---

## Quickstart

```bash
# in this repo
uv sync
.venv/bin/python -m intentgraph ingest --text "ADR-14: order amounts are stored as integer minor units." --type decision
.venv/bin/python -m intentgraph remember "what did we decide about order amounts?"

# or through the graphify CLI you already have
.venv/bin/python -m graphify intent ingest --text "..."
.venv/bin/python -m graphify intent remember "..."

# once installed from the wheel / PyPI package:
intentgraph ingest --text "..."
intentgraph remember "..."
intentgraph api            # local HTTP API on 127.0.0.1:8710
```

The memory archive lives in `./intentgraph-data/` (override with `--home` or
`INTENTGRAPH_HOME`). It is a single SQLite file plus rebuildable indexes — back it
up by copying the directory, or export a portable bundle:

```bash
intentgraph export backup.jsonl      # versioned, provenance-complete JSONL bundle
intentgraph import backup.jsonl      # idempotent restore
```

## What it does

Ask the flagship question — *"Fix the regression. Remember what we tried before,
why it failed, and which architecture decision we made afterward"* — and IntentGraph:

1. **Analyzes** the question into evidence requirements (failure evidence, attempted
   fixes, why they failed, the decision).
2. **Retrieves** a broad first round: exact ids, BM25 lexical, semantic (when
   configured), fused with weighted Reciprocal Rank Fusion.
3. **Expands** through the memory graph's high-signal relations
   (`caused_by`, `failed_because_of`, `validated_by`, `supersedes`, `derived_from`).
4. **Verifies** the working set: provenance, validity (supersession), temporal
   consistency, explicit support, declared contradictions, near-duplicates.
5. **Targets the gaps**: unmet requirements trigger follow-up retrieval sharpened
   toward exactly that missing evidence shape — never a blind neighborhood crawl.
6. **Stops** when covered, out of iterations, out of budget, over time, or making
   no progress — and says which.
7. **Compiles** a token-budgeted context packet: evidence with per-entry provenance
   and retrieval methods, relationships, contradictions, limitations, unresolved
   questions, and an explicit verdict (`sufficient` / `partial` / `insufficient`).

It never fabricates evidence. If the archive does not answer the question, the
packet says so and names the gap — a consuming model can abstain instead of guessing.

## The memory model

Every ingested memory is a **record** with a stable id (`mem_<16hex>`), original
content (stored verbatim — derived data never replaces it), content hash, type
(`conversation|fact|decision|action|experience|document|code_reference|task|preference|project_state`),
source/provenance, event time *and* ingestion time, tags, and validity window.

- **Versioning is append-only**: updating a record snapshots the previous version;
  history is always readable (`get <id> --version 2`).
- **Deletion is soft by default** (`delete`), reversible (`restore`); `--purge` is
  the retention control and leaves a tombstone so surviving references stay
  explainable.
- **Supersession** (`supersede old new`) marks validity AND records a `supersedes`
  edge — the verifier then demotes the old record and points at its successor.

Records connect through typed, directed **edges** — `related_to, derived_from,
caused_by, contradicts, supersedes, depends_on, occurred_before/after, validated_by,
failed_because_of, belongs_to, duplicate_of, mentions`. Edges are explicit or
inferred (with the recorded basis), mirroring Graphify's EXTRACTED/INFERRED edge
discipline. Mentions and near-duplicates are linked automatically at ingest (the
near-dup detector reuses Graphify's MinHash machinery — near-duplicates are
**kept distinct and linked**, never silently merged).

## Hybrid + adaptive retrieval

| Mode | What runs |
|---|---|
| `lexical` | BM25 (pure Python, deterministic, identifier-aware, lightly stemmed) |
| `semantic` | cosine over embeddings — *requires a configured provider*, falls back with a note |
| `graph` | exact-id seeds + typed-edge expansion (bounded depth/fan-out) |
| `hybrid` | exact + lexical + semantic fused (RRF), then graph expansion of the fused seeds |
| `adaptive` | the full controller: iterative, gap-targeted, verified (the `remember` command) |

Fusion never drops single-source evidence and records every retriever that found
each memory. Expansion weights per relation are **documented heuristics** (causal
edges carry more weight than grouping edges), not learned parameters.

## Embedding providers (all optional)

| Provider | What it is | Configure |
|---|---|---|
| `none` (default) | semantic disabled; everything else works | — |
| `hash` | deterministic local n-gram projection — **not a neural embedding**; a test/deterministic-fallback path that runs the whole pipeline offline | `INTENTGRAPH_EMBEDDING_PROVIDER=hash` |
| `nebius` | Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1/`) — a first-class alias over the OpenAI-compatible client with documented credential precedence (`NEBIUS_API_KEY` → `NEBIUS_TOKEN_FACTORY_KEY`); model listing via `intentgraph embed --list-models`; model-change detection and safe reindex | see "External providers" below |
| `nvidia` | NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, `NVIDIA_API_KEY` with `nvapi-*` keys) — same OpenAI-compatible client; home of the NVIDIA-built embedding models (e.g. `nvidia/nemotron-3-embed-1b`) that Token Factory may not serve — this account is exactly that split | see "External providers" below |
| `openai` | any OpenAI-compatible `/embeddings` endpoint: OpenAI, **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1` + `nvapi-*` key), **Ollama** (`http://localhost:11434/v1`), vLLM, LiteLLM proxies | see below |

```bash
export INTENTGRAPH_EMBEDDING_PROVIDER=openai
export INTENTGRAPH_EMBEDDING_BASE_URL=http://localhost:11434/v1   # or NVIDIA/OpenAI/...
export INTENTGRAPH_EMBEDDING_MODEL=nomic-embed-text                # required, no default
export INTENTGRAPH_EMBEDDING_API_KEY=...            # or OPENAI_API_KEY
intentgraph embed                                    # (re)compute + persist vectors
```

Never hardcode keys. Nothing is ever sent anywhere unless you explicitly configured
the endpoint above — and then only record text goes to that endpoint (no ids, no
provenance, no metadata).

**Privacy hard-stop**: `"privacy": {"local_only": true}` in the config file refuses
to construct any provider with a non-loopback endpoint, any LLM planning provider,
any remote reranker, and any enabled web search.

## Configuration

Resolution order: defaults → `intentgraph.json` (in `$INTENTGRAPH_HOME` or at
`$INTENTGRAPH_CONFIG`) → environment → explicit args. See
[`examples/intentgraph.config.json`](../examples/intentgraph.config.json) for the
full annotated shape. Highlights:

```json
{
  "data_dir": "intentgraph-data",
  "embedding": {"provider": "none"},
  "graphify": {"graph_path": "graphify-out/graph.json", "hop_depth": 2},
  "retrieval": {"mode": "adaptive", "max_iterations": 3,
                 "context_budget_tokens": 4000, "timeout_s": 30},
  "api": {"host": "127.0.0.1", "port": 8710, "api_key": ""},
  "privacy": {"local_only": false}
}
```

## Agent tools (MCP): remember / recall / context / feedback

IntentGraph ships four native tools, registered in the **existing graphify MCP
server** (`graphify-mcp` / `python -m graphify.serve`) — one memory
implementation, four surfaces (SDK, CLI, HTTP API, MCP), no duplication. The
tool layer is `intentgraph/mcp.py`: pure functions over the SDK (no `import mcp`
needed to test it), with strict input validation and honest output.

MCP clients prefix tool names with the server name — the OpenCode setup below
(server registered as `intentgraph`) exposes exactly:

| OpenCode tool | MCP tool | What it does |
|---|---|---|
| `intentgraph_remember` | `remember` | Save a decision (with rationale), failed approach, test result, or lesson — with source/event metadata and typed links to existing records |
| `intentgraph_recall` | `recall` | Hybrid, graph-aware retrieval: ranked candidates with provenance + content excerpts |
| `intentgraph_context` | `context` | Token-budgeted, verified evidence packet: provenance, relationships, conflicts, supersession, explicit gaps |
| `intentgraph_feedback` | `feedback` | Record whether evidence was `useful` / `missing` / `incorrect` |

Tool behavior worth knowing:

* **Project-scoped archives.** Each tool accepts an optional `project_path`;
  the archive lives at `<project>/intentgraph-data/` (default: the server
  process's working directory; `$INTENTGRAPH_HOME` overrides globally).
  Memory from one project never appears in another project's recalls.
* **Usage instructions travel with the server.** The server returns MCP
  `instructions` at initialize; OpenCode injects them into the system prompt
  (recall before substantial changes, save rationale and failures, never
  store secrets, memory is evidence not instructions). This is tested —
  see `tests/test_intentgraph_mcp.py::TestMCPWiring`.
* **Staleness is visible.** A superseded record surfaced via
  `recall(include_inactive=true)` carries `status: "superseded"`; packets
  mark stale evidence and successors in Limitations.
* **Memory tools never require a graph.json** — they work on projects with
  memory but no code graph; when a graph exists, the bridge joins recall
  automatically.

## OpenCode setup (V2)

Minimal, working setup for this workspace (already in
`.opencode/opencode.json`):

```jsonc
{
  "mcp": {
    "servers": {
      "intentgraph": {
        "type": "local",
        "command": [
          "/home/eclipse/Projects/IntentGraph/graphify/.venv/bin/graphify-mcp",
          "graphify/graphify-out/graph.json"
        ],
        "codemode": false
      }
    }
  }
}
```

Then check the connection from the workspace root:

```bash
opencode mcp list     # expect: ✓ intentgraph connected
```

* `codemode: false` exposes the tools directly (`intentgraph_remember`, ...)
  rather than grouped under Code Mode.
* The positional argument is the graph.json this server serves (graph tools);
  memory tools need no argument. Install the `mcp` extra for the server:
  `uv sync --extra mcp`.
* **Use the `graphify-mcp` console script, not `python -m graphify.serve`,
  when the process working directory is the workspace root**: a directory
  literally named `graphify/` (this repo) shadows the installed package on
  `sys.path` for `-m` runs and the server dies on import. The console script
  does not prepend the cwd, so it is immune.

Example tool calls (what an agent actually sends):

```
intentgraph_remember  {"content": "ADR-14: amounts are integer minor units — the
                                    gateway API is integer-only.", "type": "decision",
                       "source_id": "session 2026-07-14", "event_time": "2026-07-14"}
intentgraph_remember  {"content": "Tried rounding to cents first; did not work (…)",
                       "type": "experience", "links": [{"id": "mem_<prior attempt>",
                       "relation": "failed_because_of"}]}
intentgraph_recall    {"query": "checkout INVALID_AMOUNT regression — what did we try?"}
intentgraph_context   {"task": "Fix the checkout regression; what failed before and
                              which decision followed?", "budget_tokens": 4000}
intentgraph_feedback  {"query": "checkout regression", "outcome": "useful",
                       "memory_ids": ["mem_…"]}
```

Reliability model — what is and is not automatic:

* **There are no automatic session hooks.** Saving and recalling happen when
  the agent calls the tools (encouraged by the server instructions and
  AGENTS.md, but not enforced). Explicit tool calls are the reliable path.
* The old graphify nudge plugin (`.opencode/plugins/graphify.js`) is unchanged
  and unrelated to memory; the MCP server instructions are the mechanism that
  carries the memory-usage rules.
* If tools do not appear: `opencode mcp list` (connection state), `/mcps` in
  the TUI, then `~/.local/share/opencode/log/opencode.log` filtered for
  `role=server`; a missing `mcp` extra shows as a server startup failure.
* `intentgraph benchmark --agent` reproduces the with/without-memory session
  measurement through the same tool functions the MCP layer calls.

## Graphify integration (the code-graph bridge)

Records can carry `graph_anchors`: node ids from a Graphify `graph.json`
(path-flattened — e.g. `graphify_llm` for `graphify/llm.py`, exactly the ids
`graphify explain` / the MCP tools use). In hybrid and adaptive modes
IntentGraph then also retrieves records whose anchors sit near the seeds'
anchors **in your code graph** — "the decision about the auth middleware"
surfaces "the test run for the auth middleware" with no memory edge between
them. Bridge candidates are labeled `graphify-bridge` in retrieval methods,
load errors degrade to "no code-adjacency" (never an exception), and every
hop is bounded. Integration is verified against this repository's own
15k-node graph in `tests/test_intentgraph_config.py`.

## External providers (Phase 3): Nebius Token Factory + Tavily

Both providers are **optional and off by default**; the default configuration
remains fully local and makes zero network requests. Everything below requires
explicit configuration, and `privacy.local_only: true` refuses every remote
provider class at configuration load — before any request could be made.

### What leaves the machine (read this once)

| Provider class | Sends | Never sends |
|---|---|---|
| Nebius embeddings | record **text** (`content` + tags + source_id, in batches) | record ids, provenance, metadata, the rest of the archive |
| Nebius LLM planning | the question, requirement shapes, coverage summary (requirement names + covered flags) | record content, ids, the archive |
| Nebius reranking | the query + excerpts (≤4000 chars) of the top-10 candidates | the archive, metadata |
| Tavily search | the search query only | **nothing else — never any memory content** |

### Nebius Token Factory (embeddings, planning, reranking)

Nebius Token Factory is an OpenAI-compatible inference API
(`https://api.tokenfactory.nebius.com/v1/` — verified against the published
API reference). Reusing the existing OpenAI-compatible client; no second HTTP
abstraction. Get a key at tokenfactory.nebius.com → API keys, then:

```bash
export NEBIUS_API_KEY=...          # or NEBIUS_TOKEN_FACTORY_KEY (lower precedence)
export INTENTGRAPH_EMBEDDING_PROVIDER=nebius
export INTENTGRAPH_EMBEDDING_MODEL=<model-id>   # REQUIRED — no default assumed
```

**Credential precedence** (each provider class): explicitly configured
`api_key` → `NEBIUS_API_KEY` → `NEBIUS_TOKEN_FACTORY_KEY` → the generic
`INTENTGRAPH_*_API_KEY`. Keys are never written to config files, logs, or
error messages.

**Verify model availability, then run the embedding smoke test:**

```bash
.venv/bin/python -m intentgraph embed --list-models   # what YOUR account serves
.venv/bin/python -m intentgraph embed                # (re)index the archive
.venv/bin/python -m intentgraph doctor               # provider status + stale-model check
```

**Dual providers — Nebius (default) and NVIDIA NIM side by side.** Both are
first-class aliases over the same OpenAI-compatible client, each with its own
credential, endpoint, and model; a template lives at `.env.example`. Switch
by editing exactly two lines:

```bash
# Nebius (the default — existing configurations keep working)
INTENTGRAPH_EMBEDDING_PROVIDER=nebius
INTENTGRAPH_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
# NVIDIA NIM (the NVIDIA-built embedding models Token Factory lacks)
INTENTGRAPH_EMBEDDING_PROVIDER=nvidia
INTENTGRAPH_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
```

Either provider can be tested independently — per-provider credential env
(`NEBIUS_API_KEY` / `NVIDIA_API_KEY`), per-provider live smoke tests, and
`list_models` against each endpoint — without removing the other. Switching
is safe by construction: vectors are keyed by provider+model+endpoint, the
index refuses to load another model's vectors (same provider, other model,
or other provider entirely — even at identical dimensions), and
`intentgraph embed` reindexes after the switch. A **selected-but-keyless
provider fails honestly before any request**: `is_configured()` is false,
`embed`/`--list-models` refuse with the exact env var to set, `doctor`
prints a CREDENTIALS MISSING line, and `reindex_semantic` returns a distinct
note (a different condition from "no provider configured").

**Catalog ≠ servability (live-verified on NIM):** `/models` may list models
that 404 on `/embeddings` — `nvidia/embed-qa-4` is listed but unserved on
this account, while `nvidia/nemotron-3-embed-1b` is verified working (2048
dims). Never assume a catalog entry serves; probe or smoke-test the exact
model before relying on it.

No model is hardcoded as universally available — accounts differ. The
configured model must be in `--list-models` output (the live smoke test
asserts exactly this). Behavior guarantees: batch responses are re-ordered by
index; vector dimensions are validated for completeness and consistency
(deterministic protocol failures are **not** retried, only rate limits and
5xx are, with bounded backoff); failures are honest — the system never
substitutes fake embeddings. Switching the embedding model is detected at
load time: the semantic index refuses stale vectors with a `stale_model`
note and `intentgraph embed` reindexes cleanly.

**LLM-assisted retrieval planning** (optional, deterministic fallback
always): `llm.provider: nebius` + `llm.model` adds an
Nebius-backed policy for query decomposition into evidence requirements and
retrieval-action selection (local follow-up vs external search vs stop).
**Tuning note (live-verified):** reasoning models (e.g. Nemotron) spend the
output budget on reasoning — the 1024-token default truncated the response
and correctly triggered the deterministic fallback; set
`llm.max_output_tokens: 4096` for valid plans.
Outputs are strict, schema-validated JSON; unknown fields are rejected;
there is **no schema field through which the model can claim evidence
exists** — evidence admission, gap assessment, and verdicts stay fully
deterministic in verify.py. Every failure mode (auth, rate limit, non-JSON,
off-schema) falls back to the deterministic policy with a recorded note in
the trace (`status.llm.fallbacks` counts them).

**Reranking** (optional): Token Factory's documented `POST /v1/rerank`
(vLLM-style `{model, query, documents}` → relevance scores). The endpoint is
documented, but whether a specific reranker model is served to YOUR account
is an account-level fact — set `reranker.model` explicitly (the API's own
example is `Qwen/Qwen3-Reranker-8B`) and verify with the live smoke test.
Reranking reorders the fused top slice only; it never adds or drops
evidence, and a provider failure keeps the fused order with an honest note.

### Tavily web search

```bash
export TAVILY_API_KEY=tvly-...
```

Config (all optional): `web_search.provider: tavily`,
`web_search.enabled` (gate the controller's automatic use — default false),
`max_results` (1–20, default 5), `search_depth`
(`basic|advanced|fast|ultra-fast`; advanced costs 2 API credits),
`topic` (`general|news|finance`), `time_range` (`day|week|month|year`),
`include_domains` / `exclude_domains`, `timeout_s` (default 15),
`retries` (default 2, on 429/5xx only, honoring Retry-After), and
`content_max_chars` (excerpt cap, default 2000).

Every result becomes an `ExternalEvidence` object — **never a user memory**:
canonical URL, title, provider excerpt, the provider's *estimated* publish
date (or an honest "no date returned" — never fabricated), retrieval
timestamp, the query that found it, provider name, relevance score, and
request/usage metadata (credits). Results without a citable URL are dropped
and counted. External findings are **not persisted by default**;
`web_search.store_results: true` is the explicit opt-in that ingests them as
ordinary records with full external provenance (`origin: external_search`,
the URL, the provider, the request id) so they can be deleted or superseded
like any memory.

### Selective external research in the adaptive controller

External search never runs on every query. The controller retrieves local
evidence first, then decides: a **freshness-demand heuristic** (documented
signals like "latest", "current documentation", "changelog", or a freshness
adjective + documentation-noun co-occurrence) triggers exactly one bounded
search call; local-history questions ("what did we decide last week") make
**zero** external requests — asserted by tests and measured by the benchmark.
An LLM policy may propose actions, but cannot propose external search unless
it is enabled and available. Outcomes are distinct and honest in the trace
and packet: `not_needed` | `ok` | `disabled` (with a "web search is disabled"
limitation) | `failed:<kind>` (auth/plan/rate_limit/network — explicitly
labeled "a provider failure, not an evidence verdict"). Local and external
evidence coexist in one packet with origins visible; when external dates are
newer than matching local event dates, a date-based comparison note says so —
disagreements are surfaced, never auto-resolved. Web content is untrusted
input: excerpts render inside the packet's untrusted-evidence fences with
their source URL.

### Credentials: the `.env` file

Keys never belong in tracked files. IntentGraph loads a `.env` file
automatically (a small stdlib parser — no new dependency):

* **Discovery**: `$INTENTGRAPH_ENV_FILE` (explicit path), then `./.env`, then
  each parent directory (≤6 levels) — so one `.env` at the workspace root
  serves the repo CLI, the SDK, and the OpenCode MCP server regardless of
  the process working directory.
* **Precedence**: variables already set in the real environment always win;
  `.env` only fills unset ones. Resolution order overall:
  real env → `.env` → config file → defaults.
* **Disable**: `INTENTGRAPH_DISABLE_DOTENV=1` (what the offline test suite
  does — see Hermeticity below).
* **Safety**: `.env` is in `.gitignore`; the loader never evaluates values,
  never echoes them, and error messages never contain key material.

```bash
# .env — NEVER COMMIT (gitignored)
NEBIUS_API_KEY=...                    # or NEBIUS_TOKEN_FACTORY_KEY
NVIDIA_API_KEY=nvapi-...             # NVIDIA NIM (nemotron-3-embed-1b, …)
TAVILY_API_KEY=tvly-...
INTENTGRAPH_EMBEDDING_PROVIDER=nvidia   # none|hash|nebius|nvidia|openai
INTENTGRAPH_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b  # from --list-models
INTENTGRAPH_WEB_SEARCH_PROVIDER=tavily
INTENTGRAPH_WEB_SEARCH_ENABLED=true
```

**Test-suite hermeticity**: the ordinary test suite must never require
credentials or make network calls — even on a machine with a populated
`.env`. `tests/conftest.py` therefore disables `.env` loading and strips all
provider credential/config variables for every test (live mode
`INTENTGRAPH_LIVE_TESTS=1` is the exempt, explicit opt-in). The live smoke
tests load `.env` themselves at import time, so credentials can live only
in the file.

### Live verification (this workspace, 2026-10-17)

Run against this account's real credentials, results you can reproduce:

* **Model catalog** (`intentgraph embed --list-models`): 28 models served.
  **No NVIDIA embedding model is available on this account** — the
  `nvidia/*` entries are Nemotron chat/vision models. The one embedding
  model served is `Qwen/Qwen3-Embedding-8B` (4096 dims), which is what
  `.env` configures. If your catalog differs, change
  `INTENTGRAPH_EMBEDDING_MODEL`.
* **Embeddings verified live on both providers.** Nebius
  (`Qwen/Qwen3-Embedding-8B`, 4096 dims) and NVIDIA NIM
  (`nvidia/nemotron-3-embed-1b`, 2048 dims — the currently configured
  provider): catalog discovery, reindex, usage metadata, and a paraphrase
  with zero lexical overlap ("why do small orders get rejected" → the
  integer-minor-units decision) retrieved by **semantic only** — real
  semantic retrieval, not the hash projection. Switching the embedding
  provider Qwen→Nemotron on a live archive exercised the **model-change
  detection** exactly as designed: the stored vectors were refused with a
  reindex instruction (`intentgraph doctor` reported STALE MODEL), and a
  clean reindex followed. Measured on this corpus: NVIDIA arm 336 ms mean
  remember() latency vs Nebius arm 930 ms (both 1.0 recall at the ceiling
  of the 11-record corpus — wiring/latency evidence, not quality claims).
* **NVIDIA NIM catalog** (`--list-models`): 81 models served, including
  seven embedding models (`nvidia/nemotron-3-embed-1b`,
  `nvidia/nv-embedqa-mistral-7b-v2`, `nvidia/embed-qa-4`, …). Credentials:
  `NVIDIA_API_KEY` (nvapi-* key) — the same OpenAI-compatible client, the
  documented `nvidia` provider alias, no second abstraction.
* **Reranker**: `/v1/rerank` returns **HTTP 404** on this account — no
  reranker model is served. The interface stays in place and reports the
  limitation; the live test skips with exactly this reason. This is the
  account-availability case the reranker docs anticipate.
* **LLM planning verified live** with `nvidia/Nemotron-3_5-Lightning`:
  with the default 1024 output cap the model spent it all on reasoning,
  the response truncated, validation rejected it, and the policy **fell
  back to deterministic rules** (the designed safety net, verified in
  production conditions). With `llm.max_output_tokens: 4096` the model
  returned a valid schema-conforming decomposition. **Tuning note: reasoning
  models need a larger output budget** — set `max_output_tokens` (and
  optionally `INTENTGRAPH_LLM_MODEL`) accordingly.
* **Tavily verified live**: search returned 5 results with canonical URLs,
  real titles, relevance scores, provider-estimated publish dates,
  retrieval timestamps, request id, and 1 credit usage.
* **Selectivity verified live**: a local-history question made **zero**
  external requests; a freshness question made exactly one; `--live` demo
  passes all eight steps against the real providers.


### Cost, latency, and troubleshooting

- Embeddings are re-computed only on explicit `intentgraph embed` or
  `reindex_semantic()` — never per query (one query-vector HTTP call per
  semantic retrieval round).
- Tavily basic/fast searches cost 1 API credit; advanced costs 2.
  `max_calls_per_query` (default 1) bounds per-query spend; usage metadata is
  captured per response and surfaced in traces.
- **401/403** → check the key variable name and value; the key never appears
  in error text. **429** → retried with backoff honoring `Retry-After`,
  then reported as `failed:rate_limit`. **432/433** (Tavily plan limits) →
  reported, never retried. **Embedding HTTP 404** → the model id is not
  served to this account — re-run `--list-models`. Endpoint unreachable →
  `failed:network`. `intentgraph doctor` reports every provider's status.
- Live end-to-end verification (opt-in, never in the default suite):
  `INTENTGRAPH_LIVE_TESTS=1 pytest tests/test_intentgraph_live_providers.py -v`.

### Provider benchmark

`intentgraph benchmark --providers` compares, on the same synthetic corpus
and ground truth as the pipeline benchmark: the deterministic **hash baseline**
(labeled as a lexical projection, NOT semantics), the **configured real
provider arm** (skipped with a stated reason when no provider is configured —
never faked), and **web-search selectivity** (a local-history question must
produce zero external calls; a freshness question's call count and status).
Results state their own limits: small synthetic corpus, this machine and
account, wiring and latency/tokens — not production-corpus quality claims.

## The HTTP API

`intentgraph api` serves a stdlib-only JSON API (no starlette/uvicorn needed):

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + archive stats |
| `POST /v1/memories` | ingest `{content, type?, tags?, source_id?, event_time?, metadata?, graph_anchors?}` |
| `GET /v1/memories` | list with pagination/filters |
| `GET /v1/memories/{id}` | the original record, full content |
| `GET /v1/memories/{id}/versions` | version history |
| `DELETE /v1/memories/{id}` | soft delete |
| `POST /v1/memories/{id}/restore` | undo soft delete |
| `POST /v1/memories/{id}/update` | append a new version |
| `POST /v1/links` | create a typed relationship |
| `POST /v1/retrieve` | one retrieval round (any mode, transparency included) |
| `POST /v1/context` | full adaptive retrieval → evidence packet |
| `POST /v1/feedback` | record `useful\|missing\|incorrect` retrieval feedback |
| `POST /v1/export` / `POST /v1/import` | bundle portability |

Security posture: 4 MiB body cap (enforced *before* buffering, with a bounded drain
so clients still receive their 413), strict record-id validation, optional bearer
auth (`--api-key` / `INTENTGRAPH_API_KEY`) — **required automatically** if you bind
a non-loopback host — JSON-only errors that never leak stack traces, and
`Cache-Control: no-store` on every response. Retrieved memory content is wrapped in
labeled untrusted-evidence fences in packets, with known chat-template injection
sentinels defanged (the same scheme Graphify's extraction path uses).

## Evidence verification & the context packet

Before anything enters a packet, each candidate is checked for provenance presence,
validity (deleted/superseded with successor pointer), temporal-edge consistency, and
whether claims are explicitly supported (linked artifacts) or merely asserted.
Declared contradictions are **surfaced with both sides and their provenance** — never
auto-adjudicated, and a dissenting source is never dropped just because it's
outnumbered. Near-duplicates are grouped but all kept. Requirement coverage is
topic-anchored: an off-topic record cannot "cover" a requirement, which is what makes
unanswerable questions abstain.

The packet reports the token method used (tiktoken when available, documented
chars/4 approximation otherwise), truncates bodies with explicit markers, lists
budget-excluded record ids for follow-up fetch, and includes every limitation.

## Benchmark

```bash
intentgraph benchmark              # deterministic, self-contained corpus
intentgraph benchmark --agent      # same task with vs without previous-session memory
intentgraph benchmark --providers  # hash vs real embeddings vs web-search selectivity
```

**Pipeline benchmark**: seven queries over an 11-record scenario (multi-hop
failure trail, paraphrase, decision rationale, contradiction, supersession,
exact artifact, unanswerable) with hand-verified ground truth, comparing
lexical vs hybrid vs adaptive: evidence recall/precision, abstention
correctness, conflict/supersession handling, packet token usage, latency.
The output states its own limits: the default hash provider measures the
*pipeline*, not neural semantic quality.

**Agent-session benchmark** (`--agent`): a session remembers a failed
approach, its test output, the decision that followed, a stale decision it
supersedes, and a contradictory claim — a *fresh* session then faces the
same task through the actual tool path. Measures required-evidence recovery
(1.0 with memory vs 0.0 without, by record id), source attribution,
failed-approach exposure, stale-status visibility, contradiction flagging,
latency, and packet tokens. Deterministic and model-free: it measures the
memory system's outputs, not LLM behavior (stated in the report itself).

## Limitations (honest)

- The requirement analyzer, relation weights, coverage heuristics, and the
  external-search trigger are **rule-based heuristics** — documented as such,
  not learned parameters.
- LLM-assisted planning (query decomposition, action selection) is optional
  and schema-validated with the deterministic policy as fallback on every
  failure; it assists planning only — evidence admission, gap assessment,
  and verdicts stay deterministic. It is the natural place to attach a
  learned activation policy later.
- Reranking is implemented against Token Factory's documented `/v1/rerank`,
  but **model availability is account-dependent** — no model is claimed
  universally available; verify with `--list-models` and the live smoke test.
  This build's tests verify reranking against mocked responses only (no
  credentials in the build environment).
- The hash embedding provider is a lexical-overlap projection, not semantics.
- Tavily's `published_date` is the provider's best estimate (or absent) —
  preserved exactly as returned, never fabricated, never treated as ground
  truth for conflict resolution.
- One SQLite writer; the local API serializes mutating endpoints under a lock.
  A hosted multi-writer deployment would swap the storage class, not the model.
- The HTTP server is stdlib `http.server` (local pattern); front the same core
  with any framework for a cloud deployment.
- **Memory tooling has no automatic session hooks**: saving/recalling happen
  when the agent calls the tools (server instructions encourage the right
  pattern; nothing enforces it). The agent-session benchmark measures the
  system's outputs, not whether a specific model acts on them.

## Relation to Graphify

One repo, one shared core, one-way dependency: `intentgraph` imports `graphify`
(security helpers, MinHash near-dup machinery, the tiktoken tokenizer helper,
and node-link graph-loading conventions) — `graphify` never imports `intentgraph`.
Graphify's code-graph analysis is unchanged; IntentGraph adds the memory substrate
and the retrieval intelligence around it.
