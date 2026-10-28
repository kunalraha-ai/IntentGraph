#!/usr/bin/env bash
# IntentGraph end-to-end demo — deterministic, offline, no API keys.
#
# Seeds the canonical scenario (a checkout regression: failure evidence, two
# failed attempts, a superseded decision, the later decision, a contradictory
# pair, a near-duplicate), then runs the flagship adaptive query and the
# unanswerable question so you can see, in one output:
#
#   1. multi-hop evidence recovery (failure → attempts → decision)
#   2. gap-targeted follow-up rounds
#   3. declared contradictions surfaced with both sides
#   4. supersession handled with the successor
#   5. honest abstention on a question the archive cannot answer
#
# Run:  bash examples/intentgraph-demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

HOME_DIR="$(mktemp -d /tmp/opencode/intentgraph-demo-XXXXXX)"
trap 'rm -rf "$HOME_DIR"' EXIT
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "== IntentGraph demo (offline, deterministic) =================="
echo "Archive: $HOME_DIR"
echo

# --- 1. seed the scenario through the CLI (ingest + typed links) --------------
run() { "$PY" -m intentgraph --home "$HOME_DIR" "$@"; }

"$PY" - <<EOF
from intentgraph.sdk import IntentGraph
from intentgraph.scenario import seed
ig = IntentGraph(data_dir="$HOME_DIR")
seed(ig)
print("Seeded 11 memories + 10 typed relationships (failure trail,")
print("superseded decision, contradiction pair, near-duplicate).")
print("Stats:", ig.status()["records"], "records,", ig.status()["edges"], "edges")
ig.close()
EOF

echo
echo "== Flagship query: fix the regression; what we tried, why it failed, the decision =="
"$PY" - <<EOF
import json
from intentgraph.sdk import IntentGraph
ig = IntentGraph(data_dir="$HOME_DIR")
result = ig.remember(
    "Fix the checkout regression. Remember what we tried before, "
    "why it failed, and which architecture decision we made afterward.",
    budget_tokens=2000)
print(result["text"])
ig.close()
EOF

echo
echo "== Unanswerable question: must abstain, not guess ========================="
"$PY" - <<EOF
from intentgraph.sdk import IntentGraph
ig = IntentGraph(data_dir="$HOME_DIR")
result = ig.remember(
    "What did we decide about the postgres replication topology?",
    budget_tokens=1200)
packet = result["packet"]
print("verdict:", packet["verdict"])
print("iterations:", packet["iterations"], "| stop:", packet["stop_reason"])
for q in packet["unresolved_questions"]:
    print("unresolved:", q[:110])
ig.close()
EOF

echo
echo "== Benchmark: lexical vs hybrid vs adaptive ================================"
run benchmark | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
print('corpus:', d['corpus']['records'], 'records,', d['corpus']['query_count'], 'queries')
for mode, s in d['summary'].items():
    print(f\"  {mode:9s} mean_recall={s['mean_recall']} precision={s['mean_precision']}\")
print('  abstention_correct:', d['summary']['adaptive'].get('abstention_correct'))
"

echo
echo "== Export / import (portability) =========================================="
run export "$HOME_DIR-bundle.jsonl" > /dev/null
run --home "$HOME_DIR-restore" import "$HOME_DIR-bundle.jsonl" \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); print('restored:', d['records'], 'records,', d['edges'], 'edges,', 'skipped(dedup):', d['skipped'])"

echo
echo "Demo complete. Try it yourself:"
echo "  $PY -m intentgraph --home $HOME_DIR remember 'what is the gateway timeout?'"
