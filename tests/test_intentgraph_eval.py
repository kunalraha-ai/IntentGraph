"""End-to-end test for the reproducible benchmark (intentgraph.eval)."""
from __future__ import annotations

import pytest

from intentgraph.eval import run_benchmark


@pytest.fixture(scope="module")
def results():
    return run_benchmark()


def test_benchmark_runs_and_reports_schema(results):
    assert results["benchmark"].startswith("intentgraph/")
    assert results["corpus"]["records"] == 11
    assert len(results["rows"]) == 7
    for mode in ("lexical", "hybrid", "adaptive"):
        assert mode in results["summary"]


def test_benchmark_measures_recoverable_evidence(results):
    """The pipeline must actually recover the known supporting records.

    Floors are set below the measured values (see BENCHMARKS note in docs) so
    the test asserts *capability*, not exact leaderboard equality.
    """
    by_name = {r["name"]: r for r in results["rows"]}
    # the multi-hop failure/attempt/decision trail: adaptive recall floor
    multi = by_name["multi_hop_failure_trail"]["adaptive"]
    assert multi["recall"] >= 0.75, multi
    # exact artifact id lookup: lexical alone must nail it (it is the one
    # place where exactness is trivially achievable)
    exact = by_name["exact_test_output"]["lexical"]
    assert exact["recall"] == 1.0, exact
    # the paraphrase query is where hybrid must beat plain lexical
    para_hyb = by_name["paraphrase_cheap_orders"]["hybrid"]["recall"]
    para_lex = by_name["paraphrase_cheap_orders"]["lexical"]["recall"]
    assert para_hyb >= para_lex


def test_benchmark_abstains_on_unanswerable(results):
    by_name = {r["name"]: r for r in results["rows"]}
    unanswerable = by_name["unanswerable_postgres"]["adaptive"]
    assert unanswerable["abstention_correct"] is True
    assert unanswerable["verdict"] == "insufficient"


def test_benchmark_packets_respect_budget(results):
    for row in results["rows"]:
        adaptive = row["adaptive"]
        assert adaptive["packet_tokens"] <= adaptive["budget_tokens"] * 1.1


def test_benchmark_states_honest_notes(results):
    notes = " ".join(results["notes"])
    assert "NOT a neural embedding" in notes or "not measure semantic" in notes.lower()
    assert results["embedding_provider"]  # provider is always reported


def test_benchmark_is_deterministic(results):
    second = run_benchmark()
    # latency is wall-clock and excluded: determinism is about RANKING,
    # recall, verdicts, and packet sizes — not machine timing noise.

    def _stable_modes(res):
        return [{k: v for k, v in mode.items() if k != "mean_latency_ms"}
                for mode in res["summary"].values()]

    def _stable_rows(res):
        return [(r["name"],
                 {k: v for k, v in r["lexical"].items() if k != "latency_ms"},
                 {k: v for k, v in r["hybrid"].items() if k != "latency_ms"},
                 {k: v for k, v in r["adaptive"].items()
                  if k not in ("latency_ms",)})
                for r in res["rows"]]

    assert _stable_modes(second) == _stable_modes(results)
    assert _stable_rows(second) == _stable_rows(results)


# --- the provider comparison benchmark (Phase 3) ---------------------------------------


@pytest.fixture(scope="module")
def provider_results():
    from intentgraph.eval import run_provider_benchmark
    return run_provider_benchmark()


def test_provider_benchmark_structure(provider_results):
    arms = {a["arm"]: a for a in provider_results["arms"]}
    assert "hash_baseline" in arms
    assert "configured_real" in arms
    # the hash arm is always measurable and honestly labeled
    assert arms["hash_baseline"]["mean_evidence_recall"] is not None
    assert "NOT semantic embeddings" in arms["hash_baseline"]["note"]


def test_unconfigured_real_arm_skipped_with_reason(provider_results):
    arm = next(a for a in provider_results["arms"]
               if a["arm"] == "configured_real")
    assert arm.get("status") in (None, "skipped")
    if arm.get("status") == "skipped":
        assert "reason" in arm and arm["reason"]


def test_web_selectivity_measured_by_call_counts(provider_results):
    sel = provider_results["web_search_selectivity"]
    # the invariant that must hold in every configuration: a local-history
    # question performs ZERO external requests
    assert sel["local_only_query"]["external_calls"] == 0
    assert sel["local_only_query"]["status"] == "not_needed"


def test_provider_benchmark_states_its_limits(provider_results):
    notes = " ".join(provider_results["notes"])
    assert "small" in notes and "synthetic" in notes
    assert "skipped" in notes or "never" in notes


# --- the agent-session benchmark (Phase 2) -------------------------------------------


@pytest.fixture(scope="module")
def session_results():
    from intentgraph.eval import run_session_benchmark
    return run_session_benchmark()


def test_session_benchmark_measures_both_arms(session_results):
    assert "with_memory" in session_results
    assert "without_memory" in session_results
    assert session_results["required_evidence"] == [
        "attempt", "test_output", "decision"]


def test_with_memory_recovers_required_evidence_by_id(session_results):
    arm = session_results["with_memory"]
    # all three required records, by exact id, in one recall call
    assert arm["recall_of_required"] == {
        "attempt": True, "test_output": True, "decision": True}
    assert arm["required_evidence_recall"] == 1.0
    # and in the context packet
    assert arm["packet_required_evidence_hits"] == 3


def test_with_memory_carries_source_attribution(session_results):
    assert session_results["with_memory"]["source_attribution_correct"] is True


def test_with_memory_exposes_the_failed_approach(session_results):
    # the objective precondition for an agent not repeating it
    assert session_results["with_memory"]["failed_approach_exposed"] is True


def test_with_memory_marks_stale_and_conflicting_records(session_results):
    arm = session_results["with_memory"]
    assert arm["stale_decision_status_visible"] is True
    assert arm["contradiction_pair_retrieved"] is True
    assert arm["packet_flags_contradiction"] is True


def test_without_memory_has_zero_recoverable_history(session_results):
    arm = session_results["without_memory"]
    assert arm["required_evidence_recall"] == 0.0
    assert arm["failed_approach_exposed"] is False
    assert arm["abstained"] is True  # honest insufficient-evidence state


def test_session_benchmark_reports_latencies_and_tokens(session_results):
    for arm_name in ("with_memory", "without_memory"):
        arm = session_results[arm_name]
        assert arm["recall_latency_ms"] >= 0
        assert arm["packet_tokens"] > 0
    # latency is wall-clock and exact packet token counts carry record-id
    # entropy (session ids are uuid-derived; citations include them), so
    # tokens are asserted as a band, not an exact match — behavioral
    # determinism (recall, attribution, flags, verdicts) has strict tests
    from intentgraph.eval import run_session_benchmark as _run
    second = _run()
    for arm_name in ("with_memory", "without_memory"):
        a = session_results[arm_name]["packet_tokens"]
        b = second[arm_name]["packet_tokens"]
        assert abs(a - b) <= max(1, int(0.15 * a)), (arm_name, a, b)


def test_session_benchmark_states_its_limits(session_results):
    notes = " ".join(session_results["notes"])
    assert "not LLM behavior" in notes
    assert "subjective judgment" in notes
