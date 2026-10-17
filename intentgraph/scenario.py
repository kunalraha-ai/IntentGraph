"""The canonical IntentGraph scenario corpus (shared by tests + benchmark).

A deliberately memory-shaped story — a regression, its evidence, two failed
attempts, a superseded decision, the later decision, a stated fact, a
contradictory pair, and a near-duplicate — because retrieval behavior on
curated toy text proves nothing about behavior on real agent memory.

The benchmark's ground-truth "relevant record ids" (see eval.py) are the
names returned by :func:`seed`, so the corpus, its edges, and its
expectations can never drift apart.
"""

from __future__ import annotations

from intentgraph.sdk import IntentGraph

FAILURE = (
    "Regression report: on 2026-07-12 the checkout flow started failing for "
    "orders under $1 with error INVALID_AMOUNT from the payment gateway, right "
    "after the currency display refactor went out. About 3% of orders errored "
    "during the hour it was live."
)

TEST_OUTPUT = (
    "pytest output: test_checkout_under_one_dollar FAILED — "
    "PaymentGatewayError: INVALID_AMOUNT for amount=0.87 "
    "(expected integer minor units, got float) at gateway/client.py L88, "
    "exit code 1, 1 test failed, 412 passed."
)

ATTEMPT_1 = (
    "We tried rounding the amount to cents before validation. It did not work: "
    "the payment gateway still rejected the amounts because it expects integer "
    "minor units, and rounding never changed the type the client serialized."
)

ATTEMPT_2 = (
    "We tried serializing amounts as integer minor units everywhere in the "
    "client. It did not work either — the underlying regression was actually "
    "in the schema migration, which had silently dropped the minor_unit "
    "column, so half the rows had NULL minor units."
)

DECISION_OLD = (
    "ADR-11 decision: store order amounts as float dollars and round to cents "
    "on read. Chosen for display convenience and minimal code changes."
)

DECISION_NEW = (
    "ADR-14 decision: order amounts are stored as integer minor units, the "
    "migration re-adds the minor_unit column with a backfill, and the "
    "validation layer never converts through float. Rationale: the payment "
    "gateway API is integer-only, and every float conversion reintroduces the "
    "INVALID_AMOUNT failure mode we spent two days debugging."
)

GATEWAY_FACT = (
    "The payment gateway API only accepts integer minor units (e.g. 87 for "
    "$0.87); float amounts fail validation server-side."
)

TIMEOUT_A = "The payment gateway connection timeout is 5 seconds."
TIMEOUT_B = ("The payment gateway connection timeout is 30 seconds after the "
             "2026-06 reliability patch.")

FAILURE_NEAR_DUP = (
    "Checkout regression: orders below $1 began failing on 2026-07-12 with "
    "INVALID_AMOUNT from the payment gateway immediately after the currency "
    "display refactor shipped; roughly 3% of orders failed during its live hour."
)

PREFERENCE = (
    "Preference from the team channel: always use UTC timestamps in runbooks "
    "and postmortems so the on-call timeline lines up with the audit logs."
)


def seed(ig: IntentGraph, *, deterministic_ids: bool = False) -> dict[str, str]:
    """Ingest the scenario into an empty IntentGraph; return named ids.

    ``deterministic_ids=True`` derives record ids from the record's scenario
    name (sha256-truncated, still matching the ``mem_<16hex>`` shape) so the
    benchmark corpus is byte-identical across runs and machines — including
    id-tiebroken ranking order. Tests use random ids to stay isolated.
    """
    ids: dict[str, str] = {}

    def _did(name: str) -> str:
        if not deterministic_ids:
            return None  # type: ignore[return-value]
        import hashlib
        return f"mem_{hashlib.sha256(name.encode()).hexdigest()[:16]}"

    r = ig.ingest(
        FAILURE, id=_did("failure"), type="experience", source_type="tool", source_id="pagerduty-4471",
        event_time="2026-07-12T14:30:00+00:00", tags=["checkout", "regression"])
    ids["failure"] = r.record.id

    r = ig.ingest(
        TEST_OUTPUT, id=_did("test_output"), type="action", source_type="tool", source_id="ci/build-9931",
        event_time="2026-07-12T15:02:00+00:00", tags=["ci", "checkout"])
    ids["test_output"] = r.record.id

    r = ig.ingest(
        ATTEMPT_1, id=_did("attempt_1"), type="experience", source_type="conversation",
        source_id="war-room-0712", event_time="2026-07-12T16:40:00+00:00",
        tags=["attempt", "checkout"])
    ids["attempt_1"] = r.record.id

    r = ig.ingest(
        ATTEMPT_2, id=_did("attempt_2"), type="experience", source_type="conversation",
        source_id="war-room-0712", event_time="2026-07-12T18:05:00+00:00",
        tags=["attempt", "checkout", "root-cause"])
    ids["attempt_2"] = r.record.id

    r = ig.ingest(
        DECISION_OLD, id=_did("decision_old"), type="decision", source_type="conversation",
        source_id="adr-session-06", event_time="2026-06-28T10:00:00+00:00",
        tags=["adr", "amounts"])
    ids["decision_old"] = r.record.id

    r = ig.ingest(
        DECISION_NEW, id=_did("decision_new"), type="decision", source_type="conversation",
        source_id="adr-session-14", event_time="2026-07-14T09:30:00+00:00",
        tags=["adr", "amounts", "checkout"])
    ids["decision_new"] = r.record.id

    r = ig.ingest(
        GATEWAY_FACT, id=_did("gateway_fact"), type="fact", source_type="file",
        source_id="gateway-api-docs.md", tags=["gateway"])
    ids["gateway_fact"] = r.record.id

    r = ig.ingest(
        TIMEOUT_A, id=_did("timeout_a"), type="fact", source_type="file",
        source_id="gateway-api-docs.md", tags=["gateway", "timeout"])
    ids["timeout_a"] = r.record.id
    r = ig.ingest(
        TIMEOUT_B, id=_did("timeout_b"), type="fact", source_type="tool",
        source_id="reliability-patch-notes", event_time="2026-06-20T00:00:00+00:00",
        tags=["gateway", "timeout"])
    ids["timeout_b"] = r.record.id

    r = ig.ingest(
        FAILURE_NEAR_DUP, id=_did("failure_near_dup"), type="experience", source_type="conversation",
        source_id="standup-0713-notes", event_time="2026-07-13T09:00:00+00:00",
        tags=["checkout", "regression"])
    ids["failure_near_dup"] = r.record.id

    r = ig.ingest(
        PREFERENCE, id=_did("preference"), type="preference", source_type="conversation",
        source_id="team-channel", tags=["runbooks", "utc"])
    ids["preference"] = r.record.id

    # --- edges: the associative memory graph --------------------------------
    ig.link(ids["failure"], ids["test_output"], "validated_by",
            basis="the test output reproduces the reported failure")
    ig.link(ids["attempt_1"], ids["failure"], "related_to",
            basis="attempt made in response to the failure")
    ig.link(ids["attempt_2"], ids["attempt_1"], "related_to",
            basis="second attempt after the first failed")
    ig.link(ids["attempt_2"], ids["test_output"], "derived_from",
            basis="root cause came from reading the test output")
    ig.link(ids["attempt_1"], ids["gateway_fact"], "failed_because_of",
            basis="rounding did not fix the integer-only API contract")
    ig.link(ids["attempt_2"], ids["failure"], "related_to",
            basis="attempt made in response to the failure")
    ig.link(ids["decision_new"], ids["decision_old"], "supersedes",
            basis="ADR-14 replaces ADR-11 after the regression")
    ig.link(ids["decision_new"], ids["attempt_2"], "derived_from",
            basis="decision taken from the root cause the attempt surfaced")
    ig.link(ids["failure_near_dup"], ids["failure"], "duplicate_of",
            basis="standup retelling of the same incident")
    ig.link(ids["timeout_a"], ids["timeout_b"], "contradicts",
            basis="doc claims 5s, patch notes claim 30s")
    # explicit memory-graph supersedes must also flip validity state
    ig.supersede(ids["decision_old"], ids["decision_new"],
                 basis="ADR-14 replaces ADR-11 after the regression")
    return ids
