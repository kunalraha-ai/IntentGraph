#!/usr/bin/env python
"""IntentGraph Phase-3 end-to-end demonstration: Nebius + Tavily integration.

Reproducible scenario, offline by default (a mock Tavily provider stands in
for the network; every assertion is on IntentGraph's own behavior). Run with
``--live`` to exercise REAL providers (requires TAVILY_API_KEY and, for the
embedding arm, NEBIUS_API_KEY + INTENTGRAPH_EMBEDDING_MODEL — see
docs/intentgraph.md "Live verification").

The eight verified steps:
  1. store a historical architecture decision + a failed experiment locally
  2. ask a question answerable from local memory
  3. verify: local evidence used, ZERO external requests
  4. ask a question requiring current external documentation
  5. verify: Tavily called only when enabled; disabled → honest gap note
  6. combine historical and external evidence in ONE context packet
  7. confirm provenance, timestamps, and origin separation preserved
  8. disable external providers → local functionality intact
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intentgraph.sdk import IntentGraph  # noqa: E402
from intentgraph.websearch import (  # noqa: E402
    ExternalEvidence,
    TavilySearchProvider,
    WebSearchProvider,
)


class MockTavily(WebSearchProvider):
    """Offline stand-in with the same contract as the real provider.

    Returns deterministic, provenance-complete evidence for the freshness
    question and nothing for anything else — so the demo asserts IntentGraph
    behavior, not network luck."""

    name = "tavily"

    def __init__(self):
        self.calls: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def describe(self):
        return {"provider": self.name, "available": True}

    def search(self, query: str):
        self.calls.append(query)
        self.calls_made = getattr(self, "calls_made", 0) + 1
        if "gateway" not in query.lower():
            return []
        return [ExternalEvidence(
            url="https://docs.example.com/gateway-2026",
            title="Payment Gateway SDK — current documentation",
            excerpt="The 2026 gateway SDK accepts amounts as integer minor "
                    "units only. Float amounts are rejected server-side "
                    "(INVALID_AMOUNT) since SDK 4.2.",
            query=query, provider="tavily", score=0.93,
            published_date="Wed, 01 Oct 2026 00:00:00 GMT",
            retrieved_at="2026-10-16T00:00:00+00:00")]


FAILURES: list[str] = []


def count_calls(provider) -> int:
    """Uniform external-call counter for mock (calls list) and real
    (calls_made) providers — the zero-requests invariant must be assertable
    in live mode too, not only against the mock."""
    if hasattr(provider, "calls_made"):
        return provider.calls_made
    return len(getattr(provider, "calls", []))


def check(step: int, name: str, condition: bool, detail: str = "") -> None:
    status = "VERIFIED" if condition else "FAILED"
    print(f"  [{status}] step {step}: {name}"
          + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"step {step}: {name}")


def main() -> int:
    live = "--live" in sys.argv
    tmp = tempfile.mkdtemp(prefix="intentgraph-phase3-demo-")
    print(f"== IntentGraph Phase-3 demo ({'LIVE providers' if live else 'offline, mocked Tavily'}) ==")
    print(f"Archive: {tmp}\n")

    ig = IntentGraph(data_dir=tmp, overrides={"web_search.provider": "tavily"})
    provider = (TavilySearchProvider(max_results=3) if live
                else MockTavily())

    # --- 1. store local history ---------------------------------------------
    print("1) Storing a historical decision and a failed experiment locally…")
    failed_experiment = ig.ingest(
        "Failed experiment: we tried storing order amounts as float dollars "
        "and rounding to cents on read. It did not work: the payment "
        "gateway rejects float amounts with INVALID_AMOUNT.",
        type="experience", source_id="war room 2026-07-12",
        event_time="2026-07-12T18:05:00+00:00")
    decision = ig.ingest(
        "Architecture decision (ADR-11-era): amounts are stored as float "
        "dollars. Rationale: minimal migration from the legacy schema.",
        type="decision", source_id="ADR session 2026-07",
        event_time="2026-07-14T09:30:00+00:00")
    check(1, "two memories stored",
          failed_experiment.record.id and decision.record.id)

    # wire the external provider into the controller (enabled)
    ig.controller.web_search = provider
    ig.controller.external_max_calls = 1

    # --- 2/3. local-history question, zero external calls --------------------
    print("2) Asking a LOCAL-history question…")
    local_result = ig.remember(
        "what did we try for amounts before, and why did it fail?",
        budget_tokens=1500)
    print("3) Verifying it used local evidence and made no external calls…")
    check(2, "local packet has evidence",
          bool(local_result["packet"]["evidence"]))
    check(3, "zero external requests for a history question",
          count_calls(provider) == 0
          and local_result["trace"]["external_status"] == "not_needed",
          f"calls={count_calls(provider)}, status={local_result['trace']['external_status']}")

    # --- 4/5. freshness question: external only when enabled -----------------
    print("4) Asking a question needing CURRENT external documentation…")
    calls_before_fresh = count_calls(provider)
    ig.controller.web_search = None  # first: disabled
    disabled_result = ig.remember(
        "what does the latest version of the gateway SDK require for "
        "amounts?", budget_tokens=1500)
    check(5, "disabled → honest gap, zero calls",
          count_calls(provider) == calls_before_fresh
          and disabled_result["trace"]["external_status"] == "disabled"
          and any("web search is disabled" in n
                  for n in disabled_result["trace"]["notes"]))
    ig.controller.web_search = provider  # now: enabled
    external_result = ig.remember(
        "what does the latest version of the gateway SDK require for "
        "amounts?", budget_tokens=1500)
    check(5, "enabled → exactly one external call",
          count_calls(provider) == calls_before_fresh + 1
          and external_result["trace"]["external_status"] == "ok")

    # --- 6/7. combined packet: local + external, provenance preserved ---------
    print("6) Combining historical and external evidence in one packet…")
    combined = ig.remember(
        "compare our float-dollars decision with the current gateway "
        "documentation requirements", budget_tokens=2500)
    text = combined["text"]
    packet = combined["packet"]
    check(6, "packet carries BOTH local and external evidence",
          any("float dollars" in e["content"]
              for e in packet["evidence"])
          and bool(packet.get("external_evidence")))
    print("7) Verifying provenance, timestamps, origin separation…")
    check(7, "external section explicitly labeled non-memory",
          "## External evidence (web search — NOT user memories)" in text)
    # provenance asserted from the ACTUAL packet content (mock or live):
    # whatever came back must be fully cited in the rendering
    ext = (packet.get("external_evidence") or [{}])[0]
    check(7, "external entry carries URL + retrieval timestamp + provider",
          bool(ext.get("url")) and str(ext.get("url", "")) in text
          and "retrieved:" in text and "provider: tavily" in text)
    check(7, "local evidence keeps its record ids and event times",
          f"<untrusted_evidence id=\"{decision.record.id}\"" in text)
    check(7, "publish date is the provider's estimate (or honestly none)",
          "provider's estimate" in text or "no date returned" in text)
    # freshness comparison: the NOTE must fire exactly when the dates say
    # so (date-based, never fabricated) — in live mode the provider decides
    # what dates exist, so the invariant is the conditional, not the outcome
    from intentgraph.models import parse_timestamp
    from intentgraph.websearch import parse_provider_date
    local_when = parse_timestamp(decision.record.event_time)
    newer_ext = any(
        (parse_provider_date(e.get("published_date")) or local_when)
        > local_when
        for e in packet.get("external_evidence") or [])
    note_fired = any("more recent than the matching local records" in n
                     for n in combined["trace"]["notes"])
    check(7, "freshness comparison surfaced iff external dates are newer",
          note_fired == newer_ext,
          f"note={note_fired} newer_external_dates={newer_ext}")

    # --- 8. disable everything: local still works ------------------------------
    print("8) Disabling external providers; verifying local still works…")
    ig.controller.web_search = None
    calls_before = count_calls(provider)
    final_result = ig.remember(
        "why did the float-dollar experiment fail?", budget_tokens=1200)
    check(8, "local retrieval fully functional, still zero new calls",
          bool(final_result["packet"]["evidence"])
          and count_calls(provider) == calls_before
          and final_result["trace"]["external_status"] == "not_needed")

    ig.close()
    print()
    if FAILURES:
        print(f"DEMO FAILED ({len(FAILURES)} step(s)):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("DEMO COMPLETE: all 8 steps verified.")
    print("\nLive mode: INTENTGRAPH_LIVE_TESTS docs are in "
          "docs/intentgraph.md; run with --live and real credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
