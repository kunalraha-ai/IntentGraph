"""Unit tests for intentgraph.lexindex (BM25 lexical retrieval)."""
from __future__ import annotations

import math

from intentgraph.lexindex import BM25Index, tokenize


def _mini_corpus() -> BM25Index:
    idx = BM25Index()
    idx.add("mem_a", "The auth middleware rejected expired tokens during login.")
    idx.add("mem_b", "We decided to adopt SQLite for the memory archive because it is local-first.")
    idx.add("mem_c", "checkout flow fails on safari when the payment token expires")
    idx.add("mem_d", "The auth middleware caches session state in redis.")
    return idx


class TestTokenize:
    def test_identifier_splitting(self):
        toks = tokenize("fixCacheLru eviction_policy")
        assert "fixcachelru" in toks
        # "cache" stems to "cach" (silent-e rule); the whole-part pair is
        # what makes both exact-name and natural-language queries match
        assert "fix" in toks and "cach" in toks and "lru" in toks
        assert "eviction_policy" in toks
        assert "eviction" in toks and "policy" in toks

    def test_diacritics_folded(self):
        # é folds to e, then the silent-e stem rule applies ("resum");
        # consistency with the query side is what matters, and "resum" is
        # the shared form of resume/résumé/résumés alike
        assert "resum" in tokenize("résumé of the failure")

    def test_empty(self):
        assert tokenize("") == []


class TestBM25:
    def test_ranks_relevant_doc_first(self):
        idx = _mini_corpus()
        hits = idx.search("auth middleware", top_k=4)
        assert hits[0]["id"] in ("mem_a", "mem_d")
        # matched_terms reports the actual posting terms (stemmed)
        assert set(hits[0]["matched_terms"]) == {"auth", "middlewar"}

    def test_deterministic_tiebreak(self):
        idx = BM25Index()
        for i in range(10):
            idx.add(f"mem_{i:016x}", "same words here")
        hits = idx.search("same words", top_k=10)
        assert [h["id"] for h in hits] == sorted(h["id"] for h in hits)
        assert len(hits) == 10

    def test_transparency_fields(self):
        idx = _mini_corpus()
        hit = idx.search("auth middleware")[0]
        assert hit["score"] > 0
        assert "score" in hit and "matched_terms" in hit and "term_scores" in hit
        assert all(isinstance(v, float) for v in hit["term_scores"].values())

    def test_stopwords_do_not_rank(self):
        idx = _mini_corpus()
        assert idx.search("the of and") == []

    def test_prefix_wildcard(self):
        idx = _mini_corpus()
        hits = idx.search("tok*")
        ids = [h["id"] for h in hits]
        assert "mem_a" in ids   # "tokens" via prefix
        assert "mem_c" in ids   # "token"

    def test_no_match_returns_empty(self):
        idx = _mini_corpus()
        assert idx.search("zzzzqqqq") == []

    def test_doc_filter_restricts(self):
        idx = _mini_corpus()
        hits = idx.search("auth middleware", doc_filter=lambda d: d == "mem_d")
        assert [h["id"] for h in hits] == ["mem_d"]

    def test_remove_then_readd(self):
        idx = BM25Index()
        idx.add("mem_a", "alpha beta")
        idx.add("mem_b", "alpha gamma")
        idx.remove("mem_a")
        assert len(idx) == 1
        hits = idx.search("beta")
        assert hits == []
        idx.add("mem_a", "beta delta")  # replace
        assert idx.search("beta")[0]["id"] == "mem_a"

    def test_build_replaces_everything(self):
        idx = BM25Index()
        idx.add("mem_old", "stale words")
        idx.build([("mem_new", "fresh words")])
        assert idx.search("stale") == []
        assert idx.search("fresh")[0]["id"] == "mem_new"

    def test_bm25_score_direction(self):
        # Rare terms must outrank common ones for the same doc: idf does work.
        idx = BM25Index()
        for i in range(5):
            idx.add(f"mem_{i:016x}", "common term everywhere")
        idx.add("mem_special", "common term everywhere plus rareword")
        hits = idx.search("common term everywhere rareword")
        assert hits[0]["id"] == "mem_special"

    def test_length_normalization_direction(self):
        # A short doc dense in the term ranks above a padded long doc.
        idx = BM25Index()
        idx.add("mem_short", "failure failure failure")
        idx.add("mem_long", "failure " + "filler padding noise words " * 50)
        hits = idx.search("failure")
        assert hits[0]["id"] == "mem_short"


class TestExactLookup:
    def test_exact_substring(self):
        idx = _mini_corpus()
        assert idx.exact_lookup("auth middleware") == ["mem_a", "mem_d"]
        assert idx.exact_lookup("checkout flow fails") == ["mem_c"]

    def test_exact_is_case_and_accent_insensitive(self):
        idx = BM25Index()
        idx.add("mem_x", "Résumé of the outage")
        assert idx.exact_lookup("RESUME") == ["mem_x"]

    def test_absent_needle(self):
        assert _mini_corpus().exact_lookup("postgresql driver") == []


class TestMath:
    def test_idf_formula_sane(self):
        # idf = ln(1 + (N - df + 0.5)/(df + 0.5)) — always positive, grows with rarity
        N, df_common, df_rare = 100, 90, 1
        idf_common = math.log(1 + (N - df_common + 0.5) / (df_common + 0.5))
        idf_rare = math.log(1 + (N - df_rare + 0.5) / (df_rare + 0.5))
        assert 0 < idf_common < idf_rare
