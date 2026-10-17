"""Pure-Python BM25 lexical index over memory records.

Why hand-rolled BM25 instead of an engine: sqlite FTS5 availability varies
across platform builds of CPython, and pulling a search dependency into a
local-first product for a few hundred thousand records is not justified. This
implementation is textbook Okapi BM25 (k1=1.5, b=0.75), fully deterministic
(same input → same ranking, ties broken by record id), and testable against
hand-computed expectations.

Properties:

* **In-memory, rebuilt on open.** The archive (SQLite) is the source of
  truth; this index is derived state, cheap to rebuild, and never persists —
  satisfying the "rebuildable indexes" portability requirement.
* **Identifier-aware tokenization.** ``snake_case`` and ``camelCase`` are
  split into parts AND kept whole, so ``failed_because_of`` matches both
  ``failed`` and the full phrase. Diacritics are folded (same treatment as
  graphify's query normalization).
* **A small stopword list**, not aggressive stemming — memory content is
  often short; stemming errors ("trying"→"try" is fine, "analysis"→"analysi"
  is not) hurt more than they help. Prefix matching (trailing wildcard)
  covers morphological drift without lying about it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Callable, Iterable

__all__ = ["tokenize", "BM25Index"]

_K1 = 1.5
_B = 0.75

# Deliberately small and content-preserving: negation words ("not", "no")
# are NOT here because "did not work" vs "worked" is exactly the distinction
# a memory archive must keep. Everything in this list is pure grammar.
_STOPWORDS = frozenset(
    "a an the of in on at to for with and or but is are was were be been "
    "this that these those it its as by from we you i they he she "
    "what which who when where why how do does did doing done have has had "
    "if then than so such about into over under again further".split()
)

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# \w is Unicode-aware by default in Python 3, so accented words stay whole
# chunks ("résumé") instead of being shredded at each é, then fold cleanly.
_WORD_RE = re.compile(r"\w+")


def _fold(text: str) -> str:
    """NFKC + casefold + strip diacritics (accent-insensitive matching)."""
    norm = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _parts_of(chunk: str) -> list[str]:
    """Sub-identifier parts of one raw (pre-fold) identifier chunk.

    Splits snake_case on ``_`` and camelCase on case boundaries BEFORE
    folding — folding first would erase the very boundaries we need.
    """
    parts: list[str] = []
    for piece in chunk.split("_"):
        if piece:
            parts.append(piece)
    if "_" not in chunk:
        for piece in _CAMEL_RE.split(chunk):
            if piece and piece not in parts:
                parts.append(piece)
    return parts


# --- light stemming --------------------------------------------------------------
#
# A deliberately small suffix-stripping stemmer (not Porter): memory prose is
# inflection-heavy ("fail" vs "failing" vs "failed" is THE difference between
# finding the root-cause record or not), while code identifiers mostly are not
# ("fixCacheLru" splits into short parts that never reach the length guard).
# It is applied IDENTICALLY to index text and query text, so a rule that
# mis-stems a word still matches the same mis-stemming on the other side —
# the failure mode of stemmers (inconsistency) cannot occur.
#
# Guards: a stem shorter than 3 chars reverts to the whole word, and only
# words ≥ 5 chars are candidates at all ("sing" must never become "s").

def _stem(word: str) -> str:
    if len(word) < 4:
        return word
    w = word
    # ies/ied → y ("tries"→"try", "tried"→"try", "applied"→"apply") —
    # applied first (and with a lower rem-guard: +y guarantees 3 chars) so
    # the ed/es rules cannot double-strip them
    for suffix, fix in (("ied", "y"), ("ies", "y")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 2:
            return w[: -len(suffix)] + fix
    for suffix in ("ing", "ed"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            # undouble a doubled final consonant ("running"→"run",
            # "stopped"→"stop") but keep ll/ss/zz ("spelling"→"spell",
            # "processing"→"process") — Porter's rule
            if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    for suffix in ("es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[: -len(suffix)]
            break
    # final silent-e ("decide"→"decid") so stem("decided")=="decid":
    # stripping the e unconditionally is what keeps inflections consistent
    if w.endswith("e") and len(w) - 1 >= 3:
        w = w[:-1]
    return w if len(w) >= 3 else word


def _words(text: str) -> list[str]:
    """Raw folded words: whole identifiers + sub-identifier parts, stopwords
    removed BEFORE stemming (else \"having\" would stem past the stopword
    list and get indexed as grammar noise)."""
    words: list[str] = []
    for chunk in _WORD_RE.findall(text):
        folded = _fold(chunk)
        for candidate in [folded, *[_fold(p) for p in _parts_of(chunk)]]:
            if candidate and candidate not in words:
                if candidate in _STOPWORDS:
                    continue
                words.append(candidate)
    return words


def tokenize(text: str) -> list[str]:
    """Tokenize for indexing: stopwords removed, then lightly stemmed.

    ``"fixCacheLru eviction_policy failing"`` →
    ``["fixcachelru", "fix", "cache", "lru", "eviction_policy", "eviction",
    "policy", "fail"]``
    — the whole identifier is kept so exact-name queries rank first, the
    parts so natural-language queries rank it at all, and the stem so
    \"failed\"/\"failing\"/\"fail\" are one term.
    """
    return [_stem(w) for w in _words(text)]


def _query_tokens(query: str) -> list[tuple[str, bool]]:
    """Query-side tokens: (token, is_prefix_wildcard), stopworded, deduped.

    Goes through the same stopword-filter + stem pipeline as the index, or
    the query would speak a different vocabulary than the postings.
    A trailing ``*`` marks a prefix search; the bare token without the ``*``
    is kept alongside so exact matches still rank.
    """
    out: list[tuple[str, bool]] = []
    for match in re.finditer(r"([A-Za-z0-9_]+)(\*?)", query):
        tok = _fold(match.group(1))
        if not tok or tok in _STOPWORDS:
            continue
        item = (_stem(tok), match.group(2) == "*")
        if item not in out:
            out.append(item)
    return out


class _Postings(Counter):
    """term frequency per document: {doc_id: tf}."""


class BM25Index:
    """Okapi BM25 over (record_id → text) pairs, incremental add/remove."""

    def __init__(self) -> None:
        self._tf: dict[str, _Postings] = {}   # term → {doc_id: tf}
        self._doc_len: dict[str, int] = {}
        self._total_len = 0
        self._N = 0
        # Raw (folded) text per doc, bounded by the archive size — needed by
        # exact_lookup for substring matching that term tokenization splits.
        self._raw: dict[str, str] = {}

    # -- maintenance ---------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        """Index one document (replace if the id is already indexed)."""
        if doc_id in self._doc_len:
            self.remove(doc_id)
        self._raw[doc_id] = _fold(text)
        toks = tokenize(text)  # stopword-filtered + stemmed internally
        tf = Counter(toks)
        self._doc_len[doc_id] = len(toks)
        self._total_len += len(toks)
        self._N += 1
        for term, freq in tf.items():
            self._tf.setdefault(term, _Postings())[doc_id] = freq

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_len:
            return
        n = self._doc_len.pop(doc_id)
        self._total_len -= n
        self._N -= 1
        self._raw.pop(doc_id, None)
        for term in [t for t, posts in self._tf.items() if doc_id in posts]:
            posts = self._tf[term]
            del posts[doc_id]
            if not posts:
                del self._tf[term]

    def build(self, items: Iterable[tuple[str, str]]) -> None:
        """Rebuild from scratch — the canonical refresh from the store."""
        self._tf.clear()
        self._doc_len.clear()
        self._total_len = 0
        self._N = 0
        self._raw.clear()
        for doc_id, text in items:
            self.add(doc_id, text)

    def __len__(self) -> int:
        return self._N

    @property
    def vocabulary_size(self) -> int:
        return len(self._tf)

    # -- search ----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        *,
        doc_filter: Callable[[str], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search → ranked candidates with full scoring transparency.

        Returns dicts: ``{"id", "score", "matched_terms", "term_scores"}``
        (deterministic order: score desc, then id asc). ``doc_filter`` lets
        callers restrict to active/current records without the index knowing
        about store semantics.
        """
        terms = _query_tokens(query)
        if not terms or self._N == 0:
            return []
        avgdl = (self._total_len / self._N) if self._N else 0.0

        scores: dict[str, float] = {}
        matched: dict[str, set[str]] = {}
        term_scores: dict[str, dict[str, float]] = {}

        for term, is_prefix in terms:
            # exact-term postings, plus prefix expansion for ``term*`` forms
            expanded: list[str] = [term]
            if is_prefix:
                expanded = [term] + [v for v in sorted(self._tf) if v.startswith(term)]
            for var in expanded:
                postings = self._tf.get(var)
                if not postings:
                    continue
                df = len(postings)
                if df == 0:
                    continue
                idf = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
                for doc_id, tf in postings.items():
                    if doc_filter is not None and not doc_filter(doc_id):
                        continue
                    if avgdl > 0:
                        norm = 1.0 - _B + _B * (self._doc_len.get(doc_id, 0) / avgdl)
                    else:
                        norm = 1.0
                    # Okapi: tf*(k1+1) / (tf + k1*norm) — norm carries the
                    # length normalization (1-b + b*dl/avgdl).
                    score = idf * (tf * (_K1 + 1.0)) / (tf + _K1 * norm)
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
                    matched.setdefault(doc_id, set()).add(term)
                    term_scores.setdefault(doc_id, {})[term] = round(score, 6)

        ranked: list[dict[str, Any]] = [
            {
                "id": doc_id,
                "score": round(score, 6),
                "matched_terms": sorted(matched.get(doc_id, set())),
                "term_scores": term_scores.get(doc_id, {}),
            }
            for doc_id, score in scores.items()
        ]
        ranked.sort(key=lambda c: (-c["score"], c["id"]))
        return ranked[: max(0, top_k)]

    def exact_lookup(self, needle: str) -> list[str]:
        """Doc ids whose indexed text contains the exact raw substring.

        BM25 ranks terms, but identifiers and filenames often need exact
        phrase behavior ("auth_middleware.py"). Used as the exact retriever's
        fallback when tokenization would split the needle apart.
        """
        if not needle:
            return []
        needle_folded = _fold(needle)
        # cheap pre-filter through the inverted index: only docs containing
        # the needle's first (stemmed) token can contain the needle — the
        # postings keys are stemmed, so the prefilter must speak stems too
        first = _WORD_RE.findall(needle_folded)
        candidates: set[str] = set()
        if not first:
            return []
        postings = self._tf.get(_stem(first[0]) if len(first[0]) >= 4 else first[0]) \
            or self._tf.get(first[0])
        if postings:
            candidates = set(postings)
        for tok in first[1:]:
            p = (self._tf.get(_stem(tok) if len(tok) >= 4 else tok)
                 or self._tf.get(tok))
            if not p:
                return []
            candidates &= set(p)
            if not candidates:
                return []
        out = []
        for doc_id in candidates:
            if needle_folded in self._raw.get(doc_id, ""):
                out.append(doc_id)
        return sorted(out)
