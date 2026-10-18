"""Token estimation and injection-sentinel neutralization — IntentGraph's own.

Two small, load-bearing utilities the context compiler uses:

* :func:`estimate_tokens` / :func:`token_method` — budget accounting for
  evidence packets. Uses tiktoken's ``cl100k_base`` when available for
  accuracy; falls back to the documented ~chars/4 heuristic otherwise, and
  the packet reports which method produced its numbers so a consumer never
  mistakes the approximation for a guarantee.
* :func:`neutralise_injection_sentinels` — defangs known chat-template and
  jailbreak control tokens inside remembered content before it enters an
  LLM-facing packet. A zero-width space after the first character keeps the
  text human-readable while breaking any template parser that matches the
  literal token.

Written fresh for IntentGraph (previously borrowed these helpers via a
guarded import); the behavior and the public strings are unchanged so
existing tests and packet fields stay identical.
"""
from __future__ import annotations

import re

__all__ = [
    "estimate_tokens",
    "get_tokenizer",
    "neutralise_injection_sentinels",
    "token_method",
]

_CHARS_PER_TOKEN = 4  # documented fallback heuristic for English/code prose

#: Known chat-template control tokens. The form itself is the hazard — no
#: legitimate memory content needs an intact one — so defanging (not
#: deleting) keeps the text readable while neutering the delimiter. Covers
#: the <|...|> family (ChatML, Llama-3 header ids, <|endofprompt|>, etc.),
#: Llama-2 system blocks, and the [/INST]-style boundaries.
_INJECTION_SENTINELS = re.compile(
    r"<\|[A-Za-z0-9_.\-]{1,64}\|>"
    r"|<<SYS>>|<</SYS>>"
    r"|\[/?(?:INST|SYSTEM)\]"
    r"|^\s*###?\s*(?:system|instruction)s?\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _get_tokenizer():
    """Return a tiktoken ``cl100k_base`` encoder, or None when unavailable.

    ``cl100k_base`` is a proxy tokenizer: IntentGraph does not assume the
    consuming model's exact tokenizer, but its token-to-char ratio tracks
    modern BPE tokenizers closely enough for budget decisions, which only
    need to be near, not exact.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # network failure on first-use download, etc.
        return None


# Cached at import time; None if tiktoken is unavailable. Consumers must
# handle the None branch (the approximation).
_TOKENIZER = _get_tokenizer()


def get_tokenizer():
    """Return the cached encoder (or None) — the module's cache, shared."""
    return _TOKENIZER


def token_method() -> str:
    """Which counting method produced the numbers: exact or approximate."""
    return ("tiktoken(cl100k_base)" if _TOKENIZER is not None
            else f"chars/{_CHARS_PER_TOKEN} approximation")


def estimate_tokens(text: str) -> int:
    """Token estimate: exact when tiktoken is available, else chars/4."""
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def neutralise_injection_sentinels(text: str) -> str:
    """Defang known chat-template / jailbreak control tokens in untrusted text.

    Inserts a zero-width space after the first character of each match so
    the literal token is no longer recognized by any model's template parser
    or by a naive delimiter scan, while keeping the text human-readable.
    """
    return _INJECTION_SENTINELS.sub(
        lambda m: m.group(0)[0] + "​" + m.group(0)[1:], text)
