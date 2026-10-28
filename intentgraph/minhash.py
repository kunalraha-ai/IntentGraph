"""MinHash sketches for near-duplicate detection — IntentGraph's own module.

A clean, self-contained MinHash (no datasketch, no scipy): the sketch of a
document is the minimum of ``num_perm`` independent hash functions over its
shingles, and the Jaccard similarity of two documents is estimated by the
fraction of matching signature slots.

Each signature slot uses its own 64-bit constant (derived from sha-256 of the
slot index, so the hash family is identical in every process and machine — a
hard requirement here, because ingest writes the estimated similarity into
the archive as edge provenance) and mixes the shingle's 64-bit content hash
through the splitmix64 finalizer: three explicit mod-2^64 multiply/shift
steps whose wrapping is the mixer's definition, so every step is exact in
uint64 arithmetic with no hidden precision assumptions.

The public surface mirrors what :mod:`intentgraph.ingest` consumes:

    MinHash(num_perm=128)     .update(bytes)     .hashvalues (np.ndarray)
    .num_perm                .jaccard(other)     _jaccard(a, b)
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["MinHash", "_jaccard", "jaccard_estimate"]

# splitmix64 finalizer constants (David Stafford's mix13 variant).
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S1 = np.uint64(30)
_S2 = np.uint64(27)
_S3 = np.uint64(31)

# Per-num_perm slot constants, cached per process (deterministic, so sharing
# across instances is safe).
_SLOT_CONSTS: dict[int, np.ndarray] = {}


def _slot_constants(num_perm: int) -> np.ndarray:
    if num_perm not in _SLOT_CONSTS:
        consts = np.empty(num_perm, dtype=np.uint64)
        for i in range(num_perm):
            digest = hashlib.sha256(
                f"intentgraph/minhash/slot/{i}".encode()).digest()
            consts[i] = int.from_bytes(digest[:8], "little")
        _SLOT_CONSTS[num_perm] = consts
    return _SLOT_CONSTS[num_perm]


class MinHash:
    """One MinHash signature under construction.

    ``update(v)`` folds a shingle (any bytes) into the signature; slot i
    keeps the smallest mixed value it has seen. ``hashvalues`` is the
    signature — an ndarray so similarity estimation stays vectorized.
    """

    __slots__ = ("num_perm", "hashvalues", "_consts")

    def __init__(self, num_perm: int = 128) -> None:
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        self.num_perm = num_perm
        self.hashvalues = np.full(num_perm, np.uint64(0xFFFF_FFFF_FFFF_FFFF),
                                  dtype=np.uint64)
        self._consts = _slot_constants(num_perm)

    def _slot_values(self, h: np.uint64) -> np.ndarray:
        """Mix the 64-bit shingle hash through every slot's constant.

        Vectorized splitmix64 finalizer: ``z = h + const_i``, then three
        shift-multiply steps, all mod 2^64 (numpy's uint64 wrapping is the
        mixer's intended arithmetic — each step is a single multiply).
        """
        z = h + self._consts
        z = (z ^ (z >> _S1)) * _M1
        z = (z ^ (z >> _S2)) * _M2
        return z ^ (z >> _S3)

    def update(self, v: bytes) -> None:
        """Fold one shingle into the signature."""
        h = np.uint64(int.from_bytes(hashlib.sha256(v).digest()[:8], "little"))
        np.minimum(self.hashvalues, self._slot_values(h), out=self.hashvalues)

    def jaccard(self, other: "MinHash") -> float:
        """Estimate Jaccard similarity against another sketch."""
        if self.num_perm != other.num_perm:
            raise ValueError("sketches must share num_perm")
        agree = int(np.count_nonzero(self.hashvalues == other.hashvalues))
        return agree / float(self.num_perm)


def _jaccard(a: MinHash | None, b: MinHash | None) -> float:
    """Signature-agreement estimate; 0.0 for missing sketches.

    Compatibility shim matching the call shape intentgraph.ingest used with
    the previous sketch module, so the ingest path reads the same.
    """
    if a is None or b is None:
        return 0.0
    return a.jaccard(b)


def jaccard_estimate(a: MinHash | None, b: MinHash | None) -> float:
    """Public alias of :func:`_jaccard`."""
    return _jaccard(a, b)
