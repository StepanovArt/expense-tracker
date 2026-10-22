"""Exact (sha256) and near-duplicate (perceptual hash) detection.

Near-duplicates are grouped with union-find so that a whole group always lands in one split;
this is what makes the train/test leakage test meaningful for re-scans and re-crops.
"""

from __future__ import annotations

from collections.abc import Sequence

import imagehash
import numpy as np
from PIL import Image

PHASH_SIZE = 16  # 16x16 = 256-bit hash; 64-bit pHash is too coarse for white paper receipts


def phash_hex(img: Image.Image) -> str:
    return str(imagehash.phash(img.convert("L"), hash_size=PHASH_SIZE))


def _bits(hex_hashes: Sequence[str]) -> np.ndarray:
    """(n, n_bits) bool matrix."""
    raw = np.array([bytes.fromhex(h) for h in hex_hashes], dtype=object)
    arr = np.frombuffer(b"".join(raw), dtype=np.uint8).reshape(len(hex_hashes), -1)
    return np.unpackbits(arr, axis=1).astype(bool)


def hamming_matrix(hex_hashes: Sequence[str], block: int = 512) -> np.ndarray:
    """Pairwise Hamming distances (uint16), computed in blocks to bound memory."""
    if not hex_hashes:
        return np.zeros((0, 0), dtype=np.uint16)
    bits = _bits(hex_hashes).astype(np.float32)
    n = bits.shape[0]
    out = np.empty((n, n), dtype=np.uint16)
    ones = bits.sum(1)
    for i in range(0, n, block):
        b = bits[i : i + block]
        same_ones = b @ bits.T
        # popcount(a xor b) = |a| + |b| - 2|a and b|
        out[i : i + block] = (ones[i : i + block, None] + ones[None, :] - 2 * same_ones).astype(
            np.uint16
        )
    return out


def hamming(a: str, b: str) -> int:
    return int(hamming_matrix([a, b])[0, 1])


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[int]:
        return [self.find(i) for i in range(len(self.parent))]


def duplicate_groups(
    sha256s: Sequence[str], phashes: Sequence[str], max_distance: int
) -> list[int]:
    """Group id per item: equal sha256, or pHash distance <= max_distance (transitively)."""
    n = len(sha256s)
    uf = UnionFind(n)
    first: dict[str, int] = {}
    for i, h in enumerate(sha256s):
        if h in first:
            uf.union(first[h], i)
        else:
            first[h] = i
    if max_distance >= 0 and n > 1:
        d = hamming_matrix(phashes)
        ii, jj = np.nonzero(np.triu(d <= max_distance, k=1))
        for a, b in zip(ii.tolist(), jj.tolist(), strict=True):
            uf.union(a, b)
    return uf.groups()
