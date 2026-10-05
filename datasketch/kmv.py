from __future__ import annotations

import hashlib
import heapq
import struct
from collections.abc import Iterable
from functools import partial
from typing import Callable, Optional

import numpy as np


def _seeded_sha1_hash(data, seed: int = 1, width: int = 32) -> int:
    """Return a deterministic seed-salted, fixed-width SHA-1 hash."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("The default KMV hash function expects bytes-like input.")
    digest = hashlib.sha1(struct.pack("<Q", seed) + bytes(data)).digest()
    if width == 32:
        return struct.unpack("<I", digest[:4])[0]
    return struct.unpack("<Q", digest[:8])[0]


class KMV:
    """A K minimum values (KMV) sketch.

    The sketch keeps the ``k`` smallest distinct fixed-width hash values.
    Fewer than ``k`` distinct hashes means the cardinality is known exactly;
    once saturated, the k-th smallest hash ``X_k`` is treated as the k-th
    order statistic of uniform values in ``[0, 1)`` and cardinality is
    estimated as ``(k - 1) * 2**width / X_k``.

    Similarity is estimated on the smallest common capacity. Let ``U_k`` be
    that bottom-k union and ``I_k`` the hashes appearing in both retained
    sets; then Jaccard similarity is ``|I_k| / |U_k|``. When both sketches
    are unsaturated this ratio is exact.

    The default SHA-1 based hash is salted with ``seed``. Sketches may only
    be merged or compared when they have the same hash scheme, width, seed,
    and custom hash callable when applicable.

    Finite-width hashes can collide: distinct inputs with the same hash are
    deliberately stored once. Estimates therefore describe distinct hashes,
    and very large input sets should use a 64-bit hash. Estimation error is
    random, not an exact correction; for ``k > 2`` the approximate relative
    standard error of a saturated cardinality estimate is
    ``1 / sqrt(k - 2)``.

    Args:
        k: The maximum number of distinct hashes to retain. Must be at least 2
            because the saturated cardinality estimator uses ``k - 1``.
        seed: Seed used by the built-in seed-salted SHA-1 hash.
        hashfunc: A custom one-argument callable returning a nonnegative
            fixed-width integer. When supplied, ``hash_scheme`` is ``custom``.
        width: Hash width in bits; used with custom hash functions.
        hash_scheme: ``sha1-32`` (default) or ``sha1-64`` for the built-in
            hash schemes. Cannot be combined with a custom ``hashfunc``.
        hashvalues: Optional iterable of existing distinct hash values.
    """

    _BUILT_IN_SCHEMES = {
        "sha1-32": 32,
        "sha1-64": 64,
    }

    def __init__(
        self,
        k: int = 128,
        seed: int = 1,
        hashfunc: Optional[Callable] = None,
        width: Optional[int] = None,
        hash_scheme: Optional[str] = None,
        hashvalues: Optional[Iterable[int]] = None,
    ) -> None:
        if hashfunc is not None:
            if hash_scheme is not None and hash_scheme != "custom":
                raise ValueError("hash_scheme must be 'custom' when hashfunc is supplied")
            hash_scheme = "custom"
            width = 32 if width is None else width
        else:
            hash_scheme = "sha1-32" if hash_scheme is None else hash_scheme
            if hash_scheme not in self._BUILT_IN_SCHEMES:
                raise ValueError("hash_scheme must be one of: sha1-32, sha1-64, or custom")
            scheme_width = self._BUILT_IN_SCHEMES[hash_scheme]
            if width is not None and width != scheme_width:
                raise ValueError("width does not match hash_scheme")
            width = scheme_width
            hashfunc = partial(_seeded_sha1_hash, seed=seed, width=width)

        if not isinstance(k, int) or isinstance(k, bool) or k < 2:
            raise ValueError("k must be an integer no smaller than 2")
        if width not in (32, 64):
            raise ValueError("width must be 32 or 64")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if not callable(hashfunc):
            raise ValueError("hashfunc must be callable")

        self.k = k
        self.seed = seed
        self.width = width
        self.hash_scheme = hash_scheme
        self.hashfunc = hashfunc
        self.hash_range = 1 << width
        self._max_hash = self.hash_range - 1
        self._hashes: set[int] = set()
        self._heap: list[int] = []

        if hashvalues is not None:
            values = list(hashvalues)
            if len(set(values)) != len(values):
                raise ValueError("hashvalues must contain distinct values")
            for hv in values:
                self._add_validated_hash(self._validate_hash(hv))
            if len(self._hashes) > k:
                raise ValueError("hashvalues contains more than k values")

    def update(self, b) -> None:
        """Add one value after applying the configured hash function."""
        self._add_validated_hash(self._validate_hash(self.hashfunc(b)))

    def update_batch(self, values: Iterable) -> None:
        """Add many values."""
        for value in values:
            self.update(value)

    def count(self) -> float:
        """Return the exact distinct hash count when unsaturated, else an estimate."""
        if not self.is_saturated():
            return float(len(self._hashes))
        return self._estimate(self.k, self.threshold())

    def intersection_count(self, other: KMV) -> float:
        """Estimate the non-negative number of hashes in both sets."""
        union_count, _, similarity = self._similarity_estimates(other)
        return union_count * similarity

    def jaccard(self, other: KMV) -> float:
        """Estimate Jaccard similarity from shared values in the common KMV set."""
        union_count, _, similarity = self._similarity_estimates(other)
        if union_count == 0.0:
            return 1.0
        return similarity

    def _similarity_estimates(self, other: KMV):
        self._check_compatible(other)
        common_k = min(self.k, other.k)
        if not self.is_saturated() and not other.is_saturated():
            union_values = self._hashes | other._hashes
            denominator = len(union_values)
            union_count = float(denominator)
        else:
            union_values = self._bottom_k(self._hashes | other._hashes, common_k)
            denominator = common_k
            union_count = self._count_state(union_values, denominator)
        shared_count = len(union_values & self._hashes & other._hashes)
        similarity = 1.0 if denominator == 0 else float(shared_count) / float(denominator)
        return union_count, shared_count, similarity

    def merge(self, other: KMV) -> None:
        """Replace this sketch with its union with ``other``.

            Different capacities are reduced to their common capacity by
            retaining only that many smallest union hashes.
        """
        self._check_compatible(other)
        common_k = min(self.k, other.k)
        union_values = self._bottom_k(self._hashes | other._hashes, common_k)
        self._replace(common_k, union_values)

    @classmethod
    def union(cls, *sketches: KMV) -> KMV:
        """Return a new KMV sketch containing the union of all sketches."""
        if len(sketches) < 2:
            raise ValueError("Cannot union fewer than 2 KMV sketches")
        first = sketches[0]
        for sketch in sketches[1:]:
            first._check_compatible(sketch)
        common_k = min(sketch.k for sketch in sketches)
        union_values: set[int] = set()
        for sketch in sketches:
            union_values.update(sketch._hashes)
        union_values = first._bottom_k(union_values, common_k)
        return first._new_with_config(common_k, union_values)

    def digest(self) -> np.ndarray:
        """Return a sorted copy of the retained distinct hash values."""
        dtype = np.uint32 if self.width == 32 else np.uint64
        return np.array(sorted(self._hashes), dtype=dtype)

    def threshold(self) -> Optional[int]:
        """Return the largest retained hash, or ``None`` when empty."""
        if not self._heap:
            return None
        return -self._heap[0]

    def is_saturated(self) -> bool:
        """Return whether capacity has been reached."""
        return len(self._hashes) == self.k

    def is_empty(self) -> bool:
        """Return whether no hash has been retained."""
        return not self._hashes

    def clear(self) -> None:
        """Remove all retained hashes."""
        self._hashes.clear()
        self._heap.clear()

    def copy(self) -> KMV:
        """Return an independent copy of this sketch."""
        return self._new_with_config(self.k, set(self._hashes))

    def __len__(self) -> int:
        return self.k

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and self.k == other.k
            and self.seed == other.seed
            and self.width == other.width
            and self.hash_scheme == other.hash_scheme
            and self._configurations_equal(other)
            and self._hashes == other._hashes
        )

    def _validate_hash(self, hv) -> int:
        if isinstance(hv, bool) or not isinstance(hv, (int, np.integer)):
            raise ValueError("hashfunc must return an integer")
        hv = int(hv)
        if not 0 <= hv <= self._max_hash:
            raise ValueError("hash value is outside the configured %d-bit range" % self.width)
        return hv

    def _add_validated_hash(self, hv: int) -> None:
        if hv in self._hashes:
            return
        if len(self._hashes) < self.k:
            self._hashes.add(hv)
            heapq.heappush(self._heap, -hv)
        elif hv < self.threshold():
            old = -heapq.heappop(self._heap)
            self._hashes.remove(old)
            self._hashes.add(hv)
            heapq.heappush(self._heap, -hv)

    def _replace(self, k: int, values: set[int]) -> None:
        self.k = k
        self.clear()
        for hv in values:
            self._add_validated_hash(self._validate_hash(hv))

    def _new_with_config(self, k: int, values: set[int]) -> KMV:
        if self.hash_scheme == "custom":
            return self.__class__(
                k=k,
                seed=self.seed,
                hashfunc=self.hashfunc,
                width=self.width,
                hashvalues=values,
            )
        return self.__class__(
            k=k,
            seed=self.seed,
            hash_scheme=self.hash_scheme,
            hashvalues=values,
        )

    def _check_compatible(self, other: KMV) -> None:
        if type(self) is not type(other):
            raise ValueError("Cannot merge KMV with a different sketch type")
        if self.width != other.width or self.hash_scheme != other.hash_scheme:
            raise ValueError("Cannot merge KMV sketches with different hash schemes")
        if self.seed != other.seed:
            raise ValueError("Cannot merge KMV sketches with different seeds")
        if not self._configurations_equal(other):
            raise ValueError("Cannot merge KMV sketches with different hash functions")

    def _configurations_equal(self, other: KMV) -> bool:
        if self.hash_scheme != "custom":
            return True
        return self.hashfunc == other.hashfunc

    def _count_state(self, values: set[int], capacity: int) -> float:
        if len(values) < capacity:
            return float(len(values))
        threshold = max(values)
        return self._estimate(capacity, threshold)

    def _estimate(self, capacity: int, threshold: Optional[int]) -> float:
        if capacity < 2:
            raise ValueError("KMV cardinality estimation requires capacity k >= 2")
        if threshold is None or threshold <= 0:
            raise ValueError("KMV cardinality estimate has an invalid zero threshold")
        return float(capacity - 1) * float(self.hash_range) / float(threshold)

    @staticmethod
    def _bottom_k(values: set[int], k: int) -> set[int]:
        return set(heapq.nsmallest(k, values))
