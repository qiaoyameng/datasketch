from __future__ import annotations

import hashlib
import heapq
import struct
from functools import lru_cache
from typing import Callable, Iterable, Optional

import numpy as np


@lru_cache(maxsize=None)
def _seeded_sha1_hash(width: int, seed: int) -> Callable[[bytes], int]:
    """Build a deterministic width-fixed SHA-1 hash with a seed prefix."""
    prefix = struct.pack("<Q", seed)
    unpack_format = "<I" if width == 32 else "<Q"

    def hashfunc(data: bytes) -> int:
        digest = hashlib.sha1()
        digest.update(prefix)
        digest.update(data)
        return struct.unpack(unpack_format, digest.digest()[: width // 8])[0]

    hashfunc.__name__ = "seeded_sha1_hash%d" % width
    hashfunc.hash_scheme = "sha1"
    hashfunc.hash_width = width
    return hashfunc


class KMV:
    r"""A K minimum values sketch for cardinality and Jaccard estimation.

    The sketch keeps the ``k`` smallest distinct fixed-width hash values seen.
    Until it has seen fewer than ``k`` distinct hashes, :meth:`count` returns
    the exact number of distinct retained hashes. Once full, cardinality is
    estimated with the KMV estimator

    .. math::

        \\widehat{n} = \\frac{(k - 1)(2^w)}{X_{(k)} + 1},

    where :math:`X_{(k)}` is the largest retained hash and :math:`w` is the
    hash width. Adding one in the denominator is the discrete equivalent of
    using the upper boundary of the largest hash bucket; it also keeps the
    estimator defined when that hash is zero.

    For two compatible sketches, the combined sketch is first reduced to their
    common capacity. If full, let :math:`T = (X_{(k)} + 1) / 2^w` and let
    :math:`K_x` be the number of shared hashes among those :math:`k` union
    hashes. The estimates are :math:`\widehat{|A \\cup B|} = (k-1)/T`,
    :math:`\widehat{|A \\cap B|} = K_x/T`, and
    :math:`\widehat{J(A,B)} = K_x/(k-1)`. Estimates are clipped to their valid
    ranges to absorb finite-sample overshoot; clipping does not remove sampling
    error.

    The estimator assumes hash values behave like independent uniform samples.
    It is therefore approximate even when no two input elements collide: the
    retained hashes are a random sample. Increasing ``k`` reduces sampling
    error (its relative scale is approximately :math:`1/\\sqrt{k-1}`), but no
    finite sketch guarantees that every estimate is exact.

    A finite-width hash can also assign the same hash to two distinct inputs.
    That collision makes the sketch count distinct hashes rather than distinct
    inputs, can make values from different sets appear shared, and biases
    very-large-set cardinality downward while distorting similarity. The
    64-bit default makes this negligible for ordinary cardinalities; use
    32-bit hashes only when the smaller state is worth the additional collision
    risk.

    Args:
        k: Maximum number of distinct hashes to retain. Must be at least two.
        seed: Seed for the default seeded SHA-1 hash functions. Sketches with
            different seeds cannot be merged or compared.
        width: Width of fixed-width hashes: 32 or 64 bits.
        hashfunc: Hash function accepting an input value and returning an
            unsigned integer in the configured width. When omitted, a seeded
            SHA-1 hash matching ``width`` is used.
        hash_scheme: Name of a custom hash scheme. It must match for sketches
            to be merged or compared.
        hashvalues: Optional iterable of existing hash values used to
            initialize the sketch.
    """

    __slots__ = ("_heap", "_values", "hash_scheme", "hashfunc", "k", "seed", "width")

    def __init__(
        self,
        k: int = 128,
        seed: int = 0,
        width: int = 64,
        hashfunc: Optional[Callable] = None,
        hash_scheme: Optional[str] = None,
        hashvalues: Optional[Iterable[int]] = None,
    ) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 2:
            raise ValueError("k must be an integer of at least 2")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if seed >= 1 << 64:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        if width not in (32, 64):
            raise ValueError("width must be 32 or 64")
        if hashfunc is None:
            hashfunc = _seeded_sha1_hash(width, seed)
            hash_scheme = "sha1"
        elif not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        elif hash_scheme is None:
            hash_scheme = getattr(hashfunc, "hash_scheme", "custom")
        if not isinstance(hash_scheme, str) or not hash_scheme:
            raise ValueError("hash_scheme must be a non-empty string")

        self.k = k
        self.seed = seed
        self.width = width
        self.hashfunc = hashfunc
        self.hash_scheme = hash_scheme
        self._values: set[int] = set()
        self._heap: list[int] = []

        if hashvalues is not None:
            values = list(hashvalues)
            if len(values) > k:
                raise ValueError("hashvalues contains more than k values")
            for value in values:
                self._values.add(self._validate_hash(value))
            self._heap = [-value for value in self._values]
            heapq.heapify(self._heap)

    @property
    def max_hash_value(self) -> int:
        """Largest value representable by the configured fixed-width hash."""
        return (1 << self.width) - 1

    @property
    def threshold(self) -> int:
        """Current maximum retained hash."""
        if not self._values:
            raise ValueError("An empty KMV sketch has no threshold")
        return -self._heap[0]

    def update(self, b) -> None:
        """Update the sketch with one value."""
        value = self._validate_hash(self.hashfunc(b))
        self._insert(value)

    def update_batch(self, values: Iterable) -> None:
        """Update the sketch with an iterable of values."""
        for value in values:
            self.update(value)

    def count(self) -> float:
        """Return exact distinct-hash cardinality before capacity, or an estimate.

        Returns:
            The exact number of retained distinct hashes when the sketch is not
            full; otherwise the KMV cardinality estimate.
        """
        if not self.is_full():
            return float(len(self._values))
        return (self.k - 1) / self._normalized_threshold()

    def merge(self, other: KMV) -> None:
        """Merge another sketch into this one.

        Different capacities are supported by reducing both sketches to their
        common capacity. Different hash schemes, hash functions, widths, or
        seeds are rejected.
        """
        self._check_compatible(other)
        common_k = min(self.k, other.k)
        merged = sorted(self._values | other._values)[:common_k]
        self._replace_values(common_k, merged)

    @classmethod
    def union(cls, *sketches: KMV) -> KMV:
        """Create a new sketch containing the union of two or more sketches."""
        if len(sketches) < 2:
            raise ValueError("Cannot union fewer than 2 KMV sketches")
        first = sketches[0]
        for sketch in sketches[1:]:
            first._check_compatible(sketch)
        common_k = min(sketch.k for sketch in sketches)
        values: set[int] = set()
        for sketch in sketches:
            values.update(sketch._values)
        merged = sorted(values)[:common_k]
        return cls(
            k=common_k,
            seed=first.seed,
            width=first.width,
            hashfunc=first.hashfunc,
            hash_scheme=first.hash_scheme,
            hashvalues=merged,
        )

    def union_count(self, other: KMV) -> float:
        """Estimate the cardinality of the union represented by two sketches."""
        return self._set_estimates(other)[0]

    def intersection_count(self, other: KMV) -> float:
        """Estimate non-negative intersection cardinality.

        When the combined sketch is full, let ``k_shared`` be the number of
        hashes shared by both inputs among the smallest ``k`` union hashes and
        let ``T`` be the normalized upper boundary of that combined threshold.
        The estimate is ``k_shared / T``. Because ``k_shared`` is a count, the
        result cannot be negative. A zero estimate means no shared hash was
        observed in the sample; it is not a proof of disjointness.
        """
        return self._set_estimates(other)[1]

    def jaccard(self, other: KMV) -> float:
        """Estimate Jaccard similarity between the two represented sets."""
        return self._set_estimates(other)[2]

    def digest(self) -> np.ndarray:
        """Return a sorted copy of the retained hash values."""
        dtype = np.uint32 if self.width == 32 else np.uint64
        return np.asarray(sorted(self._values), dtype=dtype)

    def copy(self) -> KMV:
        """Create an independent copy of this sketch."""
        return self.__class__(
            k=self.k,
            seed=self.seed,
            width=self.width,
            hashfunc=self.hashfunc,
            hash_scheme=self.hash_scheme,
            hashvalues=self.digest(),
        )

    def clear(self) -> None:
        """Remove all retained hashes while preserving configuration."""
        self._values.clear()
        self._heap.clear()

    def is_empty(self) -> bool:
        """Return whether no hash values are retained."""
        return not self._values

    def is_full(self) -> bool:
        """Return whether the sketch has reached its retention capacity."""
        return len(self._values) == self.k

    def __len__(self) -> int:
        """Return the number of currently retained distinct hashes."""
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KMV):
            return False
        return (
            type(self) is type(other)
            and self.k == other.k
            and self.seed == other.seed
            and self.width == other.width
            and self.hash_scheme == other.hash_scheme
            and self._values == other._values
        )

    def _insert(self, value: int) -> None:
        if value in self._values:
            return
        if len(self._values) < self.k:
            self._values.add(value)
            heapq.heappush(self._heap, -value)
        elif value < self.threshold:
            old_value = -heapq.heapreplace(self._heap, -value)
            self._values.remove(old_value)
            self._values.add(value)

    def _replace_values(self, k: int, values: Iterable[int]) -> None:
        self.k = k
        self._values = set()
        self._heap = []
        for value in values:
            self._values.add(value)
            heapq.heappush(self._heap, -value)

    def _normalized_threshold(self) -> float:
        return float(self.threshold + 1) / float(self.max_hash_value + 1)

    def _set_estimates(self, other: KMV) -> tuple[float, float, float]:
        self._check_compatible(other)
        common_k = min(self.k, other.k)
        union_values = sorted(self._values | other._values)[:common_k]
        if not union_values:
            return 0.0, 0.0, 0.0
        if len(union_values) < common_k:
            union_estimate = float(len(union_values))
        else:
            union_threshold = (union_values[-1] + 1) / float(self.max_hash_value + 1)
            union_estimate = (common_k - 1) / union_threshold

        if len(union_values) < common_k:
            intersection = float(sum(1 for value in union_values if value in self._values and value in other._values))
            similarity = intersection / union_estimate if union_estimate else 0.0
            return union_estimate, intersection, similarity

        shared_count = sum(1 for value in union_values if value in self._values and value in other._values)
        intersection = shared_count / union_threshold
        raw_similarity = shared_count / float(common_k - 1)
        similarity = min(1.0, max(0.0, raw_similarity))
        intersection = min(intersection, union_estimate, self.count(), other.count())
        return union_estimate, intersection, similarity

    def _check_compatible(self, other: KMV) -> None:
        if type(self) is not type(other):
            raise ValueError("Cannot merge or compare KMV sketches of different types")
        if self.seed != other.seed:
            raise ValueError("Cannot merge or compare KMV sketches with different seeds")
        if self.width != other.width:
            raise ValueError("Cannot merge or compare KMV sketches with different hash widths")
        if self.hash_scheme != other.hash_scheme:
            raise ValueError("Cannot merge or compare KMV sketches with different hash schemes")
        if self.hashfunc is not other.hashfunc:
            raise ValueError("Cannot merge or compare KMV sketches with different hash functions")

    def _validate_hash(self, value) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            if not isinstance(value, np.integer):
                raise ValueError("hashfunc must return an integer")
        integer_value = int(value)
        if integer_value < 0 or integer_value > self.max_hash_value:
            raise ValueError("hash value %d is outside the %d-bit range" % (integer_value, self.width))
        return integer_value


KMinimumValues = KMV
