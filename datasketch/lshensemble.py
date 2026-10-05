import logging
import struct
from collections import Counter
from collections.abc import Generator, Hashable, Iterable
from numbers import Integral
from typing import Optional

import numpy as np

from datasketch.lsh import MinHashLSH, integrate
from datasketch.lshensemble_partition import optimal_partitions
from datasketch.minhash import MinHash
from datasketch.storage import _random_name

logger = logging.getLogger(__name__)

MinHashDigest = tuple[str, Optional[str], int]


def _false_positive_probability(threshold, b, r, xq):
    """Compute the false positive probability given the containment threshold.
    xq is the ratio of x/q.
    """
    _probability = lambda t: 1 - (1 - (t / (1 + xq - t)) ** float(r)) ** float(b)
    if xq >= threshold:
        a, _err = integrate(_probability, 0.0, threshold)
        return a
    a, _err = integrate(_probability, 0.0, xq)
    return a


def _false_negative_probability(threshold, b, r, xq):
    """Compute the false negative probability given the containment threshold."""
    _probability = lambda t: 1 - (1 - (1 - (t / (1 + xq - t)) ** float(r)) ** float(b))
    if xq >= 1.0:
        a, _err = integrate(_probability, threshold, 1.0)
        return a
    if xq >= threshold:
        a, _err = integrate(_probability, threshold, xq)
        return a
    return 0.0


def _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight):
    """Compute the optimal parameters that minimizes the weighted sum
    of probabilities of false positive and false negative.
    xq is the ratio of x/q.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        for r in range(1, max_r + 1):
            if b * r > num_perm:
                continue
            fp = _false_positive_probability(threshold, b, r, xq)
            fn = _false_negative_probability(threshold, b, r, xq)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def _validate_size(size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, Integral):
        raise TypeError("Set size must be an integer")
    size = int(size)
    if size <= 0:
        raise ValueError("Set size must be positive")
    return size


def _minhash_digest(minhash: MinHash, num_perm: int) -> MinHashDigest:
    try:
        length = len(minhash)
        seed = getattr(minhash, "seed", None)
    except (TypeError, AttributeError) as error:
        raise TypeError("minhash must be a MinHash or WeightedMinHash object") from error
    if length != num_perm:
        raise ValueError("Expecting minhash with length %d, got %d" % (num_perm, length))
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("MinHash seed must be an integer")
    scheme = getattr(minhash, "scheme", None)
    kind = "minhash" if scheme is not None else "weighted"
    return kind, scheme, int(seed)


def _check_minhash_digest(
    minhash: MinHash,
    num_perm: int,
    known_digest: Optional[MinHashDigest] = None,
) -> MinHashDigest:
    digest = _minhash_digest(minhash, num_perm)
    if known_digest is not None and digest != known_digest:
        raise ValueError(
            "MinHash digest parameters %r do not match parameters %r of the indexed MinHashes"
            % (digest, known_digest)
        )
    return digest


class MinHashLSHEnsemble:
    """The :ref:`minhash_lsh_ensemble` index. It supports
    :ref:`containment` queries.
    The implementation is based on
    `E. Zhu et al. <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.

    Args:
        threshold (float): The Containment threshold between 0.0 and
            1.0. The initialized LSH Ensemble will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        num_part (int): The number of partitions in LSH Ensemble.
        m (int): The memory usage factor: an LSH Ensemble uses approximately
            `m` times more memory space than a MinHash LSH with the same number of
            sets indexed. The higher the `m` the better the accuracy.
        weights (Tuple[float, float]): Used to adjust the relative importance of
            minizing false positive and false negative when optimizing
            for the Containment threshold. Similar to the `weights` parameter
            in :class:`datasketch.MinHashLSH`.
        storage_config (Optional[Dict]): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (Optional[bool]): If True, all keys are pickled to bytes before
            insertion. If None, a default value is chosen based on the
            `storage_config`.

    Note:
        Using more partitions (`num_part`) leads to better accuracy, at the
        expense of slower query performance.
        This is different from `the paper`_ and the `Go implementation`_, in which
        more partitions leads to better accuracy AND faster query performance,
        due to parallelism.

    Note:
        More information about the parameter `m` can be found in the
        `Go implementation`_
        of LSH Ensemble, in which `m` is named `MaxK`.

    .. _`Go implementation`: https://github.com/ekzhu/lshensemble
    .. _`the paper`: http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        num_part: int = 16,
        m: int = 8,
        weights: tuple[float, float] = (0.5, 0.5),
        storage_config: Optional[dict] = None,
        prepickle: Optional[bool] = None,
    ) -> None:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 2 or m > num_perm:
            raise ValueError("m must be in the range of [2, num_perm]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        self.num_part = num_part
        rs = self._init_optimal_params(weights)
        self._rs = rs
        # Initialize multiple LSH indexes for each partition
        storage_config = storage_config if storage_config else {"type": "dict"}
        self._storage_config = dict(storage_config)
        self._prepickle = prepickle
        self._basename = storage_config.get("basename", _random_name(11))
        self.indexes = self._create_indexes(self._basename)
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]
        self._entries: dict[Hashable, tuple[MinHash, int]] = {}
        self._minhash_digest: Optional[MinHashDigest] = None
        self._index_initialized = False

    def _init_optimal_params(self, weights):
        false_positive_weight, false_negative_weight = weights
        self.xqs = np.exp(np.linspace(-5, 5, 10))
        self.params = np.array(
            [
                _optimal_param(
                    self.threshold,
                    self.h,
                    self.m,
                    xq,
                    false_positive_weight,
                    false_negative_weight,
                )
                for xq in self.xqs
            ],
            dtype=int,
        )
        # Find all unique r
        rs = set()
        for _, r in self.params:
            rs.add(r)
        return rs

    def _get_optimal_param(self, x, q):
        i = np.searchsorted(self.xqs, float(x) / float(q), side="left")
        if i == len(self.params):
            i = i - 1
        return self.params[i]

    def _get_storage_config(self, basename, base_config, partition, r):
        config = dict(base_config)
        config["basename"] = b"-".join([basename, struct.pack(">H", partition), struct.pack(">H", r)])
        return config

    def _create_indexes(self, basename: Optional[bytes] = None):
        if basename is None:
            basename = _random_name(11)
        return [
            dict(
                (
                    r,
                    MinHashLSH(
                        num_perm=self.h,
                        params=(int(self.h / r), r),
                        storage_config=self._get_storage_config(
                            basename, self._storage_config, partition, r
                        ),
                        prepickle=self._prepickle,
                    ),
                )
                for r in self._rs
            )
            for partition in range(self.num_part)
        ]

    def _build_indexes(
        self, entries: list[tuple[Hashable, MinHash, int]]
    ) -> tuple[list, list[Optional[int]], list[Optional[int]], Optional[MinHashDigest]]:
        indexes = self._create_indexes()
        lowers: list[Optional[int]] = [None for _ in indexes]
        uppers: list[Optional[int]] = [None for _ in indexes]
        if not entries:
            return indexes, lowers, uppers, None

        digest = None
        for _key, minhash, _size in entries:
            digest = _check_minhash_digest(minhash, self.h, digest)

        sizes, counts = np.array(sorted(Counter(size for _key, _minhash, size in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(indexes))
        for i, (lower, upper) in enumerate(partitions):
            lowers[i], uppers[i] = int(lower), int(upper)

        ordered_entries = sorted(entries, key=lambda entry: entry[2])
        curr_part = 0
        for key, minhash, size in ordered_entries:
            upper = uppers[curr_part]
            if upper is not None and size > upper:
                curr_part += 1
            for r in indexes[curr_part]:
                indexes[curr_part][r].insert(key, minhash)
        return indexes, lowers, uppers, digest

    def _materialize_entries(
        self, entries: Iterable[tuple[Hashable, MinHash, int]]
    ) -> list[tuple[Hashable, MinHash, int]]:
        materialized = []
        seen = set()
        for key, minhash, size in entries:
            if key in seen:
                raise ValueError("Duplicate key %r in entries" % (key,))
            seen.add(key)
            size = _validate_size(size)
            _check_minhash_digest(minhash, self.h)
            materialized.append((key, minhash.copy(), size))
        return materialized

    def update_session(self) -> "MinHashLSHEnsembleUpdateSession":
        """Create a transactional session for adding, removing, and replacing entries."""
        if self._storage_config.get("type") != "dict":
            raise ValueError("update_session is only supported for in-memory dict storage")
        return MinHashLSHEnsembleUpdateSession(self)

    def index(self, entries: Iterable[tuple[Hashable, MinHash, int]]) -> None:
        """Index all sets given their keys, MinHashes, and sizes.
        It can be called only once after the index is created.

        Args:
            entries (Iterable[tuple[Hashable, MinHash, int]]): An iterable of
                tuples, each must be in the form of ``(key, minhash, size)``,
                where ``key`` is the unique
                identifier of a set, ``minhash`` is the MinHash of the set,
                and ``size`` is the size or number of unique items in the set.

        Raises:
            ValueError: If the index has already been initialized or ``entries`` is empty.

        """
        if self._index_initialized:
            raise ValueError("Cannot call index again after the index has been initialized")
        entries = self._materialize_entries(entries)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        indexes, lowers, uppers, digest = self._build_indexes(entries)
        self.indexes = indexes
        self.lowers = lowers
        self.uppers = uppers
        if self._storage_config.get("type") == "dict":
            self._entries = {key: (minhash, size) for key, minhash, size in entries}
        self._minhash_digest = digest
        self._index_initialized = True

    def query(self, minhash: MinHash, size: int) -> Generator[Hashable, None, None]:
        """Giving the MinHash and size of the query set, retrieve
        keys that references sets with containment with respect to
        the query set greater than the threshold.

        Args:
            minhash (MinHash): The MinHash of the query set.
            size (int): The size (number of unique items) of the query set.

        Returns:
            Generator[Hashable, None, None]: an iterator of keys.

        """
        size = _validate_size(size)
        _check_minhash_digest(minhash, self.h, self._minhash_digest)
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            for key in index[r]._query_b(minhash, b):
                yield key

    def __contains__(self, key: Hashable) -> bool:
        """Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.

        """
        return any(any(key in index[r] for r in index) for index in self.indexes)

    def is_empty(self) -> bool:
        """Returns:
        bool: Check if the index is empty.

        """
        return all(all(index[r].is_empty() for r in index) for index in self.indexes)


class MinHashLSHEnsembleUpdateSession:
    """Transactional update session for a memory-backed :class:`MinHashLSHEnsemble`.

    Changes are held in a private snapshot and do not affect queries until
    :meth:`commit` succeeds. Multiple changes to the same key collapse in call
    order. Leaving a ``with`` block without committing cancels the session.
    """

    def __init__(self, ensemble: MinHashLSHEnsemble) -> None:
        self.ensemble = ensemble
        self._entries = dict(ensemble._entries)
        self._digest_value = ensemble._minhash_digest
        self._closed = False
        self._committed = False

    def __enter__(self) -> "MinHashLSHEnsembleUpdateSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._committed:
            self.cancel()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This update session is closed")

    def _validate(self, minhash: MinHash, size: int) -> tuple[MinHash, int]:
        _check_minhash_digest(minhash, self.ensemble.h)
        return minhash.copy(), _validate_size(size)

    def _set_entry(self, key: Hashable, minhash: MinHash, size: int, expected: bool) -> None:
        exists = key in self._entries
        if exists is not expected:
            if expected:
                raise ValueError("The given key does not exist")
            raise ValueError("The given key already exists")
        candidate = self._validate(minhash, size)
        other_count = len(self._entries) - (1 if exists else 0)
        if other_count:
            _check_minhash_digest(minhash, self.ensemble.h, self._digest_value)
        self._entries[key] = candidate
        self._digest_value = _check_minhash_digest(minhash, self.ensemble.h)

    def insert(self, key: Hashable, minhash: MinHash, size: int) -> None:
        """Add a key that is not present in the session's dataset."""
        self._ensure_open()
        self._set_entry(key, minhash, size, expected=False)

    def remove(self, key: Hashable) -> None:
        """Remove a key from the session's dataset."""
        self._ensure_open()
        if key not in self._entries:
            raise ValueError("The given key does not exist")
        del self._entries[key]
        if not self._entries:
            self._digest_value = None

    def replace(self, key: Hashable, minhash: MinHash, size: int) -> None:
        """Replace the MinHash and size of an existing key."""
        self._ensure_open()
        self._set_entry(key, minhash, size, expected=True)

    def update(self, key: Hashable, minhash: MinHash, size: int) -> None:
        """Replace the MinHash and size of an existing key."""
        self.replace(key, minhash, size)

    def commit(self) -> None:
        """Rebuild all partitions from the session's final in-memory dataset.

        The old partitions remain available while new partitions are built. They
        are replaced only after every entry is inserted, so a failed commit
        leaves the old index complete and queryable.
        """
        self._ensure_open()
        entries = [(key, minhash, size) for key, (minhash, size) in self._entries.items()]
        digest = None
        for _key, minhash, _size in entries:
            digest = _check_minhash_digest(minhash, self.ensemble.h, digest)
        indexes, lowers, uppers, digest = self.ensemble._build_indexes(entries)
        self.ensemble.indexes = indexes
        self.ensemble.lowers = lowers
        self.ensemble.uppers = uppers
        self.ensemble._entries = dict(self._entries)
        self.ensemble._minhash_digest = digest
        self.ensemble._index_initialized = True
        self._closed = True
        self._committed = True

    def rollback(self) -> None:
        """Discard all changes made in this session."""
        self.cancel()

    def cancel(self) -> None:
        """Discard all changes made in this session."""
        if self._closed:
            return
        self._entries.clear()
        self._closed = True


if __name__ == "__main__":
    import numpy as np

    xqs = np.exp(np.linspace(-5, 5, 10))
    threshold = 0.5
    max_r = 8
    num_perm = 256
    false_negative_weight, false_positive_weight = 0.5, 0.5
    for xq in xqs:
        b, r = _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight)
        logger.info("threshold: %.2f, xq: %.3f, b: %d, r: %d", threshold, xq, b, r)
