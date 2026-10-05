import logging
import struct
from collections import Counter, deque
from collections.abc import Generator, Hashable, Iterable
from typing import Optional

import numpy as np

from datasketch.lsh import MinHashLSH, integrate
from datasketch.lshensemble_partition import optimal_partitions
from datasketch.minhash import MinHash, _check_scheme_consistency
from datasketch.storage import _random_name

logger = logging.getLogger(__name__)


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


def _find_partition(uppers, size):
    """Find the index of the partition a set of the given size belongs to,
    given the inclusive upper bounds of the partitions. Partitions without
    an upper bound (None) are empty and skipped. Sizes beyond every upper
    bound are clamped into the last non-empty partition.
    """
    last = None
    for i, u in enumerate(uppers):
        if u is None:
            continue
        last = i
        if size <= u:
            return i
    if last is None:
        raise ValueError("No non-empty partition available")
    return last


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
        rs = self._init_optimal_params(weights)
        # Initialize multiple LSH indexes for each partition
        storage_config = storage_config if storage_config else {"type": "dict"}
        basename = storage_config.get("basename", _random_name(11))
        # Remember the construction parameters so update sessions can build
        # replacement partition indexes with the same configuration.
        self._num_part = num_part
        self._rs = rs
        self._storage_config = storage_config
        self._basename = basename
        self._prepickle = prepickle
        self._storage_type = storage_config["type"]
        # Shadow state of the indexed entries, maintained by `index` and by
        # committed update sessions. Required for repartitioning, which needs
        # the MinHash and size of every indexed set. Only maintained for the
        # in-memory ("dict") storage backend.
        self._entries: dict[Hashable, tuple[MinHash, int]] = {}
        # The permutation scheme of the indexed MinHashes, learned from the
        # first indexed or staged MinHash. Used to reject mixing MinHashes
        # produced by different schemes into the same index.
        self._minhash_scheme: Optional[str] = None
        self.indexes = [
            dict(
                (
                    r,
                    MinHashLSH(
                        num_perm=self.h,
                        params=(int(self.h / r), r),
                        storage_config=self._get_storage_config(basename, storage_config, partition, r),
                        prepickle=prepickle,
                    ),
                )
                for r in rs
            )
            for partition in range(0, num_part)
        ]
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]

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
            ValueError: If the index is not empty or ``entries`` is empty.

        """
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")
        if not isinstance(entries, list):
            queue = deque()
            for key, minhash, size in entries:
                if size <= 0:
                    raise ValueError("Set size must be positive")
                queue.append((key, minhash, size))
            entries = list(queue)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        # Learn the permutation scheme from the entries and reject mixing
        # schemes: partitions are separate MinHashLSH indexes, so without
        # this check two partitions could silently hold different schemes.
        scheme = None
        for _, minhash, _ in entries:
            scheme = _check_scheme_consistency(scheme, minhash)
        # Create optimal partitions.
        sizes, counts = np.array(sorted(Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        # Insert into partitions.
        entries.sort(key=lambda e: e[2])
        curr_part = 0
        for key, minhash, size in entries:
            u = self.uppers[curr_part]
            if size > u:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)
        self._minhash_scheme = scheme
        self._entries = {key: (minhash, size) for key, minhash, size in entries}

    def update_session(self, strategy: str = "rebuild") -> "MinHashLSHEnsembleUpdateSession":
        """Create a context manager for incrementally updating this index:
        inserting new sets, removing existing sets, and replacing the
        MinHash and size of an existing set.

        Staged updates are not visible to :meth:`query` until the session is
        committed. The session commits when the ``with`` block exits without
        an exception, or when :meth:`MinHashLSHEnsembleUpdateSession.commit`
        is called explicitly. Exiting the ``with`` block with an exception,
        or calling :meth:`MinHashLSHEnsembleUpdateSession.cancel`, discards
        all staged updates and leaves the index untouched.

        Args:
            strategy (str): The partitioning strategy applied at commit time.
                ``"rebuild"`` (default) recomputes the optimal partitions from
                the size distribution of the updated collection.
                ``"keep"`` reuses the current partition boundaries; sets whose
                sizes fall outside all existing boundaries are clamped into
                the first or last partition.

        Returns:
            MinHashLSHEnsembleUpdateSession: The context manager.

        Raises:
            ValueError: If the index does not use the in-memory (``dict``)
                storage backend, or if `strategy` is unknown.

        Example:
            .. code-block:: python

                with lshensemble.update_session() as session:
                    session.insert("new_key", minhash, size)
                    session.update("existing_key", new_minhash, new_size)
                    session.remove("old_key")

        """
        return MinHashLSHEnsembleUpdateSession(self, strategy=strategy)

    def _commit_entries(self, entries: dict[Hashable, tuple[MinHash, int]], strategy: str) -> None:
        """Replace the whole index content with `entries`, atomically.

        The replacement partition indexes are built completely before any
        state of this index is touched, so a failure partway through leaves
        the old index fully intact and queryable.
        """
        new_indexes = [
            dict(
                (
                    r,
                    MinHashLSH(
                        num_perm=self.h,
                        params=(int(self.h / r), r),
                        storage_config=self._get_storage_config(self._basename, self._storage_config, partition, r),
                        prepickle=self._prepickle,
                    ),
                )
                for r in self._rs
            )
            for partition in range(self._num_part)
        ]
        new_lowers = [None] * self._num_part
        new_uppers = [None] * self._num_part
        if entries:
            if strategy == "keep" and any(u is not None for u in self.uppers):
                bounds = list(zip(self.lowers, self.uppers))
            else:
                sizes, counts = np.array(sorted(Counter(size for _, size in entries.values()).most_common())).T
                bounds = optimal_partitions(sizes, counts, self._num_part)
            for i, (lower, upper) in enumerate(bounds):
                new_lowers[i], new_uppers[i] = lower, upper
            for key, (minhash, size) in sorted(entries.items(), key=lambda e: e[1][1]):
                curr_part = _find_partition(new_uppers, size)
                for r in new_indexes[curr_part]:
                    new_indexes[curr_part][r].insert(key, minhash)
        # All-or-nothing swap: until this point the old index is untouched.
        self.indexes = new_indexes
        self.lowers = new_lowers
        self.uppers = new_uppers
        self._entries = dict(entries)
        scheme = None
        for minhash, _ in entries.values():
            scheme = _check_scheme_consistency(scheme, minhash)
        self._minhash_scheme = scheme

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
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            # The permutation-scheme guard lives in the inner MinHashLSH
            # (_query_b calls _check_scheme_consistency), so it is checked
            # lazily as this generator is iterated and only for partitions
            # that have had an insert. A query reaching only empty partitions
            # returns nothing regardless, so it cannot yield wrong results.
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
    """Context manager for staging incremental updates to a
    :class:`MinHashLSHEnsemble` and committing them atomically.

    Updates are staged per key and folded in session order: multiple
    operations on the same key collapse to the last one. Every staged
    MinHash is validated against the number of permutations of the index
    and the permutation scheme of the indexed data and of the MinHashes
    previously staged in the same session, so MinHashes from different
    schemes cannot be mixed into one index.

    Staged updates are invisible to queries until commit. Committing
    rebuilds the partition indexes from the merged collection and swaps
    them in all at once, so a failed commit leaves the old index intact.
    """

    def __init__(self, ensemble: MinHashLSHEnsemble, strategy: str = "rebuild") -> None:
        if strategy not in ("rebuild", "keep"):
            raise ValueError("strategy must be 'rebuild' or 'keep', got %r" % (strategy,))
        storage_type = getattr(ensemble, "_storage_type", None)
        if storage_type != "dict" or not hasattr(ensemble, "_entries"):
            raise ValueError(
                "MinHashLSHEnsemble update sessions require the in-memory "
                "('dict') storage backend, got storage type %r" % (storage_type,)
            )
        self._ensemble = ensemble
        self._strategy = strategy
        # Staged final state per key: (minhash, size) if the key must be
        # present after commit, None if it must be absent.
        self._staged: dict[Hashable, Optional[tuple[MinHash, int]]] = {}
        self._scheme = ensemble._minhash_scheme
        self._closed = False

    def __enter__(self) -> "MinHashLSHEnsembleUpdateSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._closed:
            if exc_type is None:
                self.commit()
            else:
                # Discard the staged updates; the index is untouched.
                self._closed = True
                self._staged.clear()
        return False

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("This update session is already closed")

    def _effective_entry(self, key: Hashable) -> Optional[tuple[MinHash, int]]:
        """The entry a key will have after commit given the staged updates,
        or None if the key will be absent.
        """
        if key in self._staged:
            return self._staged[key]
        return self._ensemble._entries.get(key)

    def _validate_entry(self, minhash: MinHash, size: int) -> None:
        if size <= 0:
            raise ValueError("Set size must be positive")
        if len(minhash) != self._ensemble.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self._ensemble.h, len(minhash)))
        self._scheme = _check_scheme_consistency(self._scheme, minhash)

    def insert(self, key: Hashable, minhash: MinHash, size: int) -> None:
        """Stage the insertion of a new set.

        Raises:
            ValueError: If the key already exists in the index or in the
                staged updates, if `size` is not positive, or if `minhash`
                is incompatible with the index.
        """
        self._check_open()
        self._validate_entry(minhash, size)
        if self._effective_entry(key) is not None:
            raise ValueError("The given key already exists")
        self._staged[key] = (minhash, size)

    def remove(self, key: Hashable) -> None:
        """Stage the removal of an existing set.

        Raises:
            ValueError: If the key exists neither in the index nor in the
                staged updates.
        """
        self._check_open()
        if self._effective_entry(key) is None:
            raise ValueError("The given key does not exist")
        self._staged[key] = None

    def update(self, key: Hashable, minhash: MinHash, size: int) -> None:
        """Stage the replacement of the MinHash and size of an existing set.

        Raises:
            ValueError: If the key exists neither in the index nor in the
                staged updates, if `size` is not positive, or if `minhash`
                is incompatible with the index.
        """
        self._check_open()
        self._validate_entry(minhash, size)
        if self._effective_entry(key) is None:
            raise ValueError("The given key does not exist")
        self._staged[key] = (minhash, size)

    def commit(self) -> None:
        """Commit all staged updates to the index atomically, applying the
        partitioning strategy of this session. The session is closed
        afterwards.
        """
        self._check_open()
        entries = dict(self._ensemble._entries)
        for key, entry in self._staged.items():
            if entry is None:
                entries.pop(key, None)
            else:
                entries[key] = entry
        try:
            self._ensemble._commit_entries(entries, self._strategy)
        finally:
            self._closed = True
            self._staged.clear()

    def cancel(self) -> None:
        """Discard all staged updates and close the session. The index is
        left untouched.
        """
        self._check_open()
        self._closed = True
        self._staged.clear()


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
