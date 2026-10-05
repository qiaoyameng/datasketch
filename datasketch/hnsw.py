from __future__ import annotations

import heapq
import io
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping, MutableMapping
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

import numpy as np


class HNSWSnapshotError(ValueError):
    """Raised when an HNSW snapshot cannot be safely read."""


_SNAPSHOT_FORMAT = "datasketch-hnsw-snapshot"
_SNAPSHOT_VERSION = 1
_SNAPSHOT_METADATA = "metadata.json"
_SNAPSHOT_POINTS = "points.npy"
_SNAPSHOT_DELETED = "deleted.npy"
_SNAPSHOT_LEVELS = "levels.npy"
_SNAPSHOT_RANDOM_KEYS = "random_state_keys.npy"
_SUPPORTED_SNAPSHOT_DTYPES = (
    np.dtype("float32"),
    np.dtype("float64"),
    np.dtype("int32"),
    np.dtype("int64"),
)
_MAX_SNAPSHOT_KEY_BYTES = 65536
_MAX_SNAPSHOT_METADATA_BYTES = 64 * 1024 * 1024
_MAX_SNAPSHOT_NPY_HEADER_BYTES = 65536
_DEFAULT_MAX_POINTS = 1_000_000
_DEFAULT_MAX_DIMENSION = 65_536
_DEFAULT_MAX_EDGES = 100_000_000
_DEFAULT_MAX_LAYER_NODES = 10_000_000
_SNAPSHOT_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _snapshot_require(condition: bool, message: str) -> None:
    if not condition:
        raise HNSWSnapshotError(message)


def _snapshot_is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _snapshot_valid_identifier(identifier: str) -> bool:
    return isinstance(identifier, str) and _SNAPSHOT_IDENTIFIER_PATTERN.fullmatch(identifier) is not None


def _snapshot_native_dtype(dtype: np.dtype) -> bool:
    native_character = "<" if sys.byteorder == "little" else ">"
    return dtype.byteorder in ("=", "|", native_character)


def _encode_snapshot_key(key: Hashable) -> dict[str, str]:
    if isinstance(key, bool):
        raise HNSWSnapshotError("Snapshot keys must be str or int, not bool.")
    if isinstance(key, str):
        encoded_key = key.encode("utf-8")
        _snapshot_require(len(encoded_key) <= _MAX_SNAPSHOT_KEY_BYTES, "Snapshot key is too large.")
        return {"type": "string", "value": key}
    if isinstance(key, int):
        encoded_key = str(key)
        _snapshot_require(len(encoded_key.encode("ascii")) <= _MAX_SNAPSHOT_KEY_BYTES, "Snapshot key is too large.")
        return {"type": "integer", "value": encoded_key}
    raise HNSWSnapshotError(f"Snapshot keys must be str or int, got {type(key).__name__}.")


def _decode_snapshot_key(encoded_key: Any) -> Hashable:
    _snapshot_require(isinstance(encoded_key, dict), "Snapshot key must be explicitly encoded.")
    _snapshot_require(set(encoded_key) == {"type", "value"}, "Snapshot key has invalid fields.")
    key_type = encoded_key["type"]
    key_value = encoded_key["value"]
    if key_type == "string":
        _snapshot_require(isinstance(key_value, str), "String snapshot key must be a string.")
        encoded_string = key_value.encode("utf-8")
        _snapshot_require(len(encoded_string) <= _MAX_SNAPSHOT_KEY_BYTES, "Snapshot key is too large.")
        return key_value
    if key_type == "integer":
        _snapshot_require(isinstance(key_value, str), "Integer snapshot key must be a decimal string.")
        _snapshot_require(
            re.fullmatch(r"-?(?:0|[1-9][0-9]*)", key_value) is not None,
            "Integer snapshot key must use canonical decimal encoding.",
        )
        _snapshot_require(len(key_value.encode("ascii")) <= _MAX_SNAPSHOT_KEY_BYTES, "Snapshot key is too large.")
        return int(key_value)
    raise HNSWSnapshotError(f"Unsupported snapshot key type {key_type!r}.")


def _reject_snapshot_json_constant(value: str) -> None:
    raise HNSWSnapshotError(f"Invalid JSON constant {value!r} in snapshot metadata.")


def _reject_snapshot_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _snapshot_require(key not in result, f"Duplicate metadata field {key!r}.")
        result[key] = value
    return result


def _read_snapshot_npy(zf: zipfile.ZipFile, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    info = zf.getinfo(name)
    expected_size = int(math.prod(shape)) * dtype.itemsize
    member_limit = expected_size + _MAX_SNAPSHOT_NPY_HEADER_BYTES
    _snapshot_require(
        info.file_size <= member_limit,
        f"Snapshot array {name!r} exceeds its declared shape budget.",
    )
    data = _read_snapshot_member(zf, info, member_limit)
    stream = io.BytesIO(data)
    try:
        magic = np.lib.format.read_magic(stream)
        if magic == (1, 0):
            array_shape, fortran_order, array_descr = np.lib.format.read_array_header_1_0(stream)
        elif magic in ((2, 0), (3, 0)):
            array_shape, fortran_order, array_descr = np.lib.format.read_array_header_2_0(stream)
        else:
            raise HNSWSnapshotError(f"Unsupported NPY version {magic} in {name!r}.")
    except HNSWSnapshotError:
        raise
    except Exception as exc:
        raise HNSWSnapshotError(f"Invalid NPY header for {name!r}.") from exc
    actual_dtype = np.dtype(array_descr)
    _snapshot_require(
        actual_dtype == dtype and _snapshot_native_dtype(actual_dtype),
        f"Snapshot array {name!r} has an unsupported dtype.",
    )
    _snapshot_require(tuple(array_shape) == shape, f"Snapshot array {name!r} has an invalid shape.")
    _snapshot_require(not fortran_order, f"Snapshot array {name!r} must use C order.")
    _snapshot_require(
        len(data) <= expected_size + _MAX_SNAPSHOT_NPY_HEADER_BYTES,
        f"Snapshot array {name!r} exceeds its declared shape budget.",
    )
    stream.seek(0)
    try:
        array = np.load(stream, allow_pickle=False, mmap_mode=None)
    except Exception as exc:
        raise HNSWSnapshotError(f"Invalid NPY data for {name!r}.") from exc
    _snapshot_require(array.dtype == dtype and array.shape == shape, f"Snapshot array {name!r} is malformed.")
    _snapshot_require(stream.tell() == len(data), f"Snapshot array {name!r} has trailing data.")
    return array


def _read_snapshot_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    _snapshot_require(
        info.file_size <= limit,
        f"Snapshot member {info.filename!r} exceeds its size budget.",
    )
    chunks = []
    total_size = 0
    with zf.open(info) as member:
        while True:
            chunk = member.read(65536)
            if not chunk:
                break
            total_size += len(chunk)
            _snapshot_require(total_size <= limit, f"Snapshot member {info.filename!r} exceeds its size budget.")
            chunks.append(chunk)
    return b"".join(chunks)


class _Layer:
    """A graph layer in the HNSW index. This is a dictionary-like object
    that maps a key to a dictionary of neighbors.

    Args:
        key (Hashable): The first key to insert into the graph.

    """

    def __init__(self, key: Hashable) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: dict[Hashable, dict[Hashable, float]] = {key: {}}

    def __contains__(self, key: Hashable) -> bool:
        return key in self._graph

    def __getitem__(self, key: Hashable) -> dict[Hashable, float]:
        return self._graph[key]

    def __setitem__(self, key: Hashable, value: dict[Hashable, float]) -> None:
        self._graph[key] = value

    def __delitem__(self, key: Hashable) -> None:
        del self._graph[key]

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _Layer):
            return False
        return self._graph == __value._graph

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._graph)

    def copy(self) -> _Layer:
        """Create a copy of the layer."""
        new_layer = _Layer(None)
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> set[Hashable]:
        reverse_edges = set()
        for neighbor, neighbors in self._graph.items():
            if key in neighbors:
                reverse_edges.add(neighbor)
        return reverse_edges


class _LayerWithReversedEdges(_Layer):
    """A graph layer in the HNSW index that also maintains reverse edges.

    Args:
        key (Hashable): The first key to insert into the graph.

    """

    def __init__(self, key: Hashable) -> None:
        # self._graph[key] contains a {j: dist} dictionary,
        # where j is a neighbor of key and dist is distance.
        self._graph: dict[Hashable, dict[Hashable, float]] = {key: {}}
        # self._reverse_edges[key] contains a set of neighbors of key.
        self._reverse_edges: dict[Hashable, set] = {}

    def __setitem__(self, key: Hashable, value: dict[Hashable, float]) -> None:
        old_neighbors = self._graph.get(key, {})
        self._graph[key] = value
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)
        for neighbor in value:
            self._reverse_edges.setdefault(neighbor, set()).add(key)
        if key not in self._reverse_edges:
            self._reverse_edges[key] = set()

    def __delitem__(self, key: Hashable) -> None:
        old_neighbors = self._graph.get(key, {})
        del self._graph[key]
        for neighbor in old_neighbors:
            self._reverse_edges[neighbor].discard(key)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _LayerWithReversedEdges):
            return False
        return self._graph == __value._graph and self._reverse_edges == __value._reverse_edges

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterable[Hashable]:
        return iter(self._graph)

    def copy(self) -> _LayerWithReversedEdges:
        """Create a copy of the layer."""
        new_layer = _LayerWithReversedEdges(None)
        new_layer._graph = {k: dict(v) for k, v in self._graph.items()}
        new_layer._reverse_edges = {k: set(v) for k, v in self._reverse_edges.items()}
        return new_layer

    def get_reverse_edges(self, key: Hashable) -> set[Hashable]:
        return self._reverse_edges[key]


class _Node:
    """A node in the HNSW graph."""

    def __init__(self, key: Hashable, point: np.ndarray, is_deleted=False) -> None:
        self.key = key
        self.point = point
        self.is_deleted = is_deleted

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _Node):
            return False
        return (
            self.key == __value.key
            and np.array_equal(self.point, __value.point)
            and self.is_deleted == __value.is_deleted
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"_Node(key={self.key}, point={self.point}, is_deleted={self.is_deleted})"

    def copy(self) -> _Node:
        return _Node(self.key, self.point, self.is_deleted)


class HNSW(MutableMapping):
    """Hierarchical Navigable Small World (HNSW) graph index for approximate
    nearest neighbor search. This implementation is based on the paper
    "Efficient and robust approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (2016),
    `<https://arxiv.org/abs/1603.09320>`_.

    Args:
        distance_func: A function that takes two vectors and returns a float
            representing the distance between them.
        m (int): The number of neighbors to keep for each node.
        ef_construction (int): The number of neighbors to consider during
            construction.
        m0 (Optional[int]): The number of neighbors to keep for each node at
            the 0th level. If None, defaults to 2 * m.
        seed (Optional[int]): The random seed to use for the random number
            generator.
        reverse_edges (bool): Whether to maintain reverse edges in the graph.
            This speeds up hard remove (:meth:`remove`) but increases memory
            usage and slows down :meth:`insert`.

    Examples:
        Create an HNSW index with Euclidean distance and insert 1000 random
        vectors of dimension 10.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            data = np.random.random_sample((1000, 10))
            index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
            for i, d in enumerate(data):
                index.insert(i, d)

            # Query the index for the 10 nearest neighbors of the first vector.
            index.query(data[0], k=10)

        Create an HNSW index with Jaccard distance and insert 1000 random
        sets of 10 elements each.

        .. code-block:: python

            from datasketch.hnsw import HNSW
            import numpy as np

            # Each set is represented as a 10-element vector of random integers
            # between 0 and 100.
            # Deduplication is handled by the distance function.
            data = np.random.randint(0, 100, size=(1000, 10))
            jaccard_distance = lambda x, y: (
                1.0 - float(len(np.intersect1d(x, y, assume_unique=False))) / float(len(np.union1d(x, y)))
            )
            index = HNSW(distance_func=jaccard_distance)
            for i, d in enumerate(data):
                index[i] = d

            # Query the index for the 10 nearest neighbors of the first set.
            index.query(data[0], k=10)

    """

    def __init__(
        self,
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        m: int = 16,
        ef_construction: int = 200,
        m0: Optional[int] = None,
        seed: Optional[int] = None,
        reversed_edges: bool = False,
    ) -> None:
        self._nodes: OrderedDict[Hashable, _Node] = OrderedDict()
        self._distance_func = distance_func
        self._m = m
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log(m)
        self._graphs: list[_Layer] = []
        self._entry_point = None
        self._random = np.random.RandomState(seed)
        self._reversed_edges = reversed_edges
        self._layer_class = _LayerWithReversedEdges if reversed_edges else _Layer

    def __len__(self) -> int:
        """Return the number of points in the index excluding those
        that were soft-removed.
        """
        return sum(not node.is_deleted for node in self._nodes.values())

    def __contains__(self, key: Hashable) -> bool:
        """Return ``True`` if the index contains the key and it was
        not soft-removed, else ``False``.
        """
        return key in self._nodes and not self._nodes[key].is_deleted

    def __getitem__(self, key: Hashable) -> np.ndarray:
        """Get the point associated with the key. Raises KeyError if the key
        does not exist in the index or it was soft-removed.
        """
        if key not in self:
            raise KeyError(key)
        return self._nodes[key].point

    def __setitem__(self, key: Hashable, value: np.ndarray) -> None:
        """Set the point associated with the key and update the index.
        This is equivalent to calling :meth:`insert` with the key and point.
        """
        self.insert(key, value)

    def __delitem__(self, key: Hashable) -> None:
        """Soft remove the point associated with the key. Raises a KeyError if the
        key does not exist in the index. This is equivalent to calling
        :meth:`remove` with the key.
        """
        self.remove(key)

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys of the index that were not
        soft-removed.
        """
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def reversed(self) -> Iterator[Hashable]:
        """Return a reverse iterator over the keys of the index that were not
        soft-removed.
        """
        return (key for key in reversed(self._nodes) if not self._nodes[key].is_deleted)

    def __eq__(self, __value: object) -> bool:
        """Return True only if the index parameters, random states, keys, points
        points, and index structures are equal, including deleted points.
        """
        if not isinstance(__value, HNSW):
            return False
        # Check if the index parameters are equal.
        if (
            self._distance_func != __value._distance_func
            or self._m != __value._m
            or self._ef_construction != __value._ef_construction
            or self._m0 != __value._m0
            or self._level_mult != __value._level_mult
            or self._entry_point != __value._entry_point
        ):
            return False
        # Check if the random states are equal.
        rand_state_1 = self._random.get_state()
        rand_state_2 = __value._random.get_state()
        for i in range(len(rand_state_1)):
            if isinstance(rand_state_1[i], np.ndarray):
                if not np.array_equal(rand_state_1[i], rand_state_2[i]):
                    return False
            else:
                if rand_state_1[i] != rand_state_2[i]:
                    return False
        # Check if keys and points are equal.
        # Note that deleted points are compared too.
        return (
            all(key in self._nodes for key in __value._nodes)
            and all(key in __value._nodes for key in self._nodes)
            and all(self._nodes[key] == __value._nodes[key] for key in self._nodes)
            and self._graphs == __value._graphs
        )

    def get(self, key: Hashable, default: Optional[np.ndarray] = None) -> Union[np.ndarray, None]:
        """Return the point for key in the index, else default. If default is not
        given and key is not in the index or it was soft-removed, return None.
        """
        if key not in self:
            return default
        return self._nodes[key].point

    def items(self) -> Iterator[tuple[Hashable, np.ndarray]]:
        """Return an iterator of the indexed points that were not soft-removed
        as (key, point) pairs.
        """
        return ((key, node.point) for key, node in self._nodes.items() if not node.is_deleted)

    def keys(self) -> Iterator[Hashable]:
        """Return an iterator of the keys of the index points that were not
        soft-removed.
        """
        return (key for key in self._nodes if not self._nodes[key].is_deleted)

    def values(self) -> Iterator[np.ndarray]:
        """Return an iterator of the index points that were not soft-removed."""
        return (node.point for node in self._nodes.values() if not node.is_deleted)

    def pop(self, key: Hashable, default: Optional[np.ndarray] = None, hard: bool = False) -> np.ndarray:
        """If key is in the index, remove it and return its associated point,
        else return default. If default is not given and key is not in the index
        or it was soft-removed, raise KeyError.
        """
        if key not in self:
            if default is None:
                raise KeyError(key)
            return default
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return point

    def popitem(self, last: bool = True, hard: bool = False) -> tuple[Hashable, np.ndarray]:
        """Remove and return a (key, point) pair from the index. Pairs are
        returned in LIFO order if `last` is true or FIFO order if false.
        If the index is empty or all points are soft-removed, raise KeyError.

        Note:
            In versions of Python before 3.7, the order of items in the index
            is not guaranteed. This method will remove and return an arbitrary
            (key, point) pair.

        """
        if not self._nodes:
            raise KeyError("popitem(): index is empty")
        if last:
            key = next(
                (key for key in reversed(self._nodes) if not self._nodes[key].is_deleted),
                None,
            )
        else:
            key = next((key for key in self._nodes if not self._nodes[key].is_deleted), None)
        if key is None:
            raise KeyError("popitem(): index is empty")
        point = self._nodes[key].point
        self.remove(key, hard=hard)
        return key, point

    def clear(self) -> None:
        """Clear the index of all data points. This will not reset the random
        number generator.
        """
        self._nodes = {}
        self._graphs = []
        self._entry_point = None

    def copy(self) -> HNSW:
        """Create a copy of the index. The copy will have the same parameters
        as the original index and the same keys and points, but will not share
        any index data structures (i.e., graphs) with the original index.
        The new index's random state will start from a copy of the original
        index's.
        """
        new_index = HNSW(
            self._distance_func,
            m=self._m,
            ef_construction=self._ef_construction,
            m0=self._m0,
            reversed_edges=self._reversed_edges,
        )
        new_index._nodes = OrderedDict((key, node.copy()) for key, node in self._nodes.items())
        new_index._graphs = [layer.copy() for layer in self._graphs]
        new_index._entry_point = self._entry_point
        new_index._random.set_state(self._random.get_state())
        return new_index

    def update(self, other: Union[Mapping, HNSW]) -> None:
        """Update the index with the points from the other Mapping or HNSW object,
        overwriting existing keys.

        Args:
            other (Union[Mapping, HNSW]): The other Mapping or HNSW object.

        Examples:
            Create an HNSW index with a dictionary of points.

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert 1000 points.
                index.update({i: d for i, d in enumerate(data)})

            Create an HNSW index with another HNSW index.

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert 1000 points.
                index1.update({i: d for i, d in enumerate(data)})

                # Update index2 with the points from index1.
                index2.update(index1)

        """
        for key, point in other.items():
            self.insert(key, point)

    def setdefault(self, key: Hashable, default: np.ndarray) -> np.ndarray:
        """If key is in the index and it was not soft-removed, return
        its associated point. If not, insert
        key with a value of default and return default. default cannot be None.
        """
        if default is None:
            raise ValueError("Default value cannot be None.")
        if key not in self._nodes or self._nodes[key].is_deleted:
            self.insert(key, default)
        return self._nodes[key]

    def insert(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """Add a new point to the index.

        Args:
            key (Hashable): The key of the new point. If the key already exists in the
                index, the point will be updated and the index will be repaired.
            new_point (np.ndarray): The new point to add to the index.
            ef (Optional[int]): The number of neighbors to consider during insertion.
                If None, use the construction ef.
            level (Optional[int]): The level at which to insert the new point.
                If None, the level will be chosen automatically.

        """
        if ef is None:
            ef = self._ef_construction
        if key in self._nodes:
            if self._nodes[key].is_deleted:
                self._nodes[key].is_deleted = False
            self._update(key, new_point, ef)
            return
        # level is the level at which we insert the element.
        if level is None:
            level = int(-np.log(self._random.random_sample()) * self._level_mult)
        self._nodes[key] = _Node(key, new_point)
        if self._entry_point is not None:  # The HNSW is not empty, we have an entry point
            dist = self._distance_func(new_point, self._nodes[self._entry_point].point)
            point = self._entry_point
            # For all levels in which we dont have to insert elem,
            # we search for the closest neighbor using greedy search.
            for layer in reversed(self._graphs[level + 1 :]):
                point, dist = self._search_ef1(new_point, point, dist, layer, allow_soft_deleted=True)
            # Entry points for search at each level to insert.
            entry_points = [(-dist, point)]
            for layer in reversed(self._graphs[: level + 1]):
                # Maximum number of neighbors to keep at this level.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                # Search this layer for neighbors to insert, and update entry points
                # for the next level.
                entry_points = self._search_base_layer(new_point, entry_points, layer, ef, allow_soft_deleted=True)
                # Insert the new node into the graph with out-going edges.
                # We prune the out-going edges to keep only the top level_m neighbors.
                layer[key] = {
                    p: d for d, p in self._heuristic_prune([(-mdist, p) for mdist, p in entry_points], level_m)
                }
                # For each neighbor of the new node, we insert the new node as a neighbor.
                for neighbor_key, dist in layer[key].items():
                    layer[neighbor_key] = {
                        p: d
                        # We prune the edges to keep only the top level_m neighbors
                        # based on heuristic.
                        for d, p in self._heuristic_prune(
                            [(d, p) for p, d in layer[neighbor_key].items()] + [(dist, key)],
                            level_m,
                        )
                    }
        # For all levels above the current level, we create an empty graph.
        for _ in range(len(self._graphs), level + 1):
            self._graphs.append(self._layer_class(key))
            # We set the entry point for each new level to be the new node.
            self._entry_point = key

    def _update(self, key: Hashable, new_point: np.ndarray, ef: int) -> None:
        """Update the point associated with the key in the index.

        Args:
            key (Hashable): The key of the point.
            new_point (np.ndarray): The new point to update to.
            ef (int): The number of neighbors to consider during insertion.

        Raises:
            KeyError: If the key does not exist in the index.

        """
        if key not in self._nodes:
            raise KeyError(key)
        # Update the point.
        self._nodes[key].point = new_point
        # If the entry point is the only point in the index, we do not need to
        # update the index.
        if self._entry_point == key and len(self._nodes) == 1:
            return
        for layer in self._graphs:
            if key not in layer:
                break
            layer_m = self._m if layer is not self._graphs[0] else self._m0
            # Create a set of points in the 2nd-degree neighborhood of the key.
            neighborhood_keys = set([key])
            for p in layer[key]:
                neighborhood_keys.add(p)
                for p2 in layer[p]:
                    neighborhood_keys.add(p2)
            for p in layer[key]:
                # For each neighbor of the key, we connects it with the top ef
                # neighbors in the 2nd-degree neighborhood of the key.
                cands = []
                elem_to_keep = min(ef, len(neighborhood_keys) - 1)
                for candidate_key in neighborhood_keys:
                    if candidate_key == p:
                        continue
                    dist = self._distance_func(self._nodes[candidate_key].point, self._nodes[p].point)
                    if len(cands) < elem_to_keep:
                        heapq.heappush(cands, (-dist, candidate_key))
                    elif dist < -cands[0][0]:
                        heapq.heappushpop(cands, (-dist, candidate_key))
                layer[p] = {p2: d2 for d2, p2 in self._heuristic_prune([(-md, p) for md, p in cands], layer_m)}
        self._repair_connections(key, new_point, ef)

    def _repair_connections(
        self,
        key: Hashable,
        new_point: np.ndarray,
        ef: int,
        key_to_delete: Optional[Hashable] = None,
    ) -> None:
        entry_point = self._entry_point
        entry_point_dist = self._distance_func(new_point, self._nodes[entry_point].point)
        entry_points = [(-entry_point_dist, entry_point)]
        for layer in reversed(self._graphs):
            if key not in layer:
                # Greedy search for the closest neighbor from the highest layer down.
                entry_point, entry_point_dist = self._search_ef1(
                    new_point,
                    entry_point,
                    entry_point_dist,
                    layer,
                    # We allow soft-deleted points to be returned and used as entry point.
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                entry_points = [(-entry_point_dist, entry_point)]
            else:
                # Search for the neighbors at this layer using ef search.
                level_m = self._m if layer is not self._graphs[0] else self._m0
                entry_points = self._search_base_layer(
                    new_point,
                    entry_points,
                    layer,
                    ef + 1,  # We add 1 to ef to account for the point itself.
                    # We allow soft-deleted points to be returned and used as entry point
                    # and neighbor candidates.
                    allow_soft_deleted=True,
                    key_to_hard_delete=key_to_delete,
                )
                # Filter out the updated node itself.
                filtered_candidates = [(-md, p) for md, p in entry_points if p != key]
                # Update the out-going edges of the updated node at this level.
                layer[key] = {p: d for d, p in self._heuristic_prune(filtered_candidates, level_m)}

    def query(
        self,
        query_point: np.ndarray,
        k: Optional[int] = None,
        ef: Optional[int] = None,
    ) -> list[tuple[Hashable, float]]:
        """Search for the k nearest neighbors of the query point.

        Args:
            query_point (np.ndarray): The query point.
            k (Optional[int]): The number of neighbors to return. If None, return
                all neighbors found.
            ef (Optional[int]): The number of neighbors to consider during search.
                If None, use the construction ef.

        Returns:
            list[Tuple[Hashable, float]]: A list of (key, distance) pairs for the k
                nearest neighbors of the query point.

        Raises:
            ValueError: If the entry point is not found.

        """
        if ef is None:
            ef = self._ef_construction
        if self._entry_point is None:
            raise ValueError("Entry point not found.")
        entry_point_dist = self._distance_func(query_point, self._nodes[self._entry_point].point)
        entry_point = self._entry_point
        # Search for the closest neighbor from the highest level to the 2nd
        # level using greedy search.
        for layer in reversed(self._graphs[1:]):
            entry_point, entry_point_dist = self._search_ef1(query_point, entry_point, entry_point_dist, layer)
        # Search for the neighbors at the base layer using ef search.
        candidates = self._search_base_layer(query_point, [(-entry_point_dist, entry_point)], self._graphs[0], ef)
        if k is not None:
            # If k is specified, we return the k nearest neighbors.
            candidates = heapq.nlargest(k, candidates)
        else:
            # Otherwise, we return all neighbors found.
            candidates.sort(reverse=True)
        # Return the neighbors as a list of (id, distance) pairs.
        return [(key, -mdist) for mdist, key in candidates]

    def _search_ef1(
        self,
        query_point: np.ndarray,
        entry_point: Hashable,
        entry_point_dist: float,
        layer: _Layer,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> tuple[Hashable, float]:
        """The greedy search algorithm for finding the closest neighbor only.

        Args:
            query (np.ndarray): The query point.
            entry_point (Hashable): The entry point for the search.
            entry_point_dist (float): The distance from the query point to the
                entry point.
            layer (_Layer): The graph for the layer.
            allow_soft_deleted (bool): Whether to allow soft-deleted points to
                be returned.
            key_to_hard_delete (Optional[Hashable]): The key of the point to be
                hard-deleted, if any. This point should never be returned.

        Returns:
            Tuple[Hashable, float]: A tuple of (key, distance) representing the closest
                neighbor found.

        """
        candidates = [(entry_point_dist, entry_point)]
        visited = set([entry_point])
        best = entry_point
        best_dist = entry_point_dist
        while candidates:
            # Pop the closest node from the heap.
            dist, curr = heapq.heappop(candidates)
            if dist > best_dist:
                # Terminate the search if the distance to the current closest node
                # is larger than the distance to the best node.
                break
            # Find the neighbors of the current node
            neighbors = [p for p in layer[curr] if p not in visited]
            visited.update(neighbors)
            dists = [self._distance_func(query_point, self._nodes[p].point) for p in neighbors]
            for p, d in zip(neighbors, dists):
                # Update the best node if we find a closer node.
                if d < best_dist:
                    if (not allow_soft_deleted and self._nodes[p].is_deleted) or (p == key_to_hard_delete):
                        # If the neighbor has been deleted or to be hard-deleted,
                        # we don't update the best node but we continue to
                        # explore the neighbor's neighbors.
                        pass
                    else:
                        best, best_dist = p, d
                    # Add the neighbor to the heap.
                    heapq.heappush(candidates, (d, p))
        return best, best_dist

    def _search_base_layer(
        self,
        query_point: np.ndarray,
        entry_points: list[tuple[float, Hashable]],
        layer: _Layer,
        ef: int,
        allow_soft_deleted: bool = False,
        key_to_hard_delete: Optional[Hashable] = None,
    ) -> list[tuple[float, Hashable]]:
        """The ef search algorithm for finding neighbors in a given layer.

        Args:
            query (np.ndarray): The query point.
            entry_points (list[Tuple[float, Hashable]]): A list of (-distance, key) pairs
                representing the entry points for the search.
            layer (_Layer): The graph for the layer.
            ef (int): The number of neighbors to consider during search.
            allow_soft_deleted (bool): Whether to allow soft-deleted points to
                be returned.
            key_to_hard_delete (Optional[Hashable]): The key of the point to be
                hard-deleted, if any. This point should never be returned.

        Returns:
            list[Tuple[float, Hashable]]: A heap of (-distance, key) pairs representing
                the neighbors found.

        Note:
            When used together with :meth:`_search_ef1`, the input entry_points
            may contain soft-deleted points depending on the flag used in
            :meth:`_search_ef1`. Therefore, the output entry_points may contain
            soft-deleted points even if allow_soft_deleted is False. Therefore,
            the caller should check input entry_points for soft-deleted
            points if necessary.

        """
        # candidates is a heap of (distance, key) pairs.
        candidates = [(-mdist, p) for mdist, p in entry_points]
        heapq.heapify(candidates)

        visited = set(p for _, p in entry_points)
        while candidates:
            # Pop the closest node from the heap.
            dist, curr_key = heapq.heappop(candidates)

            # If the neighbor has been marked as deleted, we ,

            # Terminate the search if the distance to the current closest node
            # is larger than the distance to the best node.
            closet_dist = -entry_points[0][0]
            if dist > closet_dist:
                break
            # Find the neighbors of the current node
            neighbors = [p for p in layer[curr_key] if p not in visited]
            visited.update(neighbors)
            dists = [self._distance_func(query_point, self._nodes[p].point) for p in neighbors]
            for p, dist in zip(neighbors, dists):
                if (not allow_soft_deleted and self._nodes[p].is_deleted) or (p == key_to_hard_delete):
                    if dist <= closet_dist:
                        # If the neighbor has been deleted or to be deleted,
                        # we add it to the heap to explore its neighbors but
                        # do not add it to the entry points.
                        heapq.heappush(candidates, (dist, p))
                elif len(entry_points) < ef:
                    heapq.heappush(candidates, (dist, p))
                    # If we have not found enough neighbors, we add the neighbor
                    # to the heap.
                    heapq.heappush(entry_points, (-dist, p))
                    closet_dist = -entry_points[0][0]
                elif dist <= closet_dist:
                    heapq.heappush(candidates, (dist, p))
                    # If we have found enough neighbors, we replace the worst
                    # neighbor with the neighbor if the neighbor is closer.
                    heapq.heapreplace(entry_points, (-dist, p))
                    closet_dist = -entry_points[0][0]

        return entry_points

    def _heuristic_prune(self, candidates: list[tuple[float, Hashable]], max_size: int) -> list[tuple[float, Hashable]]:
        """Prune the potential neigbors to keep only the top max_size neighbors.
        This algorithm is based on hnswlib's heuristic pruning algorithm:
        <https://github.com/nmslib/hnswlib/blob/978f7137bc9555a1b61920f05d9d0d8252ca9169/hnswlib/hnswalg.h#L382>`_.

        Args:
            candidates (list[tuple[float, Hashable]]): A list of (distance, key) pairs
                representing the potential neighbors.
            max_size (int): The maximum number of neighbors to keep.

        Returns:
            list[Tuple[float, Hashable]]: A list of (distance, key) pairs representing
                the pruned neighbors.

        """
        if len(candidates) < max_size:
            # If the number of entry points is smaller than max_size, we return
            # all entry points.
            return candidates
        # candidates is a heap of (distance, key) pairs.
        heapq.heapify(candidates)
        pruned = []
        while candidates:
            if len(pruned) >= max_size:
                break
            # Pop the closest node from the heap.
            candidate_dist, candidate_key = heapq.heappop(candidates)
            good = True
            for _, selected_key in pruned:
                dist_to_selected_neighbor = self._distance_func(
                    self._nodes[selected_key].point, self._nodes[candidate_key].point
                )
                if dist_to_selected_neighbor < candidate_dist:
                    good = False
                    break
            if good:
                pruned.append((candidate_dist, candidate_key))
        return pruned

    def remove(
        self,
        key: Hashable,
        hard: bool = False,
        ef: Optional[int] = None,
    ) -> None:
        """Remove a point from the index. This removal algorithm is based on
        the discussion on `hnswlib issue #4`_.

        There are two versions:

        * *soft remove*: the point is marked as removed from the index, but its
        data and out-going edges are kept. Future queries will not return
        the point and no new edge will direct to this point,
        but the point will still be used for graph traversal.
        This is the default behavior.
        * *hard remove*: the point is removed from the index and its data and
        out-going edges are also removed. Points with out-going edges pointing
        to the deleted point will have their out-going edges
        re-assigned using the same pruning algorithm as :meth:`insert` during
        point update.

        In both versions, if the deleted point is the current entry point,
        the entry point will be re-assigned to the next point in the highest
        layer that has other points beside the current entry point.

        Subsequent soft removes without a hard remove of the same point will
        not affect the index, **unless the point was the only point in the index
        as removing it clears the index**. This is different from :meth:`pop`
        which will always raise a KeyError if the key was removed.

        Subsequent hard removes of the same point will
        raise a KeyError. If the point is soft removed and then hard removed,
        the point will be removed from the index.
        Use :meth:`clean` for removing all soft removed points from the index.

        Args:
            key (Hashable): The key of the point to remove.
            hard (bool): If True, perform a hard remove. Otherwise, perform a
                soft remove.
            ef (Optional[int]): The number of neighbors to consider during
                re-assignment. If None, use the construction ef. This argument
                is only used when hard is True.

        Raises:
            KeyError: If the index is empty or the key does not exist in the
                index and was not soft removed.

        Example:

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data = np.random.random_sample((1000, 10))
                index = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index.update({i: d for i, d in enumerate(data)})

                # Soft remove a point with key = 0.
                index.remove(0)

                # Soft remove the same point again will not change the index
                # because the index is not empty.
                index.remove(0)

                print(0 in index)  # False

                # Hard remove the point.
                index.remove(0, hard=True)

                # Hard remove the same point again will raise a KeyError.
                # index.remove(0, hard=True)

                # Soft remove rest of the points from the index.
                for i in range(1, 1000):
                    index.remove(i)

                print(len(index))  # 0

                # Clean the index to hard remove all soft removed points.
                index.clean()

        .. _hnswlib issue #4: https://github.com/nmslib/hnswlib/issues/4

        """
        if not self._nodes or key not in self._nodes:
            raise KeyError(key)
        if self._entry_point == key:
            # If the point is the entry point, we re-assign the entry point
            # to the next point in the highest layer besides the point to be
            # deleted.
            new_entry_point = None
            for layer in reversed(list(self._graphs)):
                new_entry_point = next(
                    (p for p in layer if p != key and not self._nodes[p].is_deleted),
                    None,
                )
                if new_entry_point is not None:
                    break
                # As the layer is going to be empty after deletion, we remove it.
                self._graphs.pop()
            if new_entry_point is None:
                # If the point to be deleted is the only point in the index,
                # we clear the index.
                self.clear()
                return
            # Update the entry point.
            self._entry_point = new_entry_point
        if ef is None:
            ef = self._ef_construction

        # Soft remove.
        self._nodes[key].is_deleted = True
        if not hard:
            return

        # Hard remove.
        # Find all points that have out-going edges pointing to the deleted point.
        keys_to_update = set()
        for layer in self._graphs:
            if key not in layer:
                break
            keys_to_update.update(layer.get_reverse_edges(key))
        # Re-assign edges for these points.
        for key_to_update in keys_to_update:
            self._repair_connections(
                key_to_update,
                self._nodes[key_to_update].point,
                ef,
                key_to_delete=key,
            )
        # Remove the point to be deleted from the grpah.
        for layer in self._graphs:
            if key not in layer:
                break
            del layer[key]
        # Remove the point from the index.
        del self._nodes[key]

    def clean(self, ef: Optional[int] = None) -> None:
        """Remove all soft removed points from the index.

        Args:
            ef (Optional[int]): The number of neighbors to consider during
                re-assignment. If None, use the construction ef.

        """
        keys_to_remove = list(key for key in self._nodes if self._nodes[key].is_deleted)
        for key in keys_to_remove:
            self.remove(key, ef=ef, hard=True)

    def merge(self, other: HNSW) -> HNSW:
        """Create a new index by merging the current index with another index.
        The new index will contain all points from both indexes.
        If a point exists in both, the point from the other index will be used.
        The new index will have the same parameters as the current index and
        a copy of the current index's random state.

        Args:
            other (HNSW): The other index to merge with.

        Returns:
            HNSW: A new index containing all points from both indexes.

        Example:

            .. code-block:: python

                from datasketch.hnsw import HNSW
                import numpy as np

                data1 = np.random.random_sample((1000, 10))
                data2 = np.random.random_sample((1000, 10))
                index1 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))
                index2 = HNSW(distance_func=lambda x, y: np.linalg.norm(x - y))

                # Batch insert data into the indexes.
                index1.update({i: d for i, d in enumerate(data1)})
                index2.update({i + len(data1): d for i, d in enumerate(data2)})

                # Merge the indexes.
                index = index1.merge(index2)

        """
        new_index = self.copy()
        new_index.update(other)
        return new_index

    def save_snapshot(
        self,
        file: Union[str, os.PathLike[str], Any],
        distance_identifier: str,
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: Optional[int] = None,
    ) -> None:
        """Save a versioned, pickle-free HNSW snapshot.

        The snapshot stores points, adjacency lists, construction parameters,
        the NumPy random state, and soft-delete markers. Distance functions are
        represented only by ``distance_identifier`` and must be supplied again
        when loading.

        Args:
            file (Union[str, os.PathLike, Any]): A path or a writable binary
                file object. Paths are replaced atomically.
            distance_identifier (str): A stable identifier for the distance
                function. It is checked by the loading caller, never imported.
            compression (int): ZIP compression method.
            compresslevel (Optional[int]): Compression level, when supported.

        Raises:
            HNSWSnapshotError: If the index or snapshot arguments cannot be
                represented by the safe snapshot format.
        """
        _snapshot_require(_snapshot_valid_identifier(distance_identifier), "Invalid distance identifier.")
        snapshot_data = self._build_snapshot_data(distance_identifier)
        if isinstance(file, (str, os.PathLike)):
            path = os.fspath(file)
            directory = os.path.dirname(path) or "."
            fd, temporary_name = tempfile.mkstemp(prefix=".hnsw-snapshot-", dir=directory)
            os.close(fd)
            try:
                with zipfile.ZipFile(
                    temporary_name,
                    mode="w",
                    compression=compression,
                    compresslevel=compresslevel,
                ) as zf:
                    self._write_snapshot_data(zf, snapshot_data)
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        else:
            with zipfile.ZipFile(
                file,
                mode="w",
                compression=compression,
                compresslevel=compresslevel,
            ) as zf:
                self._write_snapshot_data(zf, snapshot_data)

    @classmethod
    def load_snapshot(
        cls,
        file: Union[str, os.PathLike[str], Any],
        distance_func: Callable[[np.ndarray, np.ndarray], float],
        distance_identifier: str,
        max_points: int = _DEFAULT_MAX_POINTS,
        max_dimension: int = _DEFAULT_MAX_DIMENSION,
        max_vector_values: int = 100_000_000,
        max_edges: int = _DEFAULT_MAX_EDGES,
        max_layer_nodes: int = _DEFAULT_MAX_LAYER_NODES,
    ) -> HNSW:
        """Load and fully validate a snapshot before returning a new index.

        Args:
            file (Union[str, os.PathLike, Any]): A path or readable binary
                file object.
            distance_func (Callable[[np.ndarray, np.ndarray], float]): The
                distance function used to rebuild adjacency distances.
            distance_identifier (str): Identifier that must match the snapshot.
            max_points (int): Maximum permitted number of stored points.
            max_dimension (int): Maximum permitted vector dimension.
            max_vector_values (int): Maximum permitted point count times
                dimension.
            max_edges (int): Maximum permitted number of stored graph edges.
            max_layer_nodes (int): Maximum permitted total layer memberships.

        Returns:
            HNSW: A newly constructed index.

        Raises:
            HNSWSnapshotError: If validation fails. No partially constructed
                index is returned.
        """
        for name, value in (
            ("max_points", max_points),
            ("max_dimension", max_dimension),
            ("max_vector_values", max_vector_values),
            ("max_edges", max_edges),
            ("max_layer_nodes", max_layer_nodes),
        ):
            _snapshot_require(_snapshot_is_int(value) and value >= 0, f"{name} must be a non-negative integer.")
        _snapshot_require(callable(distance_func), "distance_func must be callable.")
        _snapshot_require(_snapshot_valid_identifier(distance_identifier), "Invalid expected distance identifier.")
        try:
            with zipfile.ZipFile(file, mode="r") as zf:
                return _load_snapshot_from_zip(
                    zf,
                    distance_func,
                    distance_identifier,
                    max_points,
                    max_dimension,
                    max_vector_values,
                    max_edges,
                    max_layer_nodes,
                )
        except Exception as exc:
            if isinstance(exc, HNSWSnapshotError):
                raise
            raise HNSWSnapshotError("Snapshot cannot be read.") from exc

    def _build_snapshot_data(self, distance_identifier: str) -> dict[str, Any]:
        _snapshot_require(_snapshot_is_int(self._m) and self._m > 0, "Invalid HNSW construction parameter m.")
        _snapshot_require(
            _snapshot_is_int(self._ef_construction) and self._ef_construction > 0,
            "Invalid HNSW construction parameter ef_construction.",
        )
        _snapshot_require(_snapshot_is_int(self._m0) and self._m0 > 0, "Invalid HNSW construction parameter m0.")
        _snapshot_require(
            self._m <= 2**31 - 1 and self._m0 <= 2**31 - 1 and self._ef_construction <= 2**31 - 1,
            "Invalid HNSW construction parameter.",
        )
        count = len(self._nodes)
        if count == 0:
            _snapshot_require(self._entry_point is None, "Empty index cannot have an entry point.")
            _snapshot_require(self._graphs == [], "Empty index cannot contain graph layers.")
            points = np.empty((0, 0), dtype=np.dtype("float64"))
        else:
            first_point = next(iter(self._nodes.values())).point
            _snapshot_require(isinstance(first_point, np.ndarray), "Snapshot points must be NumPy arrays.")
            _snapshot_require(first_point.ndim == 1, "Snapshot vectors must be one-dimensional.")
            dtype = first_point.dtype
            _snapshot_require(
                dtype in _SUPPORTED_SNAPSHOT_DTYPES,
                f"Unsupported snapshot vector dtype {dtype.str!r}.",
            )
            _snapshot_require(_snapshot_native_dtype(dtype), "Snapshot vector dtype must use native byte order.")
            dimension = int(first_point.shape[0])
            _snapshot_require(dimension > 0, "Snapshot vectors must have a positive dimension.")
            points = np.empty((count, dimension), dtype=dtype)
            encoded_keys = []
            levels = np.empty(count, dtype=np.int64)
            deleted = np.empty(count, dtype=bool)
            node_indices = {}
            ordered_keys = list(self._nodes.keys())
            for index, (key, node) in enumerate(self._nodes.items()):
                _snapshot_require(isinstance(node, _Node), "Invalid HNSW node.")
                _snapshot_require(node.key == key, "HNSW node key does not match node storage.")
                encoded_keys.append(_encode_snapshot_key(key))
                _snapshot_require(isinstance(node.point, np.ndarray), "Snapshot points must be NumPy arrays.")
                _snapshot_require(
                    node.point.dtype == dtype and node.point.ndim == 1 and node.point.shape == (dimension,),
                    "All snapshot vectors must share one supported dtype and dimension.",
                )
                points[index] = node.point
                deleted[index] = bool(node.is_deleted)
                node_indices[key] = index
                level = next((i for i, layer in enumerate(self._graphs) if key not in layer), len(self._graphs))
                levels[index] = level - 1
            _snapshot_require(self._entry_point in node_indices, "Entry point does not exist.")
            self._validate_snapshot_layers(node_indices, count)
        random_state = self._random.get_state()
        _snapshot_require(
            isinstance(random_state, tuple) and len(random_state) == 5 and random_state[0] == "MT19937",
            "Unsupported NumPy random state.",
        )
        random_keys = np.asarray(random_state[1], dtype=np.uint32)
        _snapshot_require(random_keys.shape == (624,), "Unsupported NumPy random key state.")
        random_position = random_state[2]
        _snapshot_require(_snapshot_is_int(random_position), "Unsupported NumPy random position.")
        has_gauss = random_state[3]
        _snapshot_require(
            _snapshot_is_int(has_gauss) and int(has_gauss) in (0, 1),
            "Unsupported NumPy random Gaussian flag.",
        )
        cached_gauss = random_state[4]
        _snapshot_require(
            isinstance(cached_gauss, (int, float, np.number)) and not isinstance(cached_gauss, (bool, np.bool_)),
            "Unsupported NumPy cached Gaussian value.",
        )
        _snapshot_require(math.isfinite(float(cached_gauss)), "Unsupported NumPy cached Gaussian value.")
        layers = []
        for layer_index, layer in enumerate(self._graphs):
            layer_nodes = np.asarray([node_indices[key] for key in layer], dtype=np.int64)
            offsets = [0]
            targets = []
            for source_index in layer_nodes:
                source_key = ordered_keys[int(source_index)]
                neighbors = layer[source_key]
                for target_key in neighbors:
                    _snapshot_require(target_key in node_indices, "Graph neighbor does not exist.")
                    target_index = node_indices[target_key]
                    _snapshot_require(target_index != source_index, "Graph contains a self edge.")
                    targets.append(target_index)
                offsets.append(len(targets))
            layers.append(
                {
                    "nodes": int(len(layer_nodes)),
                    "edges": len(targets),
                    "nodes_name": f"layers/{layer_index}/nodes.npy",
                    "offsets_name": f"layers/{layer_index}/offsets.npy",
                    "targets_name": f"layers/{layer_index}/targets.npy",
                    "_nodes_array": layer_nodes,
                    "_offsets_array": np.asarray(offsets, dtype=np.int64),
                    "_targets_array": np.asarray(targets, dtype=np.int64),
                }
            )
        metadata = {
            "format": _SNAPSHOT_FORMAT,
            "version": _SNAPSHOT_VERSION,
            "distance_identifier": distance_identifier,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "m0": self._m0,
            "reversed_edges": self._reversed_edges,
            "point_count": count,
            "dimension": 0 if count == 0 else dimension,
            "points_dtype": points.dtype.str,
            "entry_point": None if self._entry_point is None else node_indices[self._entry_point],
            "keys": [] if count == 0 else encoded_keys,
            "layers": [{key: value for key, value in layer.items() if not key.startswith("_")} for layer in layers],
            "random_state": {
                "algorithm": "MT19937",
                "keys_name": _SNAPSHOT_RANDOM_KEYS,
                "position": int(random_position),
            "has_gauss": bool(has_gauss),
                "cached_gauss": float(cached_gauss),
            },
        }
        return {
            "metadata": metadata,
            "points": points,
            "deleted": np.empty(0, dtype=bool) if count == 0 else deleted,
            "levels": np.empty(0, dtype=np.int64) if count == 0 else levels,
            "random_keys": random_keys,
            "layers": layers,
        }

    def _validate_snapshot_layers(self, node_indices: dict[Hashable, int], count: int) -> None:
        _snapshot_require(len(self._graphs) > 0, "Non-empty index must contain a base graph layer.")
        previous_nodes = None
        for layer_index, layer in enumerate(self._graphs):
            layer_nodes = set(layer)
            layer_m = self._m0 if layer_index == 0 else self._m
            _snapshot_require(all(key in node_indices for key in layer_nodes), "Graph contains an unknown node.")
            _snapshot_require(len(layer_nodes) == len(layer), "Graph layer contains a duplicate node.")
            if layer_index == 0:
                _snapshot_require(len(layer_nodes) == count, "Base graph must contain every point.")
            else:
                _snapshot_require(layer_nodes.issubset(previous_nodes), "Graph layer skips a lower layer.")
            for source_key in layer:
                _snapshot_require(len(layer[source_key]) <= layer_m, "Graph node exceeds its layer degree limit.")
                for target_key in layer[source_key]:
                    _snapshot_require(target_key in node_indices, "Graph neighbor does not exist.")
                    _snapshot_require(target_key in layer, "Graph edge points outside its layer.")
            previous_nodes = layer_nodes

    @staticmethod
    def _write_snapshot_data(zf: zipfile.ZipFile, snapshot_data: dict[str, Any]) -> None:
        def write_array(name: str, array: np.ndarray) -> None:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            zf.writestr(name, buffer.getvalue())

        metadata_bytes = json.dumps(snapshot_data["metadata"], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        zf.writestr(_SNAPSHOT_METADATA, metadata_bytes)
        write_array(_SNAPSHOT_POINTS, snapshot_data["points"])
        write_array(_SNAPSHOT_DELETED, snapshot_data["deleted"])
        write_array(_SNAPSHOT_LEVELS, snapshot_data["levels"])
        write_array(_SNAPSHOT_RANDOM_KEYS, snapshot_data["random_keys"])
        for layer in snapshot_data["layers"]:
            write_array(layer["nodes_name"], layer["_nodes_array"])
            write_array(layer["offsets_name"], layer["_offsets_array"])
            write_array(layer["targets_name"], layer["_targets_array"])


def _load_snapshot_from_zip(
    zf: zipfile.ZipFile,
    distance_func: Callable[[np.ndarray, np.ndarray], float],
    expected_distance_identifier: str,
    max_points: int,
    max_dimension: int,
    max_vector_values: int,
    max_edges: int,
    max_layer_nodes: int,
) -> HNSW:
    infos = zf.infolist()
    names = [info.filename for info in infos]
    _snapshot_require(len(set(names)) == len(names), "Snapshot contains duplicate file names.")
    _snapshot_require(all(not info.is_dir() for info in infos), "Snapshot cannot contain directory entries.")
    _snapshot_require(
        all(not os.path.isabs(name) and ".." not in name.split("/") for name in names),
        "Unsafe snapshot member name.",
    )
    total_size = sum(info.file_size for info in infos)
    _snapshot_require(total_size <= 2 * 1024**3, "Snapshot exceeds total size budget.")
    _snapshot_require(_SNAPSHOT_METADATA in names, "Snapshot metadata is missing.")
    metadata_info = zf.getinfo(_SNAPSHOT_METADATA)
    _snapshot_require(
        metadata_info.file_size <= _MAX_SNAPSHOT_METADATA_BYTES,
        "Snapshot metadata exceeds size budget.",
    )
    try:
        metadata_bytes = _read_snapshot_member(zf, metadata_info, _MAX_SNAPSHOT_METADATA_BYTES)
        metadata = json.loads(
            metadata_bytes,
            parse_constant=_reject_snapshot_json_constant,
            object_pairs_hook=_reject_snapshot_duplicate_json_keys,
        )
    except HNSWSnapshotError:
        raise
    except Exception as exc:
        raise HNSWSnapshotError("Snapshot metadata is invalid JSON.") from exc

    _snapshot_require(isinstance(metadata, dict), "Snapshot metadata must be an object.")
    required_fields = {
        "format",
        "version",
        "distance_identifier",
        "m",
        "ef_construction",
        "m0",
        "reversed_edges",
        "point_count",
        "dimension",
        "points_dtype",
        "entry_point",
        "keys",
        "layers",
        "random_state",
    }
    _snapshot_require(set(metadata) == required_fields, "Snapshot metadata has invalid fields.")
    _snapshot_require(metadata["format"] == _SNAPSHOT_FORMAT, "Unsupported snapshot format.")
    _snapshot_require(metadata["version"] == _SNAPSHOT_VERSION, "Unsupported snapshot version.")
    _snapshot_require(isinstance(metadata["distance_identifier"], str), "Invalid distance identifier.")
    _snapshot_require(
        metadata["distance_identifier"] == expected_distance_identifier,
        "Distance identifier does not match.",
    )
    for parameter_name in ("m", "ef_construction", "m0", "point_count", "dimension"):
        _snapshot_require(
            _snapshot_is_int(metadata[parameter_name]),
            f"Snapshot parameter {parameter_name!r} must be an integer.",
        )
    m = int(metadata["m"])
    ef_construction = int(metadata["ef_construction"])
    m0 = int(metadata["m0"])
    point_count = int(metadata["point_count"])
    dimension = int(metadata["dimension"])
    _snapshot_require(m > 0, "Snapshot m must be positive.")
    _snapshot_require(ef_construction > 0 and m0 > 0, "Snapshot construction parameters must be positive.")
    _snapshot_require(m <= 2**31 - 1 and m0 <= 2**31 - 1, "Snapshot construction parameter is too large.")
    _snapshot_require(ef_construction <= 2**31 - 1, "Snapshot construction parameter is too large.")
    _snapshot_require(0 <= point_count <= max_points, "Snapshot point count exceeds budget.")
    _snapshot_require(0 <= dimension <= max_dimension, "Snapshot dimension exceeds budget.")
    _snapshot_require(point_count == 0 or dimension > 0, "Snapshot vectors must have a positive dimension.")
    _snapshot_require(point_count * dimension <= max_vector_values, "Snapshot vector values exceed budget.")
    _snapshot_require(isinstance(metadata["reversed_edges"], bool), "Snapshot reversed_edges flag must be boolean.")

    dtype_str = metadata["points_dtype"]
    _snapshot_require(isinstance(dtype_str, str), "Snapshot vector dtype must be a string.")
    try:
        dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise HNSWSnapshotError("Snapshot vector dtype is invalid.") from exc
    _snapshot_require(
        dtype in _SUPPORTED_SNAPSHOT_DTYPES and _snapshot_native_dtype(dtype),
        "Unsupported snapshot vector dtype.",
    )
    expected_shape = (point_count, dimension) if point_count > 0 else (0, 0)
    points = _read_snapshot_npy(zf, _SNAPSHOT_POINTS, dtype, expected_shape)
    deleted = _read_snapshot_npy(zf, _SNAPSHOT_DELETED, np.dtype(bool), (point_count,))
    levels = _read_snapshot_npy(zf, _SNAPSHOT_LEVELS, np.dtype(np.int64), (point_count,))
    if dtype.kind == "f":
        _snapshot_require(np.isfinite(points).all(), "Snapshot points contain non-finite values.")

    encoded_keys = metadata["keys"]
    _snapshot_require(isinstance(encoded_keys, list) and len(encoded_keys) == point_count, "Snapshot keys are invalid.")
    keys = [_decode_snapshot_key(encoded_key) for encoded_key in encoded_keys]
    _snapshot_require(len(set(keys)) == len(keys), "Snapshot contains duplicate keys.")

    entry_point = metadata["entry_point"]
    if point_count == 0:
        _snapshot_require(entry_point is None, "Empty snapshot must not have an entry point.")
    else:
        _snapshot_require(_snapshot_is_int(entry_point), "Snapshot entry point must be an integer.")
        _snapshot_require(0 <= int(entry_point) < point_count, "Snapshot entry point is out of range.")
        _snapshot_require(not deleted[int(entry_point)], "Snapshot entry point is soft-deleted.")

    layers_metadata = metadata["layers"]
    _snapshot_require(isinstance(layers_metadata, list), "Snapshot layers must be a list.")
    _snapshot_require(len(layers_metadata) <= max(1, point_count), "Snapshot layer count exceeds budget.")
    if point_count == 0:
        _snapshot_require(layers_metadata == [], "Empty snapshot must not contain layers.")
    else:
        _snapshot_require(layers_metadata, "Non-empty snapshot must contain graph layers.")
    total_layer_nodes = 0
    total_edges = 0
    layer_arrays = []
    for layer_index, layer_metadata in enumerate(layers_metadata):
        _snapshot_require(isinstance(layer_metadata, dict), "Snapshot layer metadata must be an object.")
        expected_layer_fields = {"nodes", "edges", "nodes_name", "offsets_name", "targets_name"}
        _snapshot_require(set(layer_metadata) == expected_layer_fields, "Snapshot layer metadata has invalid fields.")
        for field_name in ("nodes", "edges"):
            _snapshot_require(
                _snapshot_is_int(layer_metadata[field_name]),
                "Snapshot layer size must be an integer.",
            )
        nodes_count = int(layer_metadata["nodes"])
        edges_count = int(layer_metadata["edges"])
        _snapshot_require(nodes_count >= 0 and edges_count >= 0, "Snapshot layer sizes cannot be negative.")
        _snapshot_require(nodes_count > 0, "Snapshot graph layer cannot be empty.")
        total_layer_nodes += nodes_count
        total_edges += edges_count
        _snapshot_require(total_layer_nodes <= max_layer_nodes, "Snapshot layer memberships exceed budget.")
        _snapshot_require(total_edges <= max_edges, "Snapshot edges exceed budget.")
        for field_name in ("nodes_name", "offsets_name", "targets_name"):
            _snapshot_require(isinstance(layer_metadata[field_name], str), "Snapshot layer file name must be a string.")
        nodes_name = f"layers/{layer_index}/nodes.npy"
        offsets_name = f"layers/{layer_index}/offsets.npy"
        targets_name = f"layers/{layer_index}/targets.npy"
        _snapshot_require(layer_metadata["nodes_name"] == nodes_name, "Unexpected snapshot nodes file name.")
        _snapshot_require(layer_metadata["offsets_name"] == offsets_name, "Unexpected snapshot offsets file name.")
        _snapshot_require(layer_metadata["targets_name"] == targets_name, "Unexpected snapshot targets file name.")
        _snapshot_require(nodes_name in names, "Snapshot nodes array is missing.")
        _snapshot_require(offsets_name in names, "Snapshot offsets array is missing.")
        _snapshot_require(targets_name in names, "Snapshot targets array is missing.")
        layer_nodes = _read_snapshot_npy(zf, nodes_name, np.dtype(np.int64), (nodes_count,))
        offsets = _read_snapshot_npy(zf, offsets_name, np.dtype(np.int64), (nodes_count + 1,))
        targets = _read_snapshot_npy(zf, targets_name, np.dtype(np.int64), (edges_count,))
        _snapshot_require(offsets[0] == 0, "Snapshot graph offsets must start at zero.")
        _snapshot_require(offsets[-1] == edges_count, "Snapshot graph offsets do not include all edges.")
        _snapshot_require(bool((offsets[1:] >= offsets[:-1]).all()), "Snapshot graph offsets must be non-decreasing.")
        _snapshot_require(
            bool((layer_nodes >= 0).all()) and bool((layer_nodes < point_count).all()),
            "Snapshot graph node is out of range.",
        )
        _snapshot_require(len(np.unique(layer_nodes)) == nodes_count, "Snapshot graph layer contains a duplicate node.")
        _snapshot_require(
            bool((targets >= 0).all()) and bool((targets < point_count).all()),
            "Snapshot graph neighbor is out of range.",
        )
        if layer_index == 0:
            _snapshot_require(nodes_count == point_count, "Base snapshot graph must contain every point.")
            _snapshot_require(
                nodes_count == 0 or bool(np.array_equal(np.sort(layer_nodes), np.arange(point_count, dtype=np.int64))),
                "Base snapshot graph must contain every point exactly once.",
            )
        else:
            previous_nodes = layer_arrays[-1][0]
            _snapshot_require(
                nodes_count <= len(previous_nodes) and bool(np.isin(layer_nodes, previous_nodes).all()),
                "Snapshot graph has an invalid layer hierarchy.",
            )
        for source_position, source_index in enumerate(layer_nodes):
            start = int(offsets[source_position])
            end = int(offsets[source_position + 1])
            edge_targets = targets[start:end]
            layer_m = m0 if layer_index == 0 else m
            _snapshot_require(len(edge_targets) <= layer_m, "Snapshot graph node exceeds its layer degree limit.")
            _snapshot_require(bool((edge_targets != source_index).all()), "Snapshot graph contains a self edge.")
            _snapshot_require(
                len(np.unique(edge_targets)) == len(edge_targets),
                "Snapshot graph contains duplicate edges.",
            )
            _snapshot_require(bool(np.isin(edge_targets, layer_nodes).all()), "Snapshot graph has a cross-layer edge.")
        layer_arrays.append((layer_nodes, offsets, targets))

    highest_layer = len(layer_arrays) - 1
    if point_count > 0:
        _snapshot_require(
            bool(((levels >= -1) & (levels <= highest_layer)).all()),
            "Snapshot node level is out of range.",
        )
        for level in range(len(layer_arrays)):
            layer_nodes = layer_arrays[level][0]
            if level == 0:
                expected_at_level = np.arange(point_count, dtype=np.int64)
            else:
                previous_nodes = layer_arrays[level - 1][0]
                expected_at_level = previous_nodes[levels[previous_nodes] >= level]
            _snapshot_require(
                len(layer_nodes) == len(expected_at_level)
                and bool(np.array_equal(np.sort(layer_nodes), np.sort(expected_at_level))),
                "Snapshot graph does not match node levels.",
            )
        _snapshot_require(
            levels[int(entry_point)] == highest_layer,
            "Snapshot entry point is not in the highest layer.",
        )

    random_metadata = metadata["random_state"]
    _snapshot_require(isinstance(random_metadata, dict), "Snapshot random state metadata must be an object.")
    random_fields = {"algorithm", "keys_name", "position", "has_gauss", "cached_gauss"}
    _snapshot_require(set(random_metadata) == random_fields, "Snapshot random state metadata has invalid fields.")
    _snapshot_require(random_metadata["algorithm"] == "MT19937", "Unsupported snapshot random algorithm.")
    _snapshot_require(random_metadata["keys_name"] == _SNAPSHOT_RANDOM_KEYS, "Unexpected snapshot random keys file.")
    _snapshot_require(_SNAPSHOT_RANDOM_KEYS in names, "Snapshot random keys array is missing.")
    random_keys = _read_snapshot_npy(zf, _SNAPSHOT_RANDOM_KEYS, np.dtype(np.uint32), (624,))
    random_position = random_metadata["position"]
    _snapshot_require(
        _snapshot_is_int(random_position) and 0 <= int(random_position) <= 624,
        "Invalid snapshot random position.",
    )
    _snapshot_require(isinstance(random_metadata["has_gauss"], bool), "Invalid snapshot random Gaussian flag.")
    cached_gauss = random_metadata["cached_gauss"]
    _snapshot_require(
        isinstance(cached_gauss, (int, float))
        and not isinstance(cached_gauss, bool)
        and math.isfinite(float(cached_gauss)),
        "Invalid snapshot cached Gaussian value.",
    )

    expected_names = {_SNAPSHOT_METADATA, _SNAPSHOT_POINTS, _SNAPSHOT_DELETED, _SNAPSHOT_LEVELS, _SNAPSHOT_RANDOM_KEYS}
    for layer_metadata in layers_metadata:
        expected_names.update(
            (
                layer_metadata["nodes_name"],
                layer_metadata["offsets_name"],
                layer_metadata["targets_name"],
            )
        )
    _snapshot_require(set(names) == expected_names, "Snapshot contains unexpected files.")

    index = HNSW(
        distance_func,
        m=m,
        ef_construction=ef_construction,
        m0=m0,
        seed=0,
        reversed_edges=metadata["reversed_edges"],
    )
    index._nodes = OrderedDict()
    for index_position, key in enumerate(keys):
        index._nodes[key] = _Node(key, points[index_position], bool(deleted[index_position]))
    index._graphs = []
    for layer_nodes, offsets, targets in layer_arrays:
        layer = index._layer_class(keys[int(layer_nodes[0])])
        layer._graph = {}
        if metadata["reversed_edges"]:
            layer._reverse_edges = {}
        for source_position, source_index in enumerate(layer_nodes):
            source_key = keys[int(source_index)]
            start = int(offsets[source_position])
            end = int(offsets[source_position + 1])
            neighbors = {}
            for target_index in targets[start:end]:
                target_key = keys[int(target_index)]
                distance = distance_func(index._nodes[source_key].point, index._nodes[target_key].point)
                _snapshot_require(
                    isinstance(distance, (int, float, np.number))
                    and not isinstance(distance, (bool, np.bool_))
                    and math.isfinite(float(distance)),
                    "Distance function returned an invalid distance.",
                )
                neighbors[target_key] = float(distance)
            layer[source_key] = neighbors
        index._graphs.append(layer)
    index._entry_point = None if entry_point is None else keys[int(entry_point)]
    index._random.set_state(
        (
            "MT19937",
            random_keys,
            int(random_position),
            random_metadata["has_gauss"],
            float(cached_gauss),
        )
    )
    return index
