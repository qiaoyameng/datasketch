"""Offline calibration of :class:`datasketch.MinHashLSH` ``(b, r)`` parameters
using labeled sample pairs.

The default ``MinHashLSH`` parameter optimization
(:func:`datasketch.lsh._optimal_param`) assumes a generic similarity
distribution. When you have sample pairs labeled as *should recall* (the pair
must be a candidate) or *should exclude* (the pair must not be a candidate),
:func:`calibrate_lsh_params` enumerates every legal ``(b, r)`` combination
within the total permutation budget (``b * r <= num_perm``) and picks the one
that minimizes the expected loss::

    expected_loss = miss_cost * P(missed | should recall) * w_pos
                  + false_alarm_cost * P(recalled | should exclude) * w_neg

where both terms are weighted by the per-pair weights and normalized by the
total weight of the evaluation split.

This is an offline tool: it never creates or mutates a ``MinHashLSH`` index.
It returns the recommended parameters together with the full scoring table so
the caller can decide whether and how to rebuild an index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "LSHCalibrationPair",
    "LSHParamScore",
    "LSHCalibrationResult",
    "calibrate_lsh_params",
]


@dataclass(frozen=True)
class LSHCalibrationPair:
    """A labeled pair of MinHash signatures used for calibration.

    Args:
        query: A :class:`datasketch.MinHash` (or any object with a 1-D
            ``hashvalues`` array, or a raw 1-D array of hash values).
        candidate: Same accepted forms as ``query``.
        should_recall (bool): True if the pair must be recalled (returned as
            candidates) by the index; False if the pair must be excluded.
        weight (float): Non-negative importance weight of this pair.
    """

    query: Any
    candidate: Any
    should_recall: bool
    weight: float = 1.0


@dataclass(frozen=True)
class LSHParamScore:
    """Loss breakdown of one ``(b, r)`` combination on one data split.

    Attributes:
        b (int): Number of bands.
        r (int): Number of rows (hash values) per band.
        used_permutations (int): ``b * r`` permutations consumed.
        unused_permutations (int): ``num_perm - b * r`` permutations left over.
        expected_loss (float): ``miss_loss + false_alarm_loss``.
        miss_loss (float): ``miss_cost`` times the weighted fraction of
            should-recall pairs that are not recalled.
        false_alarm_loss (float): ``false_alarm_cost`` times the weighted
            fraction of should-exclude pairs that are recalled.
        missed_weight (float): Total weight of should-recall pairs not recalled.
        false_alarm_weight (float): Total weight of should-exclude pairs recalled.
        total_weight (float): Total weight of all pairs in the split.
        num_pairs (int): Number of pairs in the split.
    """

    b: int
    r: int
    used_permutations: int
    unused_permutations: int
    expected_loss: float
    miss_loss: float
    false_alarm_loss: float
    missed_weight: float
    false_alarm_weight: float
    total_weight: float
    num_pairs: int


@dataclass(frozen=True)
class LSHCalibrationResult:
    """Result of :func:`calibrate_lsh_params`.

    Attributes:
        num_perm (int): The total permutation budget.
        miss_cost (float): Cost of missing a should-recall pair.
        false_alarm_cost (float): Cost of recalling a should-exclude pair.
        params (Tuple[int, int]): The recommended ``(b, r)``, i.e. the
            combination with the lowest expected loss on the training split.
            Ties are broken deterministically in favor of the smallest ``b``
            and then the smallest ``r``.
        unused_permutations (int): ``num_perm - b * r`` for the recommended
            parameters.
        train (LSHParamScore): Score of the recommended parameters on the
            training split.
        train_scores (Tuple[LSHParamScore, ...]): Scores of every legal
            ``(b, r)`` combination on the training split, sorted by
            ``expected_loss`` (ties broken by ``b`` then ``r``).
        validation (Optional[LSHParamScore]): Score of the recommended
            parameters on the validation split, if one was given.
        validation_best (Optional[LSHParamScore]): Best achievable score on
            the validation split (oracle), if a validation split was given.
            Comparing ``validation`` with ``validation_best`` shows how well
            the training choice generalizes.
        validation_scores (Optional[Tuple[LSHParamScore, ...]]): Scores of
            every legal combination on the validation split, if given.
    """

    num_perm: int
    miss_cost: float
    false_alarm_cost: float
    params: Tuple[int, int]
    unused_permutations: int
    train: LSHParamScore
    train_scores: Tuple[LSHParamScore, ...]
    validation: Optional[LSHParamScore] = None
    validation_best: Optional[LSHParamScore] = None
    validation_scores: Optional[Tuple[LSHParamScore, ...]] = None


def _hashvalues_of(obj: Any) -> np.ndarray:
    values = getattr(obj, "hashvalues", obj)
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            "Calibration requires 1-D MinHash hash values; got an array with "
            f"shape {array.shape}. WeightedMinHash (2-D hashvalues) is not "
            "supported."
        )
    return array


def _normalize_pairs(
    pairs: Iterable[Union[LSHCalibrationPair, Sequence[Any]]],
    num_perm: int,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries: List[np.ndarray] = []
    candidates: List[np.ndarray] = []
    labels: List[bool] = []
    weights: List[float] = []
    for item in pairs:
        if isinstance(item, LSHCalibrationPair):
            pair = item
        else:
            values = tuple(item)
            if len(values) == 3:
                pair = LSHCalibrationPair(values[0], values[1], values[2])
            elif len(values) == 4:
                pair = LSHCalibrationPair(values[0], values[1], values[2], values[3])
            else:
                raise ValueError(
                    "Each pair must be an LSHCalibrationPair or a "
                    "(query, candidate, should_recall[, weight]) tuple; "
                    f"got {len(values)} elements."
                )
        if not isinstance(pair.should_recall, (bool, np.bool_)):
            raise ValueError(
                f"should_recall must be a bool, got {pair.should_recall!r}."
            )
        weight = float(pair.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "Pair weight must be a non-negative finite number, "
                f"got {pair.weight!r}."
            )
        query = _hashvalues_of(pair.query)
        candidate = _hashvalues_of(pair.candidate)
        if len(query) != num_perm or len(candidate) != num_perm:
            raise ValueError(
                f"All signatures must have exactly num_perm={num_perm} hash "
                f"values; got lengths {len(query)} and {len(candidate)} in "
                f"the {split_name} split."
            )
        queries.append(query)
        candidates.append(candidate)
        labels.append(bool(pair.should_recall))
        weights.append(weight)
    if not queries:
        raise ValueError(f"The {split_name} split must contain at least one pair.")
    weight_array = np.asarray(weights, dtype=np.float64)
    if float(weight_array.sum()) <= 0.0:
        raise ValueError(
            f"The total pair weight of the {split_name} split must be positive."
        )
    return (
        np.stack(queries),
        np.stack(candidates),
        np.asarray(labels, dtype=bool),
        weight_array,
    )


def _score_split(
    queries: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    num_perm: int,
    miss_cost: float,
    false_alarm_cost: float,
) -> List[LSHParamScore]:
    total_weight = float(weights.sum())
    positive_weight = np.where(labels, weights, 0.0)
    negative_weight = np.where(labels, 0.0, weights)
    scores: List[LSHParamScore] = []
    for b in range(1, num_perm + 1):
        max_r = num_perm // b
        for r in range(1, max_r + 1):
            used = b * r
            equal = (queries[:, :used] == candidates[:, :used]).reshape(-1, b, r)
            recalled = equal.all(axis=2).any(axis=1)
            missed_weight = float(positive_weight[~recalled].sum())
            false_alarm_weight = float(negative_weight[recalled].sum())
            miss_loss = miss_cost * missed_weight / total_weight
            false_alarm_loss = false_alarm_cost * false_alarm_weight / total_weight
            scores.append(
                LSHParamScore(
                    b=b,
                    r=r,
                    used_permutations=used,
                    unused_permutations=num_perm - used,
                    expected_loss=miss_loss + false_alarm_loss,
                    miss_loss=miss_loss,
                    false_alarm_loss=false_alarm_loss,
                    missed_weight=missed_weight,
                    false_alarm_weight=false_alarm_weight,
                    total_weight=total_weight,
                    num_pairs=len(labels),
                )
            )
    return scores


def _sort_key(score: LSHParamScore) -> Tuple[float, int, int]:
    return (score.expected_loss, score.b, score.r)


def calibrate_lsh_params(
    train_pairs: Iterable[Union[LSHCalibrationPair, Sequence[Any]]],
    num_perm: int,
    miss_cost: float = 1.0,
    false_alarm_cost: float = 1.0,
    validation_pairs: Optional[Iterable[Union[LSHCalibrationPair, Sequence[Any]]]] = None,
) -> LSHCalibrationResult:
    """Calibrate ``MinHashLSH`` ``(b, r)`` parameters on labeled sample pairs.

    Every legal combination with ``b * r <= num_perm`` is enumerated. For each
    combination a pair is *recalled* if the two signatures agree on all ``r``
    hash values of at least one of the ``b`` bands (using the first ``b * r``
    permutations). The combination minimizing the weighted expected loss on
    the training split is returned, together with the full scoring table and,
    if given, the performance on a separate validation split.

    This function is purely offline: it does not create or modify any
    ``MinHashLSH`` index.

    Args:
        train_pairs: Labeled pairs used to select the parameters. Each item is
            an :class:`LSHCalibrationPair` or a
            ``(query, candidate, should_recall[, weight])`` tuple, where
            ``query``/``candidate`` are :class:`datasketch.MinHash` objects
            (or raw 1-D hash value arrays) of length ``num_perm``.
        num_perm (int): Total permutation budget. Must equal the number of
            permutations of the MinHash signatures that will be indexed.
        miss_cost (float): Non-negative cost of failing to recall a
            should-recall pair.
        false_alarm_cost (float): Non-negative cost of recalling a
            should-exclude pair.
        validation_pairs: Optional held-out pairs, in the same format as
            ``train_pairs``. The recommended parameters are scored on this
            split so generalization can be checked without peeking during
            selection.

    Returns:
        LSHCalibrationResult: The recommended ``(b, r)`` and the full loss
        breakdown on both splits.

    Raises:
        ValueError: If ``num_perm`` is not positive, a cost is negative, a
            split is empty or has zero total weight, or a signature does not
            have exactly ``num_perm`` hash values.
    """
    if not isinstance(num_perm, (int, np.integer)) or int(num_perm) < 1:
        raise ValueError(f"num_perm must be a positive integer, got {num_perm!r}.")
    num_perm = int(num_perm)
    miss_cost = float(miss_cost)
    false_alarm_cost = float(false_alarm_cost)
    if not np.isfinite(miss_cost) or miss_cost < 0.0:
        raise ValueError(
            f"miss_cost must be a non-negative finite number, got {miss_cost!r}."
        )
    if not np.isfinite(false_alarm_cost) or false_alarm_cost < 0.0:
        raise ValueError(
            "false_alarm_cost must be a non-negative finite number, "
            f"got {false_alarm_cost!r}."
        )

    queries, candidates, labels, weights = _normalize_pairs(
        train_pairs, num_perm, "training"
    )
    train_scores = _score_split(
        queries, candidates, labels, weights, num_perm, miss_cost, false_alarm_cost
    )
    train_scores.sort(key=_sort_key)
    best = train_scores[0]

    validation = None
    validation_best = None
    validation_scores = None
    if validation_pairs is not None:
        v_queries, v_candidates, v_labels, v_weights = _normalize_pairs(
            validation_pairs, num_perm, "validation"
        )
        validation_scores = tuple(
            sorted(
                _score_split(
                    v_queries,
                    v_candidates,
                    v_labels,
                    v_weights,
                    num_perm,
                    miss_cost,
                    false_alarm_cost,
                ),
                key=_sort_key,
            )
        )
        validation_best = validation_scores[0]
        validation = next(
            score
            for score in validation_scores
            if score.b == best.b and score.r == best.r
        )

    return LSHCalibrationResult(
        num_perm=num_perm,
        miss_cost=miss_cost,
        false_alarm_cost=false_alarm_cost,
        params=(best.b, best.r),
        unused_permutations=best.unused_permutations,
        train=best,
        train_scores=tuple(train_scores),
        validation=validation,
        validation_best=validation_best,
        validation_scores=validation_scores,
    )
