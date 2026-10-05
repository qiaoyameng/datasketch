"""Offline calibration of :class:`datasketch.MinHashLSH` band parameters.

The built-in optimizer of :class:`datasketch.MinHashLSH` picks the number of
bands ``b`` and the band size ``r`` from a theoretical false positive /
false negative analysis at a single Jaccard threshold. This module provides
an *offline* calibration tool that instead evaluates every legal ``(b, r)``
combination within a total permutation budget against labeled sample pairs
(pairs that *should be recalled* and pairs that *should be excluded*), and
selects the combination that minimizes the expected weighted loss.

The tool is pure: it never touches a live index. It returns the recommended
parameters together with the full computation basis (per-combination losses
on both the training and the validation splits, and the number of unused
permutations), so the caller can decide whether to build a new index with
``MinHashLSH(num_perm=..., params=(b, r))``.

Example:

    .. code-block:: python

        from datasketch import MinHash
        from datasketch.lsh_calibration import calibrate_lsh_params

        train = [(m1, m2, True, 1.0), (m1, m3, False, 2.0)]
        validation = [(m4, m5, True, 1.0)]

        result = calibrate_lsh_params(
            num_perm=128,
            train_samples=train,
            validation_samples=validation,
            false_positive_cost=1.0,
            false_negative_cost=10.0,
        )
        print(result.best.b, result.best.r, result.best.unused_permutations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# A sample is either a CalibrationSample or a tuple
# (minhash_a, minhash_b, should_recall) or
# (minhash_a, minhash_b, should_recall, weight).
SampleLike = Union["CalibrationSample", tuple]


@dataclass(frozen=True)
class CalibrationSample:
    """One labeled pair of MinHash sketches used for calibration.

    Args:
        minhash_a: First MinHash of the pair.
        minhash_b: Second MinHash of the pair.
        should_recall: True if the index is expected to recall this pair
            (a match), False if the pair should be excluded.
        weight: Relative importance of this sample in the expected loss.
            Must be positive.
    """

    minhash_a: object
    minhash_b: object
    should_recall: bool
    weight: float = 1.0


@dataclass(frozen=True)
class ParameterEvaluation:
    """The measured outcome of one legal ``(b, r)`` combination.

    Attributes:
        b: Number of bands.
        r: Band size (rows per band).
        used_permutations: ``b * r`` permutations consumed by the bands.
        unused_permutations: ``num_perm - b * r`` permutations left unused.
        train_loss: Expected weighted loss on the training split.
        train_false_positives: Total weight of should-exclude training
            pairs that would be recalled.
        train_false_negatives: Total weight of should-recall training
            pairs that would be missed.
        validation_loss: Expected weighted loss on the validation split,
            or None if no validation samples were given.
        validation_false_positives: Same as ``train_false_positives`` but
            on the validation split (None when not applicable).
        validation_false_negatives: Same as ``train_false_negatives`` but
            on the validation split (None when not applicable).
    """

    b: int
    r: int
    used_permutations: int
    unused_permutations: int
    train_loss: float
    train_false_positives: float
    train_false_negatives: float
    validation_loss: Optional[float] = None
    validation_false_positives: Optional[float] = None
    validation_false_negatives: Optional[float] = None


@dataclass(frozen=True)
class CalibrationResult:
    """The outcome of :func:`calibrate_lsh_params`.

    Attributes:
        best: The evaluation of the selected ``(b, r)`` combination. The
            selection minimizes the *training* loss only; validation numbers
            are reported for every combination but never used for selection.
        evaluations: Every legal ``(b, r)`` combination evaluated, ordered
            by ``(b, r)``.
        num_perm: The total permutation budget.
        false_positive_cost: Cost assigned to recalling a should-exclude pair.
        false_negative_cost: Cost assigned to missing a should-recall pair.
        num_train_samples: Number of training samples used.
        num_validation_samples: Number of validation samples used (0 if none).
    """

    best: ParameterEvaluation
    evaluations: List[ParameterEvaluation]
    num_perm: int
    false_positive_cost: float
    false_negative_cost: float
    num_train_samples: int
    num_validation_samples: int = 0

    @property
    def params(self) -> Tuple[int, int]:
        """The selected ``(b, r)``, ready for ``MinHashLSH(params=...)``."""
        return (self.best.b, self.best.r)

    @property
    def unused_permutations(self) -> int:
        """Permutations left unused by the selected combination."""
        return self.best.unused_permutations

    def to_dict(self) -> dict:
        """Return the full computation basis as plain Python types."""
        return {
            "num_perm": self.num_perm,
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "num_train_samples": self.num_train_samples,
            "num_validation_samples": self.num_validation_samples,
            "best": self.best.__dict__.copy(),
            "evaluations": [e.__dict__.copy() for e in self.evaluations],
        }


def _normalize_sample(sample: SampleLike) -> CalibrationSample:
    if isinstance(sample, CalibrationSample):
        normalized = sample
    elif isinstance(sample, tuple):
        if len(sample) == 3:
            normalized = CalibrationSample(sample[0], sample[1], sample[2])
        elif len(sample) == 4:
            normalized = CalibrationSample(sample[0], sample[1], sample[2], sample[3])
        else:
            raise ValueError(
                "Sample tuples must be (minhash_a, minhash_b, should_recall) "
                "or (minhash_a, minhash_b, should_recall, weight)"
            )
    else:
        raise TypeError(
            "Samples must be CalibrationSample instances or tuples "
            "(minhash_a, minhash_b, should_recall[, weight])"
        )
    if not isinstance(normalized.should_recall, (bool, np.bool_)):
        raise ValueError("should_recall must be a boolean")
    weight = float(normalized.weight)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("Sample weight must be a positive finite number")
    return CalibrationSample(
        normalized.minhash_a,
        normalized.minhash_b,
        bool(normalized.should_recall),
        weight,
    )


def _equality_mask(sample: CalibrationSample, num_perm: int) -> np.ndarray:
    """Boolean mask of per-permutation hash value equality for a pair."""
    for name, minhash in (("minhash_a", sample.minhash_a), ("minhash_b", sample.minhash_b)):
        hashvalues = getattr(minhash, "hashvalues", None)
        if hashvalues is None:
            raise TypeError(f"{name} must be a MinHash-like object with hashvalues")
        if len(hashvalues) < num_perm:
            raise ValueError(
                f"{name} has {len(hashvalues)} permutations, fewer than the "
                f"budget num_perm={num_perm}"
            )
    values_a = np.asarray(sample.minhash_a.hashvalues[:num_perm])
    values_b = np.asarray(sample.minhash_b.hashvalues[:num_perm])
    return values_a == values_b


def _evaluate_split(
    masks: List[np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    b: int,
    r: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> Tuple[float, float, float]:
    """Return (expected_loss, weighted_fp, weighted_fn) for one split."""
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        # Empty split: deterministic zero loss and zero error mass.
        return 0.0, 0.0, 0.0
    used = b * r
    weighted_fp = 0.0
    weighted_fn = 0.0
    for mask, label, weight in zip(masks, labels, weights):
        band_equal = mask[:used].reshape(b, r).all(axis=1)
        recalled = bool(band_equal.any())
        if recalled and not label:
            weighted_fp += weight
        elif not recalled and label:
            weighted_fn += weight
    loss = (
        false_positive_cost * weighted_fp + false_negative_cost * weighted_fn
    ) / total_weight
    return loss, weighted_fp, weighted_fn


def _selection_key(evaluation: ParameterEvaluation) -> tuple:
    # Deterministic tie-breaking: lowest training loss, then fewest used
    # permutations (cheapest index), then smallest b, then smallest r.
    return (
        evaluation.train_loss,
        evaluation.used_permutations,
        evaluation.b,
        evaluation.r,
    )


def calibrate_lsh_params(
    num_perm: int,
    train_samples: Sequence[SampleLike],
    validation_samples: Optional[Sequence[SampleLike]] = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> CalibrationResult:
    """Select the ``(b, r)`` combination with the lowest expected loss.

    Every legal combination with ``b >= 1``, ``r >= 1`` and
    ``b * r <= num_perm`` is evaluated. A pair is predicted as recalled when
    at least one of the ``b`` bands of ``r`` consecutive permutation values
    matches exactly between the two MinHash sketches - the same collision
    rule used by :class:`datasketch.MinHashLSH`. The expected loss of a
    combination on a split is::

        (false_positive_cost * FP + false_negative_cost * FN) / total_weight

    where ``FP`` (``FN``) is the total weight of should-exclude
    (should-recall) pairs that would be recalled (missed).

    The combination is selected on the training split only. When validation
    samples are provided, the same combinations are scored on them and the
    numbers are reported alongside, so an overfit choice is visible instead
    of hidden.

    Args:
        num_perm: Total permutation budget. Every MinHash in the samples
            must have at least this many permutation values.
        train_samples: Labeled pairs used to select the parameters. Each
            sample is a :class:`CalibrationSample` or a tuple
            ``(minhash_a, minhash_b, should_recall[, weight])``.
        validation_samples: Optional held-out labeled pairs, kept explicitly
            separate from the training split by the caller.
        false_positive_cost: Cost of recalling a should-exclude pair.
            Must be non-negative.
        false_negative_cost: Cost of missing a should-recall pair.
            Must be non-negative.

    Returns:
        A :class:`CalibrationResult` with the selected parameters, the
        per-combination losses on both splits, and the number of unused
        permutations. The result is purely informational - no existing
        index is modified.

    Note:
        Edge cases are deterministic: pairs with Jaccard similarity 1.0
        (identical hash values) are always recalled, pairs with
        similarity 0.0 (disjoint sets) are never recalled, an empty split
        yields a zero loss, and splits where all labels are identical simply
        make one of the two error terms always zero. Ties are broken by
        preferring fewer used permutations, then smaller ``b`` and ``r``.
    """
    if not isinstance(num_perm, (int, np.integer)) or int(num_perm) < 1:
        raise ValueError("num_perm must be a positive integer")
    num_perm = int(num_perm)
    false_positive_cost = float(false_positive_cost)
    false_negative_cost = float(false_negative_cost)
    if not np.isfinite(false_positive_cost) or false_positive_cost < 0.0:
        raise ValueError("false_positive_cost must be a non-negative finite number")
    if not np.isfinite(false_negative_cost) or false_negative_cost < 0.0:
        raise ValueError("false_negative_cost must be a non-negative finite number")

    train = [_normalize_sample(s) for s in train_samples]
    validation = (
        [_normalize_sample(s) for s in validation_samples]
        if validation_samples is not None
        else []
    )

    def _prepare(samples: List[CalibrationSample]):
        masks = [_equality_mask(s, num_perm) for s in samples]
        labels = np.array([s.should_recall for s in samples], dtype=bool)
        weights = np.array([s.weight for s in samples], dtype=float)
        return masks, labels, weights

    train_masks, train_labels, train_weights = _prepare(train)
    val_masks, val_labels, val_weights = _prepare(validation)
    has_validation = validation_samples is not None

    evaluations: List[ParameterEvaluation] = []
    for b in range(1, num_perm + 1):
        max_r = num_perm // b
        for r in range(1, max_r + 1):
            used = b * r
            train_loss, train_fp, train_fn = _evaluate_split(
                train_masks,
                train_labels,
                train_weights,
                b,
                r,
                false_positive_cost,
                false_negative_cost,
            )
            if has_validation:
                val_loss, val_fp, val_fn = _evaluate_split(
                    val_masks,
                    val_labels,
                    val_weights,
                    b,
                    r,
                    false_positive_cost,
                    false_negative_cost,
                )
            else:
                val_loss, val_fp, val_fn = None, None, None
            evaluations.append(
                ParameterEvaluation(
                    b=b,
                    r=r,
                    used_permutations=used,
                    unused_permutations=num_perm - used,
                    train_loss=train_loss,
                    train_false_positives=train_fp,
                    train_false_negatives=train_fn,
                    validation_loss=val_loss,
                    validation_false_positives=val_fp,
                    validation_false_negatives=val_fn,
                )
            )

    best = min(evaluations, key=_selection_key)
    return CalibrationResult(
        best=best,
        evaluations=evaluations,
        num_perm=num_perm,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        num_train_samples=len(train),
        num_validation_samples=len(validation) if has_validation else 0,
    )
