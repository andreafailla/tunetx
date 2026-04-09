"""Distance helpers used across tunetx."""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Iterable

import numpy as np


def _as_float_array(values: Iterable[float | int | Fraction]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def generic_vector_distance(
    a: Iterable[float | int | Fraction],
    b: Iterable[float | int | Fraction],
    metric: str = "euclidean",
) -> float:
    """Return a numeric distance between two equal-length vectors."""

    left = _as_float_array(a)
    right = _as_float_array(b)
    if left.shape != right.shape:
        raise ValueError("distance vectors must have the same shape")

    if metric == "euclidean":
        return float(np.linalg.norm(left - right))
    if metric in {"manhattan", "cityblock"}:
        return float(np.abs(left - right).sum())
    if metric == "cosine":
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if denom == 0:
            return 0.0
        return float(1.0 - np.dot(left, right) / denom)
    raise ValueError(f"unsupported distance metric: {metric}")


def minimal_pitch_class_distance(
    a: Iterable[int | float],
    b: Iterable[int | float],
    tet: int = 12,
    metric: str = "euclidean",
) -> tuple[float, np.ndarray]:
    """Return the minimum voice-leading distance between same-cardinality pitch collections."""

    left = np.sort(np.asarray(list(a), dtype=float))
    right = np.sort(np.asarray(list(b), dtype=float))
    if left.shape != right.shape:
        raise ValueError("pitch collections must have the same cardinality")

    best_distance = float("inf")
    best_mapping = right.copy()
    for rotation in range(len(right)):
        rotated = np.roll(right, rotation)
        for offsets in itertools.product((-tet, 0, tet), repeat=len(right)):
            candidate = np.sort(rotated + np.asarray(offsets, dtype=float))
            distance = generic_vector_distance(left, candidate, metric=metric)
            if distance < best_distance:
                best_distance = distance
                best_mapping = candidate % tet
    return best_distance, best_mapping.astype(int)


def minimal_non_bijective_pitch_class_distance(
    a: Iterable[int | float],
    b: Iterable[int | float],
    tet: int = 12,
    metric: str = "euclidean",
) -> tuple[float, np.ndarray]:
    """Return the minimum distance between different-cardinality pitch collections."""

    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    if len(left) == len(right):
        return minimal_pitch_class_distance(left, right, tet=tet, metric=metric)

    if len(left) < len(right):
        distance, mapping = minimal_non_bijective_pitch_class_distance(right, left, tet=tet, metric=metric)
        return distance, mapping

    extra = len(left) - len(right)
    best_distance = float("inf")
    best_mapping = np.sort(right % tet).astype(int)
    for extension in itertools.combinations_with_replacement(right.tolist(), extra):
        candidate = np.asarray(list(right) + list(extension), dtype=float)
        distance, mapping = minimal_pitch_class_distance(left, candidate, tet=tet, metric=metric)
        if distance < best_distance:
            best_distance = distance
            best_mapping = mapping
    return best_distance, best_mapping


def rhythm_distance(
    a: Iterable[float | int | Fraction],
    b: Iterable[float | int | Fraction],
    metric: str = "euclidean",
) -> float:
    """Return a distance between rhythm-duration vectors."""

    return generic_vector_distance(a, b, metric=metric)
