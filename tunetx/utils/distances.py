"""Distance helpers used across `tunetx`.

These functions compare note groups, rhythm patterns, and descriptor vectors.
They are public because advanced users may want to measure relationships
without building a full graph first.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from functools import lru_cache
from typing import Iterable

import numpy as np


def _as_float_array(values: Iterable[float | int | Fraction]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


@lru_cache(maxsize=None)
def _offset_grid(size: int, tet: int) -> np.ndarray:
    tet_value = float(tet)
    return np.asarray(list(itertools.product((-tet_value, 0.0, tet_value), repeat=size)), dtype=float)


@lru_cache(maxsize=None)
def _non_bijective_extensions(values: tuple[float, ...], extra: int) -> tuple[tuple[float, ...], ...]:
    return tuple(itertools.combinations_with_replacement(values, extra))


def _vector_distance_from_arrays(left: np.ndarray, right: np.ndarray, metric: str = "euclidean") -> float:
    if left.shape != right.shape:
        raise ValueError("distance vectors must have the same shape")

    diff = left - right
    if metric == "euclidean":
        return float(np.sqrt(np.sum(diff * diff)))
    if metric in {"manhattan", "cityblock"}:
        return float(np.abs(diff).sum())
    if metric == "cosine":
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if denom == 0:
            return 0.0
        return float(1.0 - np.dot(left, right) / denom)
    raise ValueError(f"unsupported distance metric: {metric}")


def _vector_distances_from_array(left: np.ndarray, right: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if right.ndim != 2 or right.shape[1] != left.shape[0]:
        raise ValueError("distance vectors must have the same shape")

    diff = right - left
    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=1))
    if metric in {"manhattan", "cityblock"}:
        return np.abs(diff).sum(axis=1)
    if metric == "cosine":
        left_norm = float(np.linalg.norm(left))
        right_norms = np.linalg.norm(right, axis=1)
        denom = left_norm * right_norms
        result = np.zeros(right.shape[0], dtype=float)
        valid = denom != 0
        if np.any(valid):
            result[valid] = 1.0 - (right[valid] @ left) / denom[valid]
        return result
    raise ValueError(f"unsupported distance metric: {metric}")


def generic_vector_distance(
    a: Iterable[float | int | Fraction],
    b: Iterable[float | int | Fraction],
    metric: str = "euclidean",
) -> float:
    """Return a distance between two equal-length numeric vectors.

    Parameters
    ----------
    a, b : iterable of float, int, or Fraction
        Numeric vectors with the same length.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule to use.

    Returns
    -------
    float
        Numeric distance between the two vectors.

    Raises
    ------
    ValueError
        If the vectors have different shapes or the metric is unsupported.

    Examples
    --------
    >>> round(generic_vector_distance([0, 1], [0, 3]), 3)
    2.0
    >>> round(generic_vector_distance([1, 0], [1, 1], metric="cosine"), 3)
    0.293
    """

    left = _as_float_array(a)
    right = _as_float_array(b)
    return _vector_distance_from_arrays(left, right, metric=metric)


def minimal_pitch_class_distance(
    a: Iterable[int | float],
    b: Iterable[int | float],
    tet: int = 12,
    metric: str = "euclidean",
) -> tuple[float, np.ndarray]:
    """Return the smallest voice-leading distance between note groups of the same size.

    Parameters
    ----------
    a, b : iterable of int or float
        Note groups represented as pitch classes, meaning note identities
        without octave.
    tet : int, default=12
        Size of the equal-tempered system.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule to use after trying all rotations and octave-equivalent
        offsets.

    Returns
    -------
    distance : float
        Smallest distance found.
    mapping : numpy.ndarray
        Best matching version of ``b`` after normalization.

    Raises
    ------
    ValueError
        If the note groups do not have the same size.

    Notes
    -----
    This helper is the core distance measure used by voice-leading graphs.

    Examples
    --------
    >>> distance, mapping = minimal_pitch_class_distance((0, 4, 7), (0, 3, 7))
    >>> distance
    1.0
    >>> mapping.tolist()
    [0, 3, 7]
    """

    left = np.sort(np.asarray(list(a), dtype=float))
    right = np.sort(np.asarray(list(b), dtype=float))
    if left.shape != right.shape:
        raise ValueError("pitch collections must have the same cardinality")

    rotations = np.vstack([np.roll(right, rotation) for rotation in range(len(right))])
    offsets = _offset_grid(len(right), tet)
    candidates = np.sort(rotations[:, None, :] + offsets[None, :, :], axis=2).reshape(-1, len(right))
    distances = _vector_distances_from_array(left, candidates, metric=metric)
    best_index = int(np.argmin(distances))
    return float(distances[best_index]), np.sort(candidates[best_index] % tet).astype(int)


def minimal_non_bijective_pitch_class_distance(
    a: Iterable[int | float],
    b: Iterable[int | float],
    tet: int = 12,
    metric: str = "euclidean",
) -> tuple[float, np.ndarray]:
    """Return the smallest distance between note groups of different sizes.

    Parameters
    ----------
    a, b : iterable of int or float
        Note groups represented as pitch classes.
    tet : int, default=12
        Size of the equal-tempered system.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule to use.

    Returns
    -------
    distance : float
        Smallest distance found.
    mapping : numpy.ndarray
        Best expanded mapping for the smaller note group.

    Notes
    -----
    The smaller input is extended with repeated notes so it can be compared to
    the larger one.

    Examples
    --------
    >>> distance, mapping = minimal_non_bijective_pitch_class_distance((0, 4, 7, 11), (0, 4, 7))
    >>> distance
    1.0
    >>> mapping.tolist()
    [0, 0, 4, 7]
    """

    left = np.sort(np.asarray(list(a), dtype=float))
    right = np.sort(np.asarray(list(b), dtype=float))
    if len(left) == len(right):
        return minimal_pitch_class_distance(left, right, tet=tet, metric=metric)

    if len(left) < len(right):
        distance, mapping = minimal_non_bijective_pitch_class_distance(right, left, tet=tet, metric=metric)
        return distance, mapping

    extra = len(left) - len(right)
    if extra == 0:
        return minimal_pitch_class_distance(left, right, tet=tet, metric=metric)

    best_distance = float("inf")
    best_mapping = np.sort(right % tet).astype(int)
    for extension in _non_bijective_extensions(tuple(float(value) for value in right.tolist()), extra):
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
    """Return a distance between rhythm vectors.

    Parameters
    ----------
    a, b : iterable of float, int, or Fraction
        Numeric rhythm representations such as duration histograms.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule to use.

    Returns
    -------
    float
        Numeric distance between the rhythm vectors.

    Examples
    --------
    >>> rhythm_distance([2, 0, 0], [1, 1, 0])
    1.4142135623730951
    """

    return generic_vector_distance(a, b, metric=metric)
