"""Small helpers for working with Python collections."""

from __future__ import annotations

from itertools import tee
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def deduplicate(items: Iterable[T]) -> list[T]:
    """Return items without duplicates while preserving order.

    Parameters
    ----------
    items : iterable
        Values that may contain repeats.

    Returns
    -------
    list
        The first appearance of each item, in the original order.

    Examples
    --------
    >>> deduplicate([1, 2, 2, 3, 1, 4])
    [1, 2, 3, 4]
    """

    result: list[T] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def pairwise(items: Iterable[T]) -> Iterator[tuple[T, T]]:
    """Yield neighboring item pairs.

    Parameters
    ----------
    items : iterable
        Source values.

    Returns
    -------
    iterator of tuple
        Adjacent pairs such as ``(a, b)``, then ``(b, c)``.

    Examples
    --------
    >>> list(pairwise([10, 20, 30]))
    [(10, 20), (20, 30)]
    """

    first, second = tee(items)
    next(second, None)
    return zip(first, second)
