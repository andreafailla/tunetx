"""Collection helpers."""

from __future__ import annotations

from itertools import tee
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def deduplicate(items: Iterable[T]) -> list[T]:
    """Return items without duplicates while preserving order."""

    result: list[T] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def pairwise(items: Iterable[T]) -> Iterator[tuple[T, T]]:
    """Yield neighboring item pairs."""

    first, second = tee(items)
    next(second, None)
    return zip(first, second)
