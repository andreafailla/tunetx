"""Parse note-length tokens and fraction strings."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

DURATION_TOKENS: dict[str, Fraction] = {
    "w": Fraction(1, 1),
    "h": Fraction(1, 2),
    "q": Fraction(1, 4),
    "e": Fraction(1, 8),
    "s": Fraction(1, 16),
    "t": Fraction(1, 32),
    "wd": Fraction(3, 2),
    "hd": Fraction(3, 4),
    "qd": Fraction(3, 8),
    "ed": Fraction(3, 16),
    "sd": Fraction(3, 32),
    "qt": Fraction(1, 6),
    "et": Fraction(1, 12),
    "st": Fraction(1, 24),
    "qq": Fraction(1, 5),
    "eq": Fraction(1, 10),
    "sq": Fraction(1, 20),
}

DEFAULT_DURATION_BINS: tuple[Fraction, ...] = (
    Fraction(1, 8),
    Fraction(1, 4),
    Fraction(3, 8),
    Fraction(1, 2),
    Fraction(5, 8),
    Fraction(3, 4),
    Fraction(7, 8),
    Fraction(1, 1),
    Fraction(9, 8),
)


def parse_duration_value(value: str | int | float | Fraction) -> Fraction:
    """Parse one note-length value into an exact fraction.

    Parameters
    ----------
    value : str, int, float, or Fraction
        A shorthand token such as ``"q"`` for quarter note, a numeric value, or
        an existing `fractions.Fraction`.

    Returns
    -------
    Fraction
        Exact fractional representation of the note length.

    Examples
    --------
    >>> parse_duration_value("q")
    Fraction(1, 4)
    >>> parse_duration_value("3/8")
    Fraction(3, 8)
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        token = value.strip()
        if token in DURATION_TOKENS:
            return DURATION_TOKENS[token]
        return Fraction(token)
    return Fraction(value)


def parse_duration_sequence(values: Iterable[str | int | float | Fraction]) -> tuple[Fraction, ...]:
    """Parse several note-length values into exact fractions.

    Parameters
    ----------
    values : iterable of str, int, float, or Fraction
        Input note lengths.

    Returns
    -------
    tuple of Fraction
        Parsed sequence in the original order.

    Examples
    --------
    >>> parse_duration_sequence(["q", "e", "e"])
    (Fraction(1, 4), Fraction(1, 8), Fraction(1, 8))
    """

    return tuple(parse_duration_value(value) for value in values)


def parse_fraction_sequence(text: str) -> tuple[Fraction, ...]:
    """Parse a space-separated fraction string.

    Parameters
    ----------
    text : str
        Text such as ``"1/8 1/4 3/8"``.

    Returns
    -------
    tuple of Fraction
        Parsed fractions. An empty string returns an empty tuple.

    Examples
    --------
    >>> parse_fraction_sequence("1/8 1/4 3/8")
    (Fraction(1, 8), Fraction(1, 4), Fraction(3, 8))
    >>> parse_fraction_sequence("")
    ()
    """

    if not text.strip():
        return tuple()
    return tuple(Fraction(part) for part in text.split())


def duration_label(values: Iterable[Fraction]) -> str:
    """Format exact note lengths as a compact label string.

    Parameters
    ----------
    values : iterable of Fraction
        Values to format.

    Returns
    -------
    str
        Space-separated fractions.

    Examples
    --------
    >>> from fractions import Fraction
    >>> duration_label([Fraction(1, 8), Fraction(1, 4), Fraction(3, 8)])
    '1/8 1/4 3/8'
    """

    return " ".join(f"{value.numerator}/{value.denominator}" for value in values)
