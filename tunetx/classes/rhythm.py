"""Classes for working with note lengths and rhythm patterns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Iterable

import numpy as np

from ..utils.parsing import DEFAULT_DURATION_BINS, duration_label, parse_duration_sequence, parse_duration_value


def _floatize(values: Iterable[Fraction]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def _compact_rhythm_order(values: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    if len(values) <= 1:
        return values

    arr = np.asarray(values, dtype=object)
    nroll = np.arange(len(arr), dtype=int)
    dist = np.zeros(len(arr), dtype=float)
    for index in range(len(arr)):
        rolled = np.roll(arr, index)
        dist[index] = abs(float(rolled[-1] - rolled[0]))

    result = tuple(values)
    for offset in range(1, len(arr)):
        minima = np.where(dist == dist.min())[0]
        if len(minima) != 1:
            nroll = nroll[minima]
            dist = np.zeros(len(nroll), dtype=float)
            for pos, roll in enumerate(nroll):
                rolled = np.roll(arr, int(roll))
                dist[pos] = abs(float(rolled[-(1 + offset)] - rolled[0]))
        else:
            result = tuple(np.roll(arr, int(nroll[minima[0]])))
            break
    return result


@dataclass(frozen=True, slots=True)
class RhythmSequence:
    """Immutable rhythm pattern stored as exact fractions.

    Parameters
    ----------
    durations : tuple of str, int, float, or Fraction
        Note lengths such as ``"q"`` for quarter note or exact fractions.
    reference : str, default="e"
        Reference note length used by methods such as `prime_form`.
    ordered : bool, default=False
        If ``True``, sort the durations during initialization.

    Attributes
    ----------
    durations : tuple of Fraction
        Parsed note lengths.
    reference : str
        Reference token used by selected methods.
    ordered : bool
        Whether the values were sorted during initialization.

    Examples
    --------
    >>> rhythm = RhythmSequence(("q", "e", "e"))
    >>> rhythm.durations
    (Fraction(1, 4), Fraction(1, 8), Fraction(1, 8))
    """

    durations: tuple[Fraction, ...]
    reference: str = "e"
    ordered: bool = False

    def __post_init__(self) -> None:
        parsed = parse_duration_sequence(self.durations)
        if self.ordered:
            parsed = tuple(sorted(parsed))
        object.__setattr__(self, "durations", parsed)

    def to_numpy(self) -> np.ndarray:
        """Return the rhythm values as a NumPy array of fractions."""
        return np.asarray(self.durations, dtype=object)

    def to_list(self) -> list[Fraction]:
        """Return the rhythm values as a Python list."""
        return list(self.durations)

    def label(self) -> str:
        """Return a compact fraction label for the rhythm.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).label()
        '1/4 1/8 1/8'
        """
        return duration_label(self.durations)

    def normal_order(self) -> "RhythmSequence":
        """Return the most compact cyclic ordering of the rhythm.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).normal_order().durations
        (Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
        """
        return RhythmSequence(_compact_rhythm_order(self.durations), reference=self.reference)

    def augment(self, duration: str = "e") -> "RhythmSequence":
        """Lengthen each note by a fixed amount.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).augment("e").durations
        (Fraction(3, 8), Fraction(1, 4), Fraction(1, 4))
        """
        step = parse_duration_value(duration)
        return RhythmSequence(tuple(value + step for value in self.durations), reference=self.reference)

    def diminish(self, duration: str = "e") -> "RhythmSequence":
        """Shorten each note by a fixed amount, dropping non-positive results.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).diminish("s").durations
        (Fraction(3, 16), Fraction(1, 16), Fraction(1, 16))
        """
        step = parse_duration_value(duration)
        diminished = tuple(value - step for value in self.durations if value > step)
        return RhythmSequence(diminished, reference=self.reference)

    def retrograde(self) -> "RhythmSequence":
        """Return the rhythm in reverse order.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).retrograde().durations
        (Fraction(1, 8), Fraction(1, 8), Fraction(1, 4))
        """
        return RhythmSequence(tuple(reversed(self.durations)), reference=self.reference)

    def is_non_retrogradable(self) -> bool:
        """Return whether the rhythm reads the same backward.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "q")).is_non_retrogradable()
        True
        """
        return self.durations == tuple(reversed(self.durations))

    def as_floats(self) -> np.ndarray:
        """Return the note lengths as floating-point values.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).as_floats()
        array([0.25 , 0.125, 0.125])
        """
        return _floatize(self.durations)

    def floatize(self) -> np.ndarray:
        """Alias for `as_floats`."""
        return self.as_floats()

    def reduce_to_gcd(self) -> list[float]:
        """Reduce the rhythm to counts over a shared smallest grid.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).reduce_to_gcd()
        [2.0, 1.0, 1.0, 8.0]
        """
        denominators = [value.denominator for value in self.durations]
        denominator = lcm(*denominators)
        reduced = [float(value.numerator * denominator / value.denominator) for value in self.durations]
        reduced.append(float(denominator))
        return reduced

    def prime_form(self) -> "RhythmSequence":
        """Return a normalized rhythm anchored to the reference note length.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).prime_form().durations
        (Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
        """
        normal = self.normal_order().durations
        if not normal:
            return RhythmSequence(tuple(), reference=self.reference)
        reference_value = parse_duration_value(self.reference)
        factor = reference_value * normal[0].denominator / normal[0].numerator
        return RhythmSequence(tuple(value * factor for value in normal), reference=self.reference)

    def duration_vector(self, levels: list[Fraction] | None = None) -> tuple[np.ndarray, str]:
        """Count pairwise duration differences in histogram form.

        Parameters
        ----------
        levels : list of Fraction or None, default=None
            Histogram bin edges. When omitted, built-in bins are used.

        Returns
        -------
        tuple
            Histogram counts and a compact label for the bins.

        Examples
        --------
        >>> histogram, bins = RhythmSequence(("q", "e", "e")).duration_vector()
        >>> histogram
        array([2, 0, 0, 0, 0, 0, 0, 0])
        >>> bins
        '1/8 1/4 3/8 1/2 5/8 3/4 7/8 1/1'
        """
        bins = tuple(levels) if levels is not None else DEFAULT_DURATION_BINS
        distances: list[Fraction] = []
        for i in range(len(self.durations)):
            for j in range(i + 1, len(self.durations)):
                distances.append(abs(self.durations[i] - self.durations[j]))
        hist = np.histogram(_floatize(distances), bins=_floatize(bins))[0]
        return hist, duration_label(bins[:-1])

    def inter_onset_interval_vector(self, levels: list[Fraction] | None = None) -> tuple[np.ndarray, str]:
        """Count inter-onset intervals in histogram form.

        Examples
        --------
        >>> histogram, bins = RhythmSequence(("q", "e", "e")).inter_onset_interval_vector()
        >>> histogram
        array([2, 2, 2, 0, 0, 0, 0, 0])
        >>> bins
        '1/8 1/4 3/8 1/2 5/8 3/4 7/8 1/1'
        """
        bins = tuple(levels) if levels is not None else DEFAULT_DURATION_BINS
        values = list(self.durations)
        values.extend(
            self.durations[index] + self.durations[(index + 1) % len(self.durations)]
            for index in range(len(self.durations))
        )
        hist = np.histogram(_floatize(values), bins=_floatize(bins))[0]
        return hist, duration_label(bins[:-1])

    def binary_onsets(self) -> list[int]:
        """Return a binary onset pattern on the smallest shared grid.

        Examples
        --------
        >>> RhythmSequence(("q", "e", "e")).binary_onsets()
        [1, 0, 1, 1]
        """
        binary: list[int] = []
        for count in self.reduce_to_gcd()[:-1]:
            for index in range(int(count)):
                binary.append(1 if index == 0 else 0)
        return binary
