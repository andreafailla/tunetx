"""Rhythm-domain classes."""

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
    """Immutable rhythm sequence represented with exact fractions."""

    durations: tuple[Fraction, ...]
    reference: str = "e"
    ordered: bool = False

    def __post_init__(self) -> None:
        parsed = parse_duration_sequence(self.durations)
        if self.ordered:
            parsed = tuple(sorted(parsed))
        object.__setattr__(self, "durations", parsed)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.durations, dtype=object)

    def to_list(self) -> list[Fraction]:
        return list(self.durations)

    def label(self) -> str:
        return duration_label(self.durations)

    def normal_order(self) -> "RhythmSequence":
        return RhythmSequence(_compact_rhythm_order(self.durations), reference=self.reference)

    def augment(self, duration: str = "e") -> "RhythmSequence":
        step = parse_duration_value(duration)
        return RhythmSequence(tuple(value + step for value in self.durations), reference=self.reference)

    def diminish(self, duration: str = "e") -> "RhythmSequence":
        step = parse_duration_value(duration)
        diminished = tuple(value - step for value in self.durations if value > step)
        return RhythmSequence(diminished, reference=self.reference)

    def retrograde(self) -> "RhythmSequence":
        return RhythmSequence(tuple(reversed(self.durations)), reference=self.reference)

    def is_non_retrogradable(self) -> bool:
        return self.durations == tuple(reversed(self.durations))

    def as_floats(self) -> np.ndarray:
        return _floatize(self.durations)

    def floatize(self) -> np.ndarray:
        return self.as_floats()

    def reduce_to_gcd(self) -> list[float]:
        denominators = [value.denominator for value in self.durations]
        denominator = lcm(*denominators)
        reduced = [float(value.numerator * denominator / value.denominator) for value in self.durations]
        reduced.append(float(denominator))
        return reduced

    def prime_form(self) -> "RhythmSequence":
        normal = self.normal_order().durations
        if not normal:
            return RhythmSequence(tuple(), reference=self.reference)
        reference_value = parse_duration_value(self.reference)
        factor = reference_value * normal[0].denominator / normal[0].numerator
        return RhythmSequence(tuple(value * factor for value in normal), reference=self.reference)

    def duration_vector(self, levels: list[Fraction] | None = None) -> tuple[np.ndarray, str]:
        bins = tuple(levels) if levels is not None else DEFAULT_DURATION_BINS
        distances: list[Fraction] = []
        for i in range(len(self.durations)):
            for j in range(i + 1, len(self.durations)):
                distances.append(abs(self.durations[i] - self.durations[j]))
        hist = np.histogram(_floatize(distances), bins=_floatize(bins))[0]
        return hist, duration_label(bins[:-1])

    def inter_onset_interval_vector(self, levels: list[Fraction] | None = None) -> tuple[np.ndarray, str]:
        bins = tuple(levels) if levels is not None else DEFAULT_DURATION_BINS
        values = list(self.durations)
        values.extend(
            self.durations[index] + self.durations[(index + 1) % len(self.durations)]
            for index in range(len(self.durations))
        )
        hist = np.histogram(_floatize(values), bins=_floatize(bins))[0]
        return hist, duration_label(bins[:-1])

    def binary_onsets(self) -> list[int]:
        binary: list[int] = []
        for count in self.reduce_to_gcd()[:-1]:
            for index in range(int(count)):
                binary.append(1 if index == 0 else 0)
        return binary
