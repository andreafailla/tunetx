"""Pitch and MIDI classes."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import music21 as m21
import numpy as np

from ..utils.distances import minimal_non_bijective_pitch_class_distance
from ._algorithms import compact_normal_order, cyclic_intervals, interval_vector


def _coerce_pitch_classes(
    values: Iterable[int | float],
    tet: int,
    *,
    unique: bool,
    ordered: bool,
) -> tuple[int, ...]:
    array = np.asarray(list(values), dtype=int) % tet
    if unique:
        array = np.unique(array) % tet
    if ordered:
        array = np.sort(array) % tet
    return tuple(int(value) for value in array.tolist())


def _parse_operator_values(name: str) -> np.ndarray:
    return np.asarray([int(match) for match in re.findall(r"-?\d+", name)], dtype=int)


def _label_for_pitch_classes(pitch_classes: Sequence[int], tet: int) -> str:
    if tet != 12:
        return "[" + ",".join(str(value) for value in pitch_classes) + "]"
    labels = []
    for pitch_class in pitch_classes:
        labels.append(m21.pitch.Pitch(midi=60 + int(pitch_class)).name)
    return "".join(labels)


def _best_rotation_difference(a: np.ndarray, b: np.ndarray, tet: int) -> np.ndarray:
    rotations = np.vstack([np.roll(b, index) for index in range(len(b))])
    diffs = a - rotations
    diffs = np.where(diffs >= tet / 2, diffs - tet, diffs)
    diffs = np.where(diffs < -tet / 2, diffs + tet, diffs)
    distances = np.sum(np.abs(diffs) * np.abs(diffs), axis=1)
    diff = rotations[int(np.argmin(distances))] - a
    diff = np.where(diff >= tet / 2, diff - tet, diff)
    diff = np.where(diff < -tet / 2, diff + tet, diff)
    return diff.astype(int)


@dataclass(frozen=True, slots=True)
class PitchClassSet:
    """Immutable pitch-class collection."""

    pcs: tuple[int, ...]
    tet: int = 12
    unique: bool = True
    ordered: bool = True

    def __post_init__(self) -> None:
        normalized = _coerce_pitch_classes(self.pcs, self.tet, unique=self.unique, ordered=self.ordered)
        object.__setattr__(self, "pcs", normalized)

    def __iter__(self) -> Iterator[int]:
        return iter(self.pcs)

    def __len__(self) -> int:
        return len(self.pcs)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.pcs, dtype=int)

    def to_list(self) -> list[int]:
        return list(self.pcs)

    def label(self) -> str:
        return _label_for_pitch_classes(self.pcs, self.tet)

    def _wrap(self, values: Iterable[int | float], *, unique: bool = False, ordered: bool = False) -> "PitchClassSet":
        return PitchClassSet(tuple(int(value) for value in values), tet=self.tet, unique=unique, ordered=ordered)

    def normal_order(self) -> "PitchClassSet":
        return self._wrap(compact_normal_order(self.to_numpy(), self.tet), unique=False, ordered=False)

    def normal_zero_order(self) -> "PitchClassSet":
        normal = self.normal_order().to_numpy()
        return self._wrap((normal - normal[0]) % self.tet, unique=False, ordered=False)

    def transpose(self, steps: int = 0) -> "PitchClassSet":
        return self._wrap((self.to_numpy() + steps) % self.tet, unique=True, ordered=True)

    def multiply(self, factor: int = 1) -> "PitchClassSet":
        return self._wrap((self.to_numpy() * factor) % self.tet, unique=True, ordered=True)

    def zero_order(self) -> "PitchClassSet":
        values = self.to_numpy()
        return self._wrap((values - values[0]) % self.tet, unique=False, ordered=False)

    def invert(self, pivot: int = 0) -> "PitchClassSet":
        return self._wrap((pivot - self.to_numpy()) % self.tet, unique=False, ordered=False)

    def invert_zero(self) -> "PitchClassSet":
        return self._wrap((-self.to_numpy()) % self.tet, unique=False, ordered=False)

    def prime_form(self) -> "PitchClassSet":
        original = self.to_numpy()
        normal = compact_normal_order(original, self.tet)
        normal_zero = (normal - normal[0]) % self.tet
        inverted = compact_normal_order((-original) % self.tet, self.tet)
        inverted_zero = (inverted - inverted[0]) % self.tet
        if normal_zero.sum() <= inverted_zero.sum():
            return self._wrap(normal_zero, unique=False, ordered=False)
        return self._wrap(inverted_zero, unique=False, ordered=False)

    def interval_vector(self) -> np.ndarray:
        return interval_vector(self.to_numpy(), self.tet)

    def lis_vector(self) -> np.ndarray:
        return cyclic_intervals(self.normal_order().to_numpy(), self.tet).astype(int)

    def intervals(self) -> np.ndarray:
        return cyclic_intervals(self.to_numpy(), self.tet).astype(int)

    def operator_name(self, other: "PitchClassSet", *, normalize: bool = True) -> str:
        left = self.normal_order().to_numpy() if normalize else self.to_numpy()
        right = other.normal_order().to_numpy() if normalize else other.to_numpy()
        diff = np.sort(np.abs(_best_rotation_difference(left, right, self.tet)))
        diff = np.trim_zeros(diff)
        if len(diff) == 0:
            return "O()"
        return "O(" + ",".join(str(int(value)) for value in diff) + ")"

    def voice_leading_operator_name(self, other: "PitchClassSet", *, normalize: bool = True) -> str:
        left = self.normal_order().to_numpy() if normalize else self.to_numpy()
        right = other.normal_order().to_numpy() if normalize else other.to_numpy()
        if len(left) != len(right):
            _, reduced = minimal_non_bijective_pitch_class_distance(left, right, tet=self.tet)
            right = np.asarray(reduced, dtype=int)
        diff = _best_rotation_difference(left, right, self.tet)
        return "R(" + ",".join(str(int(value)) for value in diff) + ")"

    def apply_voice_leading(self, name: str, *, normalize: bool = True) -> "PitchClassSet":
        op = _parse_operator_values(name)
        values = self.normal_order().to_numpy() if normalize else self.to_numpy()
        if len(op) != len(values):
            raise ValueError("voice-leading operator cardinality must match the pitch-class set")
        result = (values + op) % self.tet
        return self._wrap(result, unique=False, ordered=False)

    def neo_riemannian(self, operator: str | None) -> "PitchClassSet | None":
        if operator is None or len(self.pcs) != 3:
            return None
        values = self.to_numpy()
        x = values[1] - values[0]
        y = values[2] - values[1]
        if operator == "P":
            return self._wrap((x + y - values) % self.tet, unique=True, ordered=True)
        if operator == "L":
            return self._wrap((x - values) % self.tet, unique=True, ordered=True)
        if operator == "R":
            return self._wrap((2 * x + y - values) % self.tet, unique=True, ordered=True)
        return None


@dataclass(frozen=True, slots=True)
class PitchClassRow:
    """Immutable pitch-class row."""

    pcs: tuple[int, ...]
    tet: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "pcs", tuple(int(value) % self.tet for value in self.pcs))

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.pcs, dtype=int)

    def normal_order(self) -> "PitchClassRow":
        values = self.to_numpy()
        return PitchClassRow(tuple((values - values[0]) % self.tet), tet=self.tet)

    def intervals(self) -> np.ndarray:
        return cyclic_intervals(self.to_numpy(), self.tet).astype(int)

    def transpose(self, steps: int = 0) -> "PitchClassRow":
        return PitchClassRow(tuple((self.to_numpy() + steps) % self.tet), tet=self.tet)

    def invert(self, pivot: int = 0) -> "PitchClassRow":
        return PitchClassRow(tuple((pivot - self.to_numpy()) % self.tet), tet=self.tet)

    def retrograde(self, steps: int = 0) -> "PitchClassRow":
        return PitchClassRow(tuple(self.to_numpy()[::-1]), tet=self.tet).transpose(steps)

    def multiply(self, factor: int = 5) -> "PitchClassRow":
        return PitchClassRow(tuple((self.to_numpy() * factor) % self.tet), tet=self.tet)

    def q_transform(self) -> "PitchClassRow":
        intervals = self.intervals()
        sixes = np.where(intervals == self.tet // 2)[0]
        if len(sixes) < 2:
            raise ValueError("q_transform requires at least two tritone intervals in the row")
        rotated = np.roll(intervals, int(sixes[1] - sixes[0]))
        values = [0]
        for interval in rotated:
            values.append((values[-1] + int(interval)) % self.tet)
        return PitchClassRow(tuple(values[:-1]), tet=self.tet)

    def star(self) -> list[dict[str, tuple[int, ...] | str]]:
        return [
            {"op": "P", "row": self.pcs},
            {"op": "I", "row": self.invert().pcs},
            {"op": "R", "row": self.retrograde().pcs},
            {"op": "Q", "row": self.q_transform().pcs},
            {"op": "M", "row": self.multiply().pcs},
        ]

    def constellation(self) -> list[dict[str, tuple[int, ...] | str]]:
        return [
            {
                "family": "P",
                "P": self.pcs,
                "I": self.invert().pcs,
                "IM": self.multiply().invert().pcs,
                "M": self.multiply().pcs,
            },
            {
                "family": "R",
                "P": self.retrograde().pcs,
                "I": self.invert().retrograde().pcs,
                "IM": self.multiply().invert().retrograde().pcs,
                "M": self.multiply().retrograde().pcs,
            },
            {
                "family": "QR",
                "P": self.retrograde().q_transform().pcs,
                "I": self.invert().retrograde().q_transform().pcs,
                "IM": self.multiply().invert().retrograde().q_transform().pcs,
                "M": self.multiply().retrograde().q_transform().pcs,
            },
            {
                "family": "Q",
                "P": self.q_transform().pcs,
                "I": self.invert().q_transform().pcs,
                "IM": self.multiply().invert().q_transform().pcs,
                "M": self.multiply().q_transform().pcs,
            },
        ]


@dataclass(frozen=True, slots=True)
class MidiSet:
    """Immutable MIDI collection."""

    midi: tuple[float, ...]
    tet: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "midi", tuple(float(value) for value in _coerce_midi_values(self.midi)))

    def __iter__(self) -> Iterator[float]:
        return iter(self.midi)

    def __len__(self) -> int:
        return len(self.midi)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.midi, dtype=float)

    def to_list(self) -> list[float]:
        return list(self.midi)

    def pitch_names(self) -> tuple[str, ...]:
        return tuple(str(m21.pitch.Pitch(value)) for value in self.midi)

    def sort(self) -> "MidiSet":
        return MidiSet(tuple(sorted(self.midi)), tet=self.tet)

    def zero_order(self) -> "MidiSet":
        values = self.to_numpy()
        return MidiSet(tuple(values - values[0] + 60), tet=self.tet)

    def normal_order(self) -> "MidiSet":
        pcs = self.to_pitch_class_set(unique=False, ordered=False).normal_order().to_numpy()
        return MidiSet(tuple(pcs + 60), tet=self.tet)

    def transpose(self, steps: int | Sequence[int] = 0) -> "MidiSet":
        values = self.to_numpy()
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            delta = np.asarray(list(steps), dtype=float)
            if delta.shape != values.shape:
                raise ValueError("transpose vector must match the MIDI set shape")
            return MidiSet(tuple(values + delta), tet=self.tet)
        return MidiSet(tuple(values + float(steps)), tet=self.tet)

    def invert(self, pivot: int | Sequence[int] = 60) -> "MidiSet":
        values = self.to_numpy()
        if isinstance(pivot, Sequence) and not isinstance(pivot, (str, bytes)):
            pivots = np.asarray(list(pivot), dtype=float)
            if pivots.shape != values.shape:
                raise ValueError("pivot vector must match the MIDI set shape")
            return MidiSet(tuple((2 * pivots) - values), tet=self.tet)
        return MidiSet(tuple((2 * float(pivot)) - values), tet=self.tet)

    def apply_voice_leading(self, name: str) -> "MidiSet":
        op = _parse_operator_values(name)
        values = self.to_numpy()
        if len(op) != len(values):
            raise ValueError("voice-leading operator cardinality must match the MIDI set")
        return MidiSet(tuple(values + op), tet=self.tet)

    def voice_leading_operator_name(self, other: "MidiSet") -> str:
        diff = other.to_numpy() - self.to_numpy()
        rounded = [str(int(value)) if math.isclose(value, int(value)) else f"{value:g}" for value in diff]
        return "R(" + ",".join(rounded) + ")"

    def intervals(self) -> np.ndarray:
        return cyclic_intervals(self.to_numpy(), self.tet)

    def to_pitch_class_set(self, *, unique: bool = True, ordered: bool = True) -> PitchClassSet:
        return PitchClassSet(tuple(int(value) % self.tet for value in self.midi), tet=self.tet, unique=unique, ordered=ordered)


def _coerce_midi_values(values: Iterable[float | int | str]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, str):
            result.append(float(m21.pitch.Pitch(value).ps))
        else:
            result.append(float(value))
    return tuple(result)
