"""Classes for working with notes, note groups, and MIDI pitches."""

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
    """Immutable group of notes reduced to note names without octave.

    In music theory this is called a pitch-class set. In plain terms, it is a
    group of note identities such as C, E, and G, ignoring which octave those
    notes came from.

    Parameters
    ----------
    pcs : tuple of int
        Input note classes as integers.
    tet : int, default=12
        Size of the equal-tempered system.
    unique : bool, default=True
        If ``True``, repeated note classes are removed during normalization.
    ordered : bool, default=True
        If ``True``, values are sorted during normalization.

    Attributes
    ----------
    pcs : tuple of int
        Normalized note classes.
    tet : int
        Size of the equal-tempered system.
    unique : bool
        Whether duplicate note classes are removed.
    ordered : bool
        Whether note classes are sorted.

    Examples
    --------
    >>> pcs = PitchClassSet((7, 0, 4, 7))
    >>> pcs.pcs
    (0, 4, 7)
    >>> pcs.label()
    'CEG'
    """

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
        """Return the note group as a NumPy array.

        Returns
        -------
        numpy.ndarray
            One-dimensional integer array of note classes.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).to_numpy().tolist()
        [0, 4, 7]
        """
        return np.asarray(self.pcs, dtype=int)

    def to_list(self) -> list[int]:
        """Return the note group as a Python list.

        Returns
        -------
        list of int
            Note classes as a list.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).to_list()
        [0, 4, 7]
        """
        return list(self.pcs)

    def label(self) -> str:
        """Return a note-name label for the group.

        Returns
        -------
        str
            Concatenated note names such as ``"CEG"`` in 12-tone equal
            temperament.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).label()
        'CEG'
        """
        return _label_for_pitch_classes(self.pcs, self.tet)

    def _wrap(self, values: Iterable[int | float], *, unique: bool = False, ordered: bool = False) -> "PitchClassSet":
        return PitchClassSet(tuple(int(value) for value in values), tet=self.tet, unique=unique, ordered=ordered)

    def normal_order(self) -> "PitchClassSet":
        """Return the most compact ordering of the note group.

        Returns
        -------
        PitchClassSet
            Compact ordering of the same notes.

        Examples
        --------
        >>> PitchClassSet((7, 0, 4)).normal_order().pcs
        (0, 4, 7)
        """
        return self._wrap(compact_normal_order(self.to_numpy(), self.tet), unique=False, ordered=False)

    def normal_zero_order(self) -> "PitchClassSet":
        """Return normal order transposed so the first note becomes zero.

        Returns
        -------
        PitchClassSet
            Zero-based normal order.

        Examples
        --------
        >>> PitchClassSet((7, 0, 4)).normal_zero_order().pcs
        (0, 4, 7)
        """
        normal = self.normal_order().to_numpy()
        return self._wrap((normal - normal[0]) % self.tet, unique=False, ordered=False)

    def transpose(self, steps: int = 0) -> "PitchClassSet":
        """Transpose every note class by the same amount.

        Parameters
        ----------
        steps : int, default=0
            Number of semitone steps in 12-tone equal temperament.

        Returns
        -------
        PitchClassSet
            Transposed note group.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).transpose(3).pcs
        (3, 7, 10)
        """
        return self._wrap((self.to_numpy() + steps) % self.tet, unique=True, ordered=True)

    def multiply(self, factor: int = 1) -> "PitchClassSet":
        """Multiply each note class modulo the tuning system.

        Parameters
        ----------
        factor : int, default=1
            Multiplication factor.

        Returns
        -------
        PitchClassSet
            Transformed note group.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).multiply(5).pcs
        (0, 8, 11)
        """
        return self._wrap((self.to_numpy() * factor) % self.tet, unique=True, ordered=True)

    def zero_order(self) -> "PitchClassSet":
        """Return the current ordering shifted so the first note becomes zero.

        Returns
        -------
        PitchClassSet
            Zero-based version of the current ordering.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).zero_order().pcs
        (0, 4, 7)
        """
        values = self.to_numpy()
        return self._wrap((values - values[0]) % self.tet, unique=False, ordered=False)

    def invert(self, pivot: int = 0) -> "PitchClassSet":
        """Invert the note group around a pivot value.

        Parameters
        ----------
        pivot : int, default=0
            Pivot used for the inversion.

        Returns
        -------
        PitchClassSet
            Inverted note group.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).invert(2).pcs
        (2, 10, 7)
        """
        return self._wrap((pivot - self.to_numpy()) % self.tet, unique=False, ordered=False)

    def invert_zero(self) -> "PitchClassSet":
        """Invert the note group around zero.

        Returns
        -------
        PitchClassSet
            Inverted note group.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).invert_zero().pcs
        (0, 8, 5)
        """
        return self._wrap((-self.to_numpy()) % self.tet, unique=False, ordered=False)

    def prime_form(self) -> "PitchClassSet":
        """Return the prime form of the note group.

        Returns
        -------
        PitchClassSet
            Canonical zero-based form used for comparisons.

        Notes
        -----
        Prime form chooses between the original and inverted compact forms,
        keeping the version with the smaller summed values.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).prime_form().pcs
        (0, 3, 7)
        """
        original = self.to_numpy()
        normal = compact_normal_order(original, self.tet)
        normal_zero = (normal - normal[0]) % self.tet
        inverted = compact_normal_order((-original) % self.tet, self.tet)
        inverted_zero = (inverted - inverted[0]) % self.tet
        if normal_zero.sum() <= inverted_zero.sum():
            return self._wrap(normal_zero, unique=False, ordered=False)
        return self._wrap(inverted_zero, unique=False, ordered=False)

    def interval_vector(self) -> np.ndarray:
        """Count interval content inside the note group.

        Returns
        -------
        numpy.ndarray
            Interval vector.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).interval_vector()
        array([0, 0, 1, 1, 1, 0])
        """
        return interval_vector(self.to_numpy(), self.tet)

    def lis_vector(self) -> np.ndarray:
        """Return the adjacent interval steps from normal order.

        Returns
        -------
        numpy.ndarray
            Interval-step vector.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).lis_vector()
        array([4, 3, 5])
        """
        return cyclic_intervals(self.normal_order().to_numpy(), self.tet).astype(int)

    def intervals(self) -> np.ndarray:
        """Return adjacent interval steps in the current ordering.

        Returns
        -------
        numpy.ndarray
            Interval-step vector for the current order.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).intervals()
        array([4, 3, 5])
        """
        return cyclic_intervals(self.to_numpy(), self.tet).astype(int)

    def operator_name(self, other: "PitchClassSet", *, normalize: bool = True) -> str:
        """Return a compact operator label between two note groups.

        Parameters
        ----------
        other : PitchClassSet
            Target note group.
        normalize : bool, default=True
            If ``True``, compare normal orders instead of the raw stored order.

        Returns
        -------
        str
            Operator label such as ``"O(1)"``.

        Examples
        --------
        >>> major = PitchClassSet((0, 4, 7))
        >>> minor = PitchClassSet((0, 3, 7))
        >>> major.operator_name(minor)
        'O(1)'
        """
        left = self.normal_order().to_numpy() if normalize else self.to_numpy()
        right = other.normal_order().to_numpy() if normalize else other.to_numpy()
        diff = np.sort(np.abs(_best_rotation_difference(left, right, self.tet)))
        diff = np.trim_zeros(diff)
        if len(diff) == 0:
            return "O()"
        return "O(" + ",".join(str(int(value)) for value in diff) + ")"

    def voice_leading_operator_name(self, other: "PitchClassSet", *, normalize: bool = True) -> str:
        """Return the note-by-note motion needed to reach another group.

        Parameters
        ----------
        other : PitchClassSet
            Target note group.
        normalize : bool, default=True
            If ``True``, compare normal orders instead of the raw stored order.

        Returns
        -------
        str
            Voice-leading label such as ``"R(0,-1,0)"``.

        Examples
        --------
        >>> major = PitchClassSet((0, 4, 7))
        >>> minor = PitchClassSet((0, 3, 7))
        >>> major.voice_leading_operator_name(minor)
        'R(0,-1,0)'
        """
        left = self.normal_order().to_numpy() if normalize else self.to_numpy()
        right = other.normal_order().to_numpy() if normalize else other.to_numpy()
        if len(left) != len(right):
            _, reduced = minimal_non_bijective_pitch_class_distance(left, right, tet=self.tet)
            right = np.asarray(reduced, dtype=int)
        diff = _best_rotation_difference(left, right, self.tet)
        return "R(" + ",".join(str(int(value)) for value in diff) + ")"

    def apply_voice_leading(self, name: str, *, normalize: bool = True) -> "PitchClassSet":
        """Apply a voice-leading operator string to the note group.

        Parameters
        ----------
        name : str
            Operator string such as ``"R(0,-1,0)"``.
        normalize : bool, default=True
            If ``True``, apply the operator to the normal order.

        Returns
        -------
        PitchClassSet
            Result of the transformation.

        Raises
        ------
        ValueError
            If the operator length does not match the note-group size.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).apply_voice_leading("R(0,-1,0)").pcs
        (0, 3, 7)
        """
        op = _parse_operator_values(name)
        values = self.normal_order().to_numpy() if normalize else self.to_numpy()
        if len(op) != len(values):
            raise ValueError("voice-leading operator cardinality must match the pitch-class set")
        result = (values + op) % self.tet
        return self._wrap(result, unique=False, ordered=False)

    def neo_riemannian(self, operator: str | None) -> "PitchClassSet | None":
        """Apply a simple neo-Riemannian triad transform.

        Parameters
        ----------
        operator : {"P", "L", "R"} or None
            Transform to apply.

        Returns
        -------
        PitchClassSet or None
            Transformed triad, or ``None`` when the input is not a triad or the
            operator is unsupported.

        Examples
        --------
        >>> PitchClassSet((0, 4, 7)).neo_riemannian("P").pcs
        (0, 3, 7)
        >>> PitchClassSet((0, 4, 7)).neo_riemannian(None) is None
        True
        """
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
    """Immutable ordered row of note classes.

    Unlike `PitchClassSet`, this class keeps order and duplicates exactly as
    given, which is useful for row-based transformations.

    Parameters
    ----------
    pcs : tuple of int
        Ordered note classes.
    tet : int, default=12
        Size of the equal-tempered system.

    Examples
    --------
    >>> row = PitchClassRow((0, 11, 7, 4, 2, 9))
    >>> row.pcs
    (0, 11, 7, 4, 2, 9)
    """

    pcs: tuple[int, ...]
    tet: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "pcs", tuple(int(value) % self.tet for value in self.pcs))

    def to_numpy(self) -> np.ndarray:
        """Return the row as a NumPy array.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7)).to_numpy().tolist()
        [0, 11, 7]
        """
        return np.asarray(self.pcs, dtype=int)

    def normal_order(self) -> "PitchClassRow":
        """Shift the row so the first item becomes zero.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).normal_order().pcs
        (0, 11, 7, 4, 2, 9)
        """
        values = self.to_numpy()
        return PitchClassRow(tuple((values - values[0]) % self.tet), tet=self.tet)

    def intervals(self) -> np.ndarray:
        """Return adjacent interval steps around the row.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).intervals()
        array([11,  8,  9, 10,  7,  3])
        """
        return cyclic_intervals(self.to_numpy(), self.tet).astype(int)

    def transpose(self, steps: int = 0) -> "PitchClassRow":
        """Transpose every row item by the same amount.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).transpose(2).pcs
        (2, 1, 9, 6, 4, 11)
        """
        return PitchClassRow(tuple((self.to_numpy() + steps) % self.tet), tet=self.tet)

    def invert(self, pivot: int = 0) -> "PitchClassRow":
        """Invert the row around a pivot.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).invert(3).pcs
        (3, 4, 8, 11, 1, 6)
        """
        return PitchClassRow(tuple((pivot - self.to_numpy()) % self.tet), tet=self.tet)

    def retrograde(self, steps: int = 0) -> "PitchClassRow":
        """Reverse the row, then optionally transpose it.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).retrograde(1).pcs
        (10, 3, 5, 8, 0, 1)
        """
        return PitchClassRow(tuple(self.to_numpy()[::-1]), tet=self.tet).transpose(steps)

    def multiply(self, factor: int = 5) -> "PitchClassRow":
        """Multiply each row item modulo the tuning system.

        Examples
        --------
        >>> PitchClassRow((0, 11, 7, 4, 2, 9)).multiply().pcs
        (0, 7, 11, 8, 10, 9)
        """
        return PitchClassRow(tuple((self.to_numpy() * factor) % self.tet), tet=self.tet)

    def q_transform(self) -> "PitchClassRow":
        """Return the Q transform of the row.

        Raises
        ------
        ValueError
            If the row does not contain at least two tritone intervals.

        Examples
        --------
        >>> row = PitchClassRow((0, 1, 2, 3, 4, 5, 11, 7, 8, 9, 10, 6))
        >>> row.q_transform().pcs
        (0, 8, 9, 10, 11, 7, 1, 2, 3, 4, 5, 6)
        """
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
        """Return a small family of related row transforms.

        Examples
        --------
        >>> row = PitchClassRow((0, 1, 2, 3, 4, 5, 11, 7, 8, 9, 10, 6))
        >>> [item["op"] for item in row.star()]
        ['P', 'I', 'R', 'Q', 'M']
        """
        return [
            {"op": "P", "row": self.pcs},
            {"op": "I", "row": self.invert().pcs},
            {"op": "R", "row": self.retrograde().pcs},
            {"op": "Q", "row": self.q_transform().pcs},
            {"op": "M", "row": self.multiply().pcs},
        ]

    def constellation(self) -> list[dict[str, tuple[int, ...] | str]]:
        """Return grouped transform families for the row.

        Examples
        --------
        >>> row = PitchClassRow((0, 1, 2, 3, 4, 5, 11, 7, 8, 9, 10, 6))
        >>> [item["family"] for item in row.constellation()]
        ['P', 'R', 'QR', 'Q']
        """
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
    """Immutable collection of concrete MIDI notes.

    This class keeps octave information, unlike `PitchClassSet`.

    Parameters
    ----------
    midi : tuple of float, int, or str
        MIDI notes or pitch names such as ``"C4"``.
    tet : int, default=12
        Size of the equal-tempered system for conversions back to note classes.

    Examples
    --------
    >>> midi = MidiSet(("C4", "E4", "G4"))
    >>> midi.midi
    (60.0, 64.0, 67.0)
    """

    midi: tuple[float, ...]
    tet: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "midi", tuple(float(value) for value in _coerce_midi_values(self.midi)))

    def __iter__(self) -> Iterator[float]:
        return iter(self.midi)

    def __len__(self) -> int:
        return len(self.midi)

    def to_numpy(self) -> np.ndarray:
        """Return the MIDI notes as a NumPy array."""
        return np.asarray(self.midi, dtype=float)

    def to_list(self) -> list[float]:
        """Return the MIDI notes as a Python list."""
        return list(self.midi)

    def pitch_names(self) -> tuple[str, ...]:
        """Return human-readable pitch names.

        Examples
        --------
        >>> MidiSet(("C4", "E4", "G4")).pitch_names()
        ('C4', 'E4', 'G4')
        """
        return tuple(str(m21.pitch.Pitch(value)) for value in self.midi)

    def sort(self) -> "MidiSet":
        """Return the notes sorted from low to high."""
        return MidiSet(tuple(sorted(self.midi)), tet=self.tet)

    def zero_order(self) -> "MidiSet":
        """Shift the notes so the first item becomes middle C."""
        values = self.to_numpy()
        return MidiSet(tuple(values - values[0] + 60), tet=self.tet)

    def normal_order(self) -> "MidiSet":
        """Return the compact pitch-class ordering anchored near middle C.

        Examples
        --------
        >>> MidiSet(("C4", "E4", "G4")).normal_order().midi
        (60.0, 64.0, 67.0)
        """
        pcs = self.to_pitch_class_set(unique=False, ordered=False).normal_order().to_numpy()
        return MidiSet(tuple(pcs + 60), tet=self.tet)

    def transpose(self, steps: int | Sequence[int] = 0) -> "MidiSet":
        """Transpose all MIDI notes by a scalar or per-note vector.

        Examples
        --------
        >>> MidiSet(("C4", "E4", "G4")).transpose(2).midi
        (62.0, 66.0, 69.0)
        """
        values = self.to_numpy()
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            delta = np.asarray(list(steps), dtype=float)
            if delta.shape != values.shape:
                raise ValueError("transpose vector must match the MIDI set shape")
            return MidiSet(tuple(values + delta), tet=self.tet)
        return MidiSet(tuple(values + float(steps)), tet=self.tet)

    def invert(self, pivot: int | Sequence[int] = 60) -> "MidiSet":
        """Invert MIDI notes around a pivot note or pivot vector."""
        values = self.to_numpy()
        if isinstance(pivot, Sequence) and not isinstance(pivot, (str, bytes)):
            pivots = np.asarray(list(pivot), dtype=float)
            if pivots.shape != values.shape:
                raise ValueError("pivot vector must match the MIDI set shape")
            return MidiSet(tuple((2 * pivots) - values), tet=self.tet)
        return MidiSet(tuple((2 * float(pivot)) - values), tet=self.tet)

    def apply_voice_leading(self, name: str) -> "MidiSet":
        """Apply a voice-leading operator string to the MIDI notes.

        Examples
        --------
        >>> MidiSet(("C4", "E4", "G4")).apply_voice_leading("R(0,-1,0)").midi
        (60.0, 63.0, 67.0)
        """
        op = _parse_operator_values(name)
        values = self.to_numpy()
        if len(op) != len(values):
            raise ValueError("voice-leading operator cardinality must match the MIDI set")
        return MidiSet(tuple(values + op), tet=self.tet)

    def voice_leading_operator_name(self, other: "MidiSet") -> str:
        """Return the note-by-note motion between two MIDI note groups.

        Examples
        --------
        >>> MidiSet((60, 64, 67)).voice_leading_operator_name(MidiSet((60, 63, 67)))
        'R(0,-1,0)'
        """
        diff = other.to_numpy() - self.to_numpy()
        rounded = [str(int(value)) if math.isclose(value, int(value)) else f"{value:g}" for value in diff]
        return "R(" + ",".join(rounded) + ")"

    def intervals(self) -> np.ndarray:
        """Return adjacent interval steps for the current ordering.

        Examples
        --------
        >>> MidiSet(("C4", "E4", "G4")).intervals()
        array([4., 3., 5.])
        """
        return cyclic_intervals(self.to_numpy(), self.tet)

    def to_pitch_class_set(self, *, unique: bool = True, ordered: bool = True) -> PitchClassSet:
        """Drop octave information and return a `PitchClassSet`.

        Examples
        --------
        >>> MidiSet((60, 64, 67)).to_pitch_class_set().pcs
        (0, 4, 7)
        """
        return PitchClassSet(tuple(int(value) % self.tet for value in self.midi), tet=self.tet, unique=unique, ordered=ordered)


def _coerce_midi_values(values: Iterable[float | int | str]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, str):
            result.append(float(m21.pitch.Pitch(value).ps))
        else:
            result.append(float(value))
    return tuple(result)
