"""Enumerate musical material and summarize it with descriptor vectors."""

from __future__ import annotations

import itertools
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..classes import PitchClassSet, RhythmSequence
from ..classes._algorithms import compact_normal_order, cyclic_intervals, interval_vector
from ..utils.collections import deduplicate


@dataclass(frozen=True, slots=True)
class PitchClassDescription:
    """Precomputed descriptors for a note group.

    Attributes
    ----------
    label : str
        Human-readable note-name label.
    pitch_class_set : PitchClassSet
        Original note group.
    prime_form : tuple of int
        Canonical zero-based form.
    interval_vector : tuple of int
        Interval-content summary.
    lis_vector : tuple of int
        Adjacent step sizes in normal order.
    """

    label: str
    pitch_class_set: PitchClassSet
    prime_form: tuple[int, ...]
    interval_vector: tuple[int, ...]
    lis_vector: tuple[int, ...]

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.interval_vector, dtype=float)


@dataclass(frozen=True, slots=True)
class RhythmDescription:
    """Precomputed descriptors for a rhythm pattern.

    Attributes
    ----------
    label : str
        Human-readable fraction label.
    rhythm_sequence : RhythmSequence
        Original rhythm pattern.
    duration_vector : tuple of int
        Histogram of pairwise duration differences.
    inter_onset_vector : tuple of int
        Histogram of inter-onset intervals.
    binary_onsets : tuple of int
        Binary onset pattern on the smallest shared grid.
    """

    label: str
    rhythm_sequence: RhythmSequence
    duration_vector: tuple[int, ...]
    inter_onset_vector: tuple[int, ...]
    binary_onsets: tuple[int, ...]

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.duration_vector, dtype=float)


@dataclass(frozen=True, slots=True)
class TimbreDescription:
    """Lightweight timbre summary for a WAV file.

    Attributes
    ----------
    label : str
        File stem.
    path : str
        Path to the WAV file.
    rms : float
        Root-mean-square energy.
    spectral_centroid : float
        Brightness-related spectral center.
    spectral_bandwidth : float
        Spread of the spectrum.
    zero_crossing_rate : float
        Rate of waveform sign changes.
    duration : float
        Audio duration in seconds.
    """

    label: str
    path: str
    rms: float
    spectral_centroid: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    duration: float

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.rms,
                self.spectral_centroid,
                self.spectral_bandwidth,
                self.zero_crossing_rate,
                self.duration,
            ],
            dtype=float,
        )


def enumerate_pitch_class_sets(
    cardinality: int,
    tet: int = 12,
    form: str = "prime",
    row: Iterable[int] | None = None,
) -> list[PitchClassSet]:
    """Enumerate note groups of a given size.

    Parameters
    ----------
    cardinality : int
        Number of notes in each group.
    tet : int, default=12
        Size of the equal-tempered system.
    form : {"prime", "normal", "normal_zero", "normal-zero", "raw"}, default="prime"
        Normalization to apply to each result.
    row : iterable of int or None, default=None
        Source values to choose from. When omitted, ``range(tet)`` is used.

    Returns
    -------
    list of PitchClassSet
        Unique normalized note groups.

    Raises
    ------
    ValueError
        If ``form`` is unsupported.

    Examples
    --------
    >>> items = enumerate_pitch_class_sets(3)
    >>> items[0].pcs
    (0, 1, 2)
    >>> len(items) > 0
    True
    """

    source = tuple(row) if row is not None else tuple(range(tet))
    if form == "prime":
        def normalize(values: tuple[int, ...]) -> tuple[int, ...]:
            array = np.asarray(values, dtype=int)
            normal = compact_normal_order(array, tet)
            normal_zero = tuple(int(value) for value in ((normal - normal[0]) % tet).tolist())
            inverted = compact_normal_order((-array) % tet, tet)
            inverted_zero = tuple(int(value) for value in ((inverted - inverted[0]) % tet).tolist())
            return normal_zero if sum(normal_zero) <= sum(inverted_zero) else inverted_zero
    elif form == "normal":
        def normalize(values: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(int(value) for value in compact_normal_order(np.asarray(values, dtype=int), tet).tolist())
    elif form in {"normal_zero", "normal-zero"}:
        def normalize(values: tuple[int, ...]) -> tuple[int, ...]:
            normal = compact_normal_order(np.asarray(values, dtype=int), tet)
            return tuple(int(value) for value in ((normal - normal[0]) % tet).tolist())
    elif form == "raw":
        def normalize(values: tuple[int, ...]) -> tuple[int, ...]:
            return values
    else:
        raise ValueError(f"unsupported pitch-class enumeration form: {form}")

    results: list[PitchClassSet] = []
    seen: set[tuple[int, ...]] = set()
    for combination in itertools.combinations(source, cardinality):
        normalized_values = normalize(tuple(int(value) for value in combination))
        if normalized_values not in seen:
            seen.add(normalized_values)
            results.append(PitchClassSet(normalized_values, tet=tet, unique=False, ordered=False))
    return results


def enumerate_rhythm_sequences(
    length: int,
    durations: Iterable[str | int | float],
    reference: str = "e",
) -> list[RhythmSequence]:
    """Enumerate rhythm patterns from a duration palette.

    Parameters
    ----------
    length : int
        Number of note lengths per pattern.
    durations : iterable of str, int, or float
        Available note-length tokens or values.
    reference : str, default="e"
        Reference token passed to each `RhythmSequence`.

    Returns
    -------
    list of RhythmSequence
        Unique normalized rhythm patterns.

    Examples
    --------
    >>> items = enumerate_rhythm_sequences(2, ("q", "e", "s"))
    >>> [item.label() for item in items[:3]]
    ['1/4 1/8', '1/4 1/16', '1/8 1/16']
    """

    sequences: list[RhythmSequence] = []
    for combination in itertools.combinations(durations, length):
        sequences.append(RhythmSequence(tuple(combination), reference=reference).normal_order())
    return deduplicate(sequences)


def describe_pitch_class_set(item: PitchClassSet | Iterable[int], tet: int = 12) -> PitchClassDescription:
    """Describe a note group with common symbolic descriptors.

    Parameters
    ----------
    item : PitchClassSet or iterable of int
        Note group to describe.
    tet : int, default=12
        Size of the equal-tempered system.

    Returns
    -------
    PitchClassDescription
        Precomputed descriptors for the note group.

    Examples
    --------
    >>> description = describe_pitch_class_set((0, 4, 7))
    >>> description.label
    'CEG'
    >>> description.interval_vector
    (0, 0, 1, 1, 1, 0)
    """

    pcs = item if isinstance(item, PitchClassSet) else PitchClassSet(tuple(item), tet=tet)
    values = pcs.to_numpy()
    normal = compact_normal_order(values, tet)
    normal_zero = tuple(int(value) for value in ((normal - normal[0]) % tet).tolist())
    inverted = compact_normal_order((-values) % tet, tet)
    inverted_zero = tuple(int(value) for value in ((inverted - inverted[0]) % tet).tolist())
    prime_form = normal_zero if sum(normal_zero) <= sum(inverted_zero) else inverted_zero
    return PitchClassDescription(
        label=pcs.label(),
        pitch_class_set=pcs,
        prime_form=prime_form,
        interval_vector=tuple(int(value) for value in interval_vector(values, tet).tolist()),
        lis_vector=tuple(int(value) for value in cyclic_intervals(normal, tet).astype(int).tolist()),
    )


def describe_rhythm_sequence(item: RhythmSequence | Iterable[str | int | float], reference: str = "e") -> RhythmDescription:
    """Describe a rhythm pattern with common summary vectors.

    Parameters
    ----------
    item : RhythmSequence or iterable of str, int, or float
        Rhythm pattern to describe.
    reference : str, default="e"
        Reference token used when coercing plain iterables into a
        `RhythmSequence`.

    Returns
    -------
    RhythmDescription
        Precomputed descriptors for the rhythm.

    Examples
    --------
    >>> description = describe_rhythm_sequence(("q", "e", "e"))
    >>> description.binary_onsets
    (1, 0, 1, 1)
    >>> description.duration_vector
    (2, 0, 0, 0, 0, 0, 0, 0)
    """

    rhythm = item if isinstance(item, RhythmSequence) else RhythmSequence(tuple(item), reference=reference)
    duration_vector, _ = rhythm.duration_vector()
    inter_onset_vector, _ = rhythm.inter_onset_interval_vector()
    return RhythmDescription(
        label=rhythm.label(),
        rhythm_sequence=rhythm,
        duration_vector=tuple(int(value) for value in duration_vector.tolist()),
        inter_onset_vector=tuple(int(value) for value in inter_onset_vector.tolist()),
        binary_onsets=tuple(int(value) for value in rhythm.binary_onsets()),
    )


def describe_timbre(item: TimbreDescription | str | Path) -> TimbreDescription:
    """Extract lightweight timbre descriptors from a WAV file.

    Parameters
    ----------
    item : TimbreDescription, str, or Path
        Existing descriptor or path to a WAV file.

    Returns
    -------
    TimbreDescription
        Summary descriptor for the WAV file.

    Notes
    -----
    This helper is intended for simple comparisons between files, not for
    studio-grade timbre analysis.
    """

    if isinstance(item, TimbreDescription):
        return item

    path = Path(item)
    samples, sample_rate = _read_wav(path)
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    magnitude_sum = float(spectrum.sum()) or 1.0
    centroid = float(np.sum(freqs * spectrum) / magnitude_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / magnitude_sum))
    zero_crossings = int(np.count_nonzero(np.diff(np.signbit(samples))))
    return TimbreDescription(
        label=path.stem,
        path=str(path),
        rms=float(np.sqrt(np.mean(samples**2))),
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        zero_crossing_rate=float(zero_crossings / max(len(samples) - 1, 1)),
        duration=float(len(samples) / sample_rate),
    )


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 1:
        dtype = np.uint8
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    if not np.any(samples):
        samples = np.zeros(max(len(samples), 1), dtype=np.float32)
    return samples.astype(np.float32), sample_rate
