"""Enumeration and descriptor helpers for network workflows."""

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
    """Precomputed pitch-class descriptors."""

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
    """Precomputed rhythm descriptors."""

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
    """Lightweight timbral descriptor vector for WAV audio."""

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
    """Enumerate a pitch-class space and normalize each item into the requested form."""

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
    """Enumerate rhythm cells from a duration palette."""

    sequences: list[RhythmSequence] = []
    for combination in itertools.combinations(durations, length):
        sequences.append(RhythmSequence(tuple(combination), reference=reference).normal_order())
    return deduplicate(sequences)


def describe_pitch_class_set(item: PitchClassSet | Iterable[int], tet: int = 12) -> PitchClassDescription:
    """Describe a pitch-class set with common symbolic descriptors."""

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
    """Describe a rhythm sequence with common rhythmic descriptors."""

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
    """Extract lightweight timbral descriptors from a WAV file."""

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
