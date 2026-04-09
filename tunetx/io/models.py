"""Dataclasses shared across score I/O and sonification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """A scheduled note or chord event."""

    pitches: tuple[int, ...]
    start: float
    duration: float
    velocity: int = 80


@dataclass(frozen=True, slots=True)
class MidiNote:
    """A single parsed note event."""

    pitch: float
    start: float
    duration: float
    velocity: int


@dataclass(frozen=True, slots=True)
class MidiChordSlice:
    """A simultaneous pitch slice extracted from a score."""

    pitches: tuple[float, ...]
    pitch_classes: tuple[int, ...]
    start: float
    duration: float
    label: str


@dataclass(frozen=True, slots=True)
class MidiScore:
    """Normalized score representation used by tunetx."""

    notes: tuple[MidiNote, ...]
    chords: tuple[MidiChordSlice, ...]
    tempo: float | None
    meter: str | None
    source_path: str | None = None
