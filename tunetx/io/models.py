"""Small data containers shared across score I/O and sonification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """A scheduled note or chord event.

    Parameters
    ----------
    pitches : tuple of int
        MIDI notes to play together.
    start : float
        Start time of the event.
    duration : float
        Length of the event.
    velocity : int, default=80
        MIDI velocity, where larger numbers are louder.

    Attributes
    ----------
    pitches : tuple of int
        MIDI notes played by the event.
    start : float
        Event start time.
    duration : float
        Event length.
    velocity : int
        MIDI velocity.

    Examples
    --------
    >>> event = NoteEvent(pitches=(60, 64, 67), start=0.0, duration=1.0, velocity=90)
    >>> event.pitches
    (60, 64, 67)
    >>> event.velocity
    90
    """

    pitches: tuple[int, ...]
    start: float
    duration: float
    velocity: int = 80


@dataclass(frozen=True, slots=True)
class MidiNote:
    """A single parsed note from a score.

    Attributes
    ----------
    pitch : float
        MIDI pitch value from the parsed score.
    start : float
        Start time in score units.
    duration : float
        Note length in score units.
    velocity : int
        MIDI velocity if available in the source score.

    Examples
    --------
    >>> note = MidiNote(pitch=60.0, start=0.0, duration=1.0, velocity=64)
    >>> (note.pitch, note.duration)
    (60.0, 1.0)
    """

    pitch: float
    start: float
    duration: float
    velocity: int


@dataclass(frozen=True, slots=True)
class MidiChordSlice:
    """A group of notes that sound at the same time in a parsed score.

    Attributes
    ----------
    pitches : tuple of float
        Raw MIDI pitches in the slice.
    pitch_classes : tuple of int
        Notes without octave, reduced to numbers from 0 to 11.
    start : float
        Start time in score units.
    duration : float
        Slice length in score units.
    label : str
        Human-readable note-name label such as ``"CEG"``.

    Examples
    --------
    >>> chord = MidiChordSlice(pitches=(60.0, 64.0, 67.0), pitch_classes=(0, 4, 7), start=0.0, duration=1.0, label="CEG")
    >>> chord.label
    'CEG'
    """

    pitches: tuple[float, ...]
    pitch_classes: tuple[int, ...]
    start: float
    duration: float
    label: str


@dataclass(frozen=True, slots=True)
class MidiScore:
    """Normalized score representation used throughout `tunetx`.

    Attributes
    ----------
    notes : tuple of MidiNote
        Flattened note list from the source score.
    chords : tuple of MidiChordSlice
        Simultaneous note groups extracted from the score.
    tempo : float or None
        First detected tempo marking, if present.
    meter : str or None
        First detected time signature, if present.
    source_path : str or None, default=None
        Original file path when the score came from disk.

    Notes
    -----
    Reusing one `MidiScore` object is the fast path when you want to run
    several analyses on the same file.

    Examples
    --------
    >>> score = MidiScore(notes=tuple(), chords=tuple(), tempo=120.0, meter="4/4")
    >>> (score.tempo, score.meter)
    (120.0, '4/4')
    """

    notes: tuple[MidiNote, ...]
    chords: tuple[MidiChordSlice, ...]
    tempo: float | None
    meter: str | None
    source_path: str | None = None
