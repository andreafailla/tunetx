"""Score parsing and MIDI writing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import music21 as m21

from ..classes import PitchClassSet
from .models import MidiChordSlice, MidiNote, MidiScore, NoteEvent


def read_midi(path: str) -> MidiScore:
    """Parse a MIDI file into a normalized score representation."""

    return read_score(path)


def read_score(source: str | m21.stream.base.Stream) -> MidiScore:
    """Parse a MIDI or MusicXML score into a normalized representation."""

    stream = source if isinstance(source, m21.stream.base.Stream) else m21.converter.parse(source)
    flat = stream.flatten()

    tempo = None
    tempo_marks = list(flat.getElementsByClass(m21.tempo.MetronomeMark))
    if tempo_marks and tempo_marks[0].number is not None:
        tempo = float(tempo_marks[0].number)

    meter = None
    meters = list(flat.getElementsByClass(m21.meter.TimeSignature))
    if meters:
        meter = meters[0].ratioString

    notes: list[MidiNote] = []
    for element in flat.notes:
        if isinstance(element, m21.note.Note):
            notes.append(
                MidiNote(
                    pitch=float(element.pitch.ps),
                    start=float(element.offset),
                    duration=float(element.duration.quarterLength),
                    velocity=int(element.volume.velocity or 64),
                )
            )
        elif isinstance(element, m21.chord.Chord):
            velocity = int(element.volume.velocity or 64)
            for pitch in element.pitches:
                notes.append(
                    MidiNote(
                        pitch=float(pitch.ps),
                        start=float(element.offset),
                        duration=float(element.duration.quarterLength),
                        velocity=velocity,
                    )
                )

    chords = extract_chord_slices(stream)
    return MidiScore(
        notes=tuple(notes),
        chords=tuple(chords),
        tempo=tempo,
        meter=meter,
        source_path=None if isinstance(source, m21.stream.base.Stream) else str(source),
    )


def extract_chord_slices(source: str | MidiScore | m21.stream.base.Stream) -> tuple[MidiChordSlice, ...]:
    """Extract simultaneous pitch slices from a score-like input."""

    if isinstance(source, MidiScore):
        return source.chords
    stream = source if isinstance(source, m21.stream.base.Stream) else m21.converter.parse(source)
    chordified = stream.chordify().flatten()
    chords: list[MidiChordSlice] = []
    for element in chordified.notes:
        if isinstance(element, m21.note.Note):
            pitches = (float(element.pitch.ps),)
        elif isinstance(element, m21.chord.Chord):
            pitches = tuple(float(pitch.ps) for pitch in element.pitches)
        else:
            continue
        pitch_classes = PitchClassSet(tuple(int(pitch) % 12 for pitch in pitches)).pcs
        label = PitchClassSet(pitch_classes).label()
        chords.append(
            MidiChordSlice(
                pitches=pitches,
                pitch_classes=pitch_classes,
                start=float(element.offset),
                duration=float(element.duration.quarterLength),
                label=label,
            )
        )
    return tuple(chords)


def pitch_class_sequence_from_score(source: str | MidiScore | m21.stream.base.Stream) -> tuple[PitchClassSet, ...]:
    """Return a pitch-class sequence derived from score chord slices."""

    return tuple(PitchClassSet(slice_item.pitch_classes) for slice_item in extract_chord_slices(source))


def write_midi(
    events_or_pitches: Iterable[NoteEvent | int | float | Iterable[int | float]],
    path: str,
    *,
    tempo: float = 120.0,
    durations: Iterable[float] | None = None,
    velocities: Iterable[int] | None = None,
) -> str:
    """Write note events or simple pitch/chord sequences to a MIDI file."""

    events = _coerce_events(events_or_pitches, durations=durations, velocities=velocities)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = m21.stream.Stream()
    stream.append(m21.tempo.MetronomeMark(number=tempo))
    for event in events:
        if len(event.pitches) == 1:
            musical_object: m21.base.Music21Object = m21.note.Note(event.pitches[0])
        else:
            musical_object = m21.chord.Chord(event.pitches)
        musical_object.duration = m21.duration.Duration(event.duration)
        musical_object.volume.velocity = int(event.velocity)
        stream.insert(event.start, musical_object)
    stream.write("midi", fp=str(output_path))
    return str(output_path)


def _coerce_events(
    events_or_pitches: Iterable[NoteEvent | int | float | Iterable[int | float]],
    *,
    durations: Iterable[float] | None,
    velocities: Iterable[int] | None,
) -> tuple[NoteEvent, ...]:
    material = list(events_or_pitches)
    if not material:
        return tuple()
    if isinstance(material[0], NoteEvent):
        return tuple(material)  # type: ignore[return-value]

    duration_values = list(durations) if durations is not None else [1.0] * len(material)
    velocity_values = list(velocities) if velocities is not None else [80] * len(material)

    events: list[NoteEvent] = []
    start = 0.0
    for index, item in enumerate(material):
        if isinstance(item, (int, float)):
            pitches = (int(item),)
        else:
            pitches = tuple(int(value) for value in item)
        duration = float(duration_values[index])
        velocity = int(velocity_values[index])
        events.append(NoteEvent(pitches=pitches, start=start, duration=duration, velocity=velocity))
        start += duration
    return tuple(events)
