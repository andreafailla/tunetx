"""Turn numeric data into timed notes, MIDI files, and WAV files."""

from __future__ import annotations

import wave
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from ..io import NoteEvent, write_midi
from .mapping import map_to_scale


def series_to_note_events(
    values: Iterable[float | int],
    *,
    scale: str | Iterable[int] = "chromatic",
    midi_range: tuple[int, int] = (48, 84),
    step: float = 1.0,
    duration: float = 1.0,
    velocity_range: tuple[int, int] = (64, 100),
) -> tuple[NoteEvent, ...]:
    """Convert numbers into scheduled note events.

    Parameters
    ----------
    values : iterable of float or int
        Numbers to sonify.
    scale : str or iterable of int, default="chromatic"
        Named scale or explicit MIDI notes used for the pitch mapping.
    midi_range : tuple of int, default=(48, 84)
        Lowest and highest note available when using a named scale.
    step : float, default=1.0
        Time between note starts, measured in quarter-note units.
    duration : float, default=1.0
        Length of each note, measured in quarter-note units.
    velocity_range : tuple of int, default=(64, 100)
        Lowest and highest MIDI velocity. Lower input values get lower
        velocities.

    Returns
    -------
    tuple of NoteEvent
        Timed note events ready to write to MIDI or render to audio.

    See Also
    --------
    series_to_midi : Write the generated events to a MIDI file.
    sonify_series_to_wav : Render the generated events to a WAV file.

    Notes
    -----
    This function always returns single-note events. Chords can still be built
    later by combining `NoteEvent` objects manually.

    Examples
    --------
    >>> events = series_to_note_events([1, 2, 3], scale=[60, 62, 64], step=0.5, duration=0.25)
    >>> [event.pitches for event in events]
    [(60,), (62,), (64,)]
    >>> [event.start for event in events]
    [0.0, 0.5, 1.0]
    >>> [event.velocity for event in events]
    [64, 82, 100]
    """

    source = np.asarray(list(values), dtype=float)
    if len(source) == 0:
        return tuple()

    pitches = map_to_scale(source, scale=scale, midi_range=midi_range)
    velocities = _velocity_values(source, velocity_range)
    starts = np.arange(len(source), dtype=float) * step
    return tuple(
        NoteEvent(
            pitches=(int(pitch),),
            start=float(start),
            duration=duration,
            velocity=int(velocity),
        )
        for pitch, start, velocity in zip(pitches.tolist(), starts.tolist(), velocities.tolist())
    )


def series_to_midi(
    values: Iterable[float | int],
    path: str,
    *,
    scale: str | Iterable[int] = "chromatic",
    midi_range: tuple[int, int] = (48, 84),
    step: float = 1.0,
    duration: float = 1.0,
    tempo: float = 120.0,
    velocity_range: tuple[int, int] = (64, 100),
) -> str:
    """Write a numeric series to a MIDI file.

    Parameters
    ----------
    values : iterable of float or int
        Numbers to sonify.
    path : str
        Output path for the MIDI file.
    scale : str or iterable of int, default="chromatic"
        Named scale or explicit MIDI notes used for the pitch mapping.
    midi_range : tuple of int, default=(48, 84)
        Lowest and highest note available when using a named scale.
    step : float, default=1.0
        Time between note starts, measured in quarter-note units.
    duration : float, default=1.0
        Length of each note, measured in quarter-note units.
    tempo : float, default=120.0
        Tempo written into the MIDI file.
    velocity_range : tuple of int, default=(64, 100)
        Lowest and highest MIDI velocity.

    Returns
    -------
    str
        The output path, returned unchanged for convenient chaining.

    See Also
    --------
    series_to_note_events : Build the underlying timed notes first.
    tunetx.io.write_midi : Lower-level MIDI writing helper.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> path = os.path.join(tempfile.gettempdir(), "tunetx-series.mid")
    >>> result = series_to_midi([0, 1, 2, 3], path, scale="major", duration=0.5)  # doctest: +ELLIPSIS
    >>> result.endswith("tunetx-series.mid")
    True
    """

    events = series_to_note_events(
        values,
        scale=scale,
        midi_range=midi_range,
        step=step,
        duration=duration,
        velocity_range=velocity_range,
    )
    return write_midi(events, path, tempo=tempo)


def sonify_series_to_wav(
    values: Iterable[float | int],
    path: str,
    *,
    scale: str | Iterable[int] = "chromatic",
    midi_range: tuple[int, int] = (48, 84),
    step: float = 0.5,
    duration: float = 0.45,
    sample_rate: int = 44100,
    velocity_range: tuple[int, int] = (64, 100),
) -> str:
    """Render a numeric series to a simple WAV file.

    Parameters
    ----------
    values : iterable of float or int
        Numbers to sonify.
    path : str
        Output path for the WAV file.
    scale : str or iterable of int, default="chromatic"
        Named scale or explicit MIDI notes used for the pitch mapping.
    midi_range : tuple of int, default=(48, 84)
        Lowest and highest note available when using a named scale.
    step : float, default=0.5
        Time between note starts, measured in seconds for the rendered audio.
    duration : float, default=0.45
        Length of each note in seconds.
    sample_rate : int, default=44100
        Audio sample rate in Hertz.
    velocity_range : tuple of int, default=(64, 100)
        Lowest and highest event amplitudes before normalization.

    Returns
    -------
    str
        The output path, returned unchanged for convenient chaining.

    See Also
    --------
    series_to_note_events : Build the note plan used by the renderer.
    series_to_midi : Write the same musical mapping to a MIDI file.

    Notes
    -----
    The waveform is a simple sine tone with a short fade-in and fade-out. It is
    intended for predictable examples and lightweight prototyping rather than
    polished synthesis.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> path = os.path.join(tempfile.gettempdir(), "tunetx-series.wav")
    >>> result = sonify_series_to_wav([0, 1, 2, 3], path, scale="minor")
    >>> result.endswith("tunetx-series.wav")
    True
    """

    events = series_to_note_events(
        values,
        scale=scale,
        midi_range=midi_range,
        step=step,
        duration=duration,
        velocity_range=velocity_range,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not events:
        samples = np.zeros(sample_rate // 4, dtype=np.float32)
    else:
        total_duration = max(event.start + event.duration for event in events)
        total_samples = max(int(total_duration * sample_rate), 1)
        samples = np.zeros(total_samples, dtype=np.float32)
        for event in events:
            start = int(event.start * sample_rate)
            length = max(int(event.duration * sample_rate), 1)
            t = np.arange(length, dtype=np.float32) / sample_rate
            frequency = _midi_frequency(event.pitches[0])
            envelope = _cached_envelope(length, sample_rate)
            amplitude = event.velocity / 127.0
            tone = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32) * envelope
            samples[start : start + length] += tone

    peak = float(np.max(np.abs(samples))) or 1.0
    pcm = np.clip(samples / peak, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return str(output_path)


def _velocity_values(source: np.ndarray, velocity_range: tuple[int, int]) -> np.ndarray:
    minimum = float(source.min())
    maximum = float(source.max())
    low_velocity, high_velocity = velocity_range
    if np.isclose(minimum, maximum):
        return np.full(len(source), low_velocity, dtype=int)
    normalized = (source - minimum) / (maximum - minimum)
    return np.rint(low_velocity + normalized * (high_velocity - low_velocity)).astype(int)


@lru_cache(maxsize=None)
def _midi_frequency(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


@lru_cache(maxsize=None)
def _cached_envelope(length: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float32)
    fade = min(int(sample_rate * 0.01), max(length // 2, 1))
    if fade > 0:
        envelope[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        envelope[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return envelope
