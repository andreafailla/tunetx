"""Deterministic sonification helpers."""

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
    """Convert a numeric series into scheduled note events."""

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
    """Map a numeric series to MIDI and write it to disk."""

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
    """Render a numeric series to a simple sine-wave WAV file."""

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
