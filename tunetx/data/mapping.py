"""Numeric-to-pitch mapping helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def resolve_scale(scale: str | Iterable[int], midi_range: tuple[int, int] = (48, 84)) -> np.ndarray:
    """Resolve a named scale or explicit pitch collection into MIDI pitches."""

    if not isinstance(scale, str):
        return np.asarray(list(scale), dtype=int)

    low, high = midi_range
    named_scales = {
        "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10),
        "pentatonic": (0, 2, 4, 7, 9),
        "whole_tone": (0, 2, 4, 6, 8, 10),
    }
    if scale not in named_scales:
        raise ValueError(f"unsupported scale name: {scale}")

    pcs = named_scales[scale]
    pitches = [midi for midi in range(low, high + 1) if midi % 12 in pcs]
    return np.asarray(pitches, dtype=int)


def map_to_scale(
    values: Iterable[float | int],
    scale: str | Iterable[int] = "chromatic",
    midi_range: tuple[int, int] = (48, 84),
) -> np.ndarray:
    """Map a numeric series onto a scale in MIDI space."""

    source = np.asarray(list(values), dtype=float)
    pitches = resolve_scale(scale, midi_range=midi_range)
    if len(source) == 0:
        return np.asarray([], dtype=int)
    if len(pitches) == 0:
        raise ValueError("resolved scale cannot be empty")

    minimum = float(source.min())
    maximum = float(source.max())
    if np.isclose(minimum, maximum):
        indices = np.zeros(len(source), dtype=int)
    else:
        normalized = (source - minimum) / (maximum - minimum)
        indices = np.floor(normalized * (len(pitches) - 1)).astype(int)
    return pitches[np.clip(indices, 0, len(pitches) - 1)]
