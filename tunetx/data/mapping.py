"""Map numeric data to notes.

The helpers in this module turn numbers into MIDI note values. They are used by
the sonification functions and can also be used directly when you want full
control over how a series of numbers is mapped onto a set of notes.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def resolve_scale(scale: str | Iterable[int], midi_range: tuple[int, int] = (48, 84)) -> np.ndarray:
    """Resolve a scale name or note list into MIDI notes.

    Parameters
    ----------
    scale : str or iterable of int
        Either the name of a built-in scale such as ``"major"`` or an explicit
        list of MIDI note numbers such as ``[60, 62, 64]``.
    midi_range : tuple of int, default=(48, 84)
        Lowest and highest MIDI notes to include when ``scale`` is a named
        scale. The defaults cover a practical middle register.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array of MIDI note numbers.

    Raises
    ------
    ValueError
        If ``scale`` is a string but does not match a supported scale name.

    See Also
    --------
    map_to_scale : Map raw numbers onto the notes returned by this helper.

    Notes
    -----
    Built-in scale names return every matching note in the requested MIDI
    range, not just one octave of note classes.

    Examples
    --------
    >>> resolve_scale("major", midi_range=(60, 72)).tolist()
    [60, 62, 64, 65, 67, 69, 71, 72]
    >>> resolve_scale([60, 63, 67]).tolist()
    [60, 63, 67]
    """

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
    """Map a series of numbers onto notes.

    Parameters
    ----------
    values : iterable of float or int
        Numbers to map. The lowest value is sent to the lowest note in the
        chosen scale range, and the highest value is sent to the highest note.
    scale : str or iterable of int, default="chromatic"
        Named scale or explicit list of MIDI note numbers to map onto.
    midi_range : tuple of int, default=(48, 84)
        Lowest and highest MIDI notes available when ``scale`` is a named
        scale.

    Returns
    -------
    numpy.ndarray
        MIDI note numbers with the same length as ``values``.

    Raises
    ------
    ValueError
        If the resolved scale is empty.

    See Also
    --------
    resolve_scale : Expand a named scale into concrete MIDI notes.
    tunetx.data.series_to_note_events : Turn the mapped notes into timed events.

    Notes
    -----
    When all input values are the same, every item maps to the first available
    note. This keeps the result deterministic.

    Examples
    --------
    >>> map_to_scale([0.0, 5.0, 10.0], scale=[60, 62, 64, 65]).tolist()
    [60, 62, 65]
    >>> map_to_scale([3, 3, 3], scale=[60, 62, 64]).tolist()
    [60, 60, 60]
    """

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
