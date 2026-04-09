"""Numeric mapping and deterministic sonification."""

from .mapping import map_to_scale, resolve_scale
from .sonification import series_to_midi, series_to_note_events, sonify_series_to_wav

__all__ = [
    "map_to_scale",
    "resolve_scale",
    "series_to_midi",
    "series_to_note_events",
    "sonify_series_to_wav",
]
