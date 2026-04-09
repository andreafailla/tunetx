"""Public API for the lightweight TuNetX subset."""

from importlib import import_module

from .classes import MidiSet, PitchClassRow, PitchClassSet, RhythmSequence

__all__ = [
    "MidiSet",
    "PitchClassRow",
    "PitchClassSet",
    "RhythmSequence",
    "classes",
    "data",
    "io",
    "networks",
    "utils",
]


def __getattr__(name: str):
    if name in {"classes", "data", "io", "networks", "utils"}:
        return import_module(f"tunetx.{name}")
    raise AttributeError(f"module 'tunetx' has no attribute {name!r}")
