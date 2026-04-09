"""General-purpose helpers for tunetx."""

from .collections import deduplicate, pairwise
from .distances import (
    generic_vector_distance,
    minimal_non_bijective_pitch_class_distance,
    minimal_pitch_class_distance,
    rhythm_distance,
)
from .parsing import (
    DEFAULT_DURATION_BINS,
    duration_label,
    parse_duration_sequence,
    parse_duration_value,
    parse_fraction_sequence,
)

__all__ = [
    "DEFAULT_DURATION_BINS",
    "deduplicate",
    "duration_label",
    "generic_vector_distance",
    "minimal_non_bijective_pitch_class_distance",
    "minimal_pitch_class_distance",
    "pairwise",
    "parse_duration_sequence",
    "parse_duration_value",
    "parse_fraction_sequence",
    "rhythm_distance",
]
