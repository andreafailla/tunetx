"""Sphinx configuration for the tunetx documentation."""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "tunetx"
author = "Andrea Failla"
copyright = "2026, Andrea Failla"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "tunetx documentation"

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
napoleon_use_ivar = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "music21": ("https://www.music21.org/music21docs", None),
}

doctest_global_setup = """
import os
import tempfile
from fractions import Fraction

import numpy as np

from tunetx import MidiSet, PitchClassRow, PitchClassSet, RhythmSequence
from tunetx.data import map_to_scale, resolve_scale, series_to_midi, series_to_note_events, sonify_series_to_wav
from tunetx.io import MidiChordSlice, MidiNote, MidiScore, NoteEvent, extract_chord_slices, pitch_class_sequence_from_score, read_midi, read_score, write_midi
from tunetx.networks import (
    cooccurrence_network,
    describe_pitch_class_set,
    describe_rhythm_sequence,
    enumerate_pitch_class_sets,
    enumerate_rhythm_sequences,
    melody_network,
    pitch_class_network,
    rhythm_network,
    score_network,
    voice_leading_network,
)
from tunetx.utils import (
    deduplicate,
    duration_label,
    generic_vector_distance,
    minimal_non_bijective_pitch_class_distance,
    minimal_pitch_class_distance,
    pairwise,
    parse_duration_sequence,
    parse_duration_value,
    parse_fraction_sequence,
    rhythm_distance,
)
"""

os.environ.setdefault("PYTHONHASHSEED", "0")
