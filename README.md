<p align="center">
  <img src="logo_white_back.png" alt="TuneTx logo" width="420">
</p>

# tunetx

`tunetx` is a small Python library for working with notes, rhythms, score files, musical graphs, and data sonification.

It focuses on a compact public API:

- note-group classes such as `PitchClassSet`, `PitchClassRow`, `MidiSet`, and `RhythmSequence`
- graph builders for descriptor, voice-leading, melody-transition, and co-occurrence workflows
- score parsing and MIDI writing
- turning numeric data into notes, MIDI, and WAV audio

## Install

```bash
pip install .
```

For development and docs:

```bash
pip install -e '.[dev,docs]'
```

## Quick example

```python
from tunetx import PitchClassSet
from tunetx.networks import voice_leading_network

major = PitchClassSet((0, 4, 7))
minor = PitchClassSet((0, 3, 7))

print(major.prime_form().pcs)
print(major.voice_leading_operator_name(minor))

graph = voice_leading_network([major, minor], max_distance=2.0)
print(graph.number_of_nodes(), graph.number_of_edges())
```

Expected output:

```text
(0, 3, 7)
R(0,-1,0)
2 1
```

## Documentation

The full documentation now lives in the `docs/` tree and is designed for Read the Docs. It includes:

- plain-English tutorials
- an accessible glossary
- API reference generated from NumPy-style docstrings
- examples with visible outputs

Build locally with:

```bash
sphinx-build -b html docs docs/_build/html
```
