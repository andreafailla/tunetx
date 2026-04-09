# tunetx

`tunetx` is a lightweight Python library for symbolic music analysis, musical-network construction, score/MIDI I/O, and deterministic sonification.

It is designed as a smaller, dependency-light subset of the broader `musicntwrk` ecosystem, with a focus on:

- pitch-class and rhythm-domain data structures
- descriptor-based network workflows
- score-derived harmonic/voice-leading analysis
- numeric-series sonification to MIDI and WAV

Compared to `musicntwrk`, this package has improved performance, less dependencies, and a more pythonic interface.

## Features

- Symbolic music classes:
  - `PitchClassSet`
  - `PitchClassRow`
  - `MidiSet`
  - `RhythmSequence`
- Descriptor helpers:
  - `describe_pitch_class_set`
  - `describe_rhythm_sequence`
  - `describe_timbre`
- Enumeration helpers:
  - `enumerate_pitch_class_sets`
  - `enumerate_rhythm_sequences`
- Network builders:
  - `pitch_class_network`
  - `voice_leading_network`
  - `rhythm_network`
  - `timbre_network`
  - `score_network`
- I/O:
  - `read_score`
  - `read_midi`
  - `write_midi`
  - `extract_chord_slices`
  - `pitch_class_sequence_from_score`
- Data sonification:
  - `map_to_scale`
  - `series_to_note_events`
  - `series_to_midi`
  - `sonify_series_to_wav`

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e '.[dev]'
```

Runtime dependencies are intentionally small:

- `numpy`
- `networkx`
- `music21`

## Package Layout

```text
tunetx/
  classes/    # pitch, MIDI, and rhythm classes
  networks/   # descriptors, enumeration, graph builders, score networks
  io/         # score parsing and MIDI writing
  data/       # scale mapping and sonification
  utils/      # distances and support utilities
```

## Quick Start

### Pitch-class sets and descriptors

```python
from tunetx import PitchClassSet
from tunetx.networks import describe_pitch_class_set

pcs = PitchClassSet((0, 4, 7))
print(pcs.prime_form().pcs)          # (0, 3, 7)
print(pcs.interval_vector())         # [0 0 1 1 1 0]

description = describe_pitch_class_set(pcs)
print(description.label)             # CEG
print(description.lis_vector)        # (4, 3, 5)
```

### Build a voice-leading network

```python
from tunetx import PitchClassSet
from tunetx.networks import voice_leading_network

items = [
    PitchClassSet((0, 4, 7)),
    PitchClassSet((0, 3, 7)),
    PitchClassSet((0, 5, 9)),
]

graph = voice_leading_network(items, max_distance=2.0)
print(graph.number_of_nodes())
print(graph.number_of_edges())
```

### Parse a score and derive a network

```python
from tunetx.io import read_score
from tunetx.networks import score_network

score = read_score("example.mid")
graph = score_network(score, max_distance=12.0)
```

Passing a parsed `MidiScore` is the preferred fast path when you are running more than one analysis on the same material.

### Sonify a numeric series

```python
from tunetx.data import series_to_midi, sonify_series_to_wav

series_to_midi([0, 1, 2, 3], "series.mid", scale="major", duration=0.5)
sonify_series_to_wav([0, 1, 2, 3], "series.wav", scale="minor")
```

## Main Workflows

### 1. Symbolic set and row analysis

Use `PitchClassSet`, `PitchClassRow`, `MidiSet`, and `RhythmSequence` for common transformations:

- normal order
- prime form
- interval vectors
- inversion, transposition, multiplication
- voice-leading operator naming/application

### 2. Descriptor and enumeration workflows

Use the `tunetx.networks` descriptor helpers to:

- enumerate spaces of pitch-class sets or rhythm cells
- compute descriptor vectors once
- feed those descriptors into network construction pipelines

### 3. Network analysis

`tunetx` supports several graph types:

- descriptor-space pitch graphs
- voice-leading graphs
- rhythm-distance graphs
- timbre-distance graphs for WAV files
- score-derived directed graphs from consecutive chord slices

All graph builders use NetworkX graphs and attach useful node/edge metadata such as labels, descriptors, distances, weights, and operators.

### 4. Score parsing and reuse

`read_score()` normalizes MIDI or MusicXML into a `MidiScore` object containing:

- flattened note events
- chord slices
- tempo
- meter

Downstream APIs such as `extract_chord_slices()`, `pitch_class_sequence_from_score()`, and `score_network()` can reuse that parsed object directly.

### 5. Deterministic sonification

The `tunetx.data` module maps numeric data into MIDI pitches and velocities and can render:

- structured `NoteEvent` sequences
- MIDI files via `music21`
- simple sine-wave WAV files

## Performance

Recent work in this repo focused on the main bottlenecks:

- cached/vectorized pitch-class distance evaluation
- reduced redundant normalization in graph workflows
- better score-data reuse once a `MidiScore` is parsed
- lower-overhead event generation and cached envelopes/frequencies in sonification
- reduced duplicate work in pitch-class description/enumeration

Current benchmark results from `scripts/benchmark_performance.py` in this repo state:

- `voice_leading_network(40 triads)`: about `0.070s/run`
- `read_score(mid)`: about `0.028s/run`
- `score_network(mid)`: about `0.028s/run`
- `score_network(MidiScore)`: about `0.00086s/run`
- `series_to_midi(100 points)`: about `0.010s/run`
- `sonify_series_to_wav(100 points)`: about `0.014s/run`

Observed baseline before the recent optimization pass:

- `voice_leading_network(40 triads)`: about `0.135s/run`
- `read_score(mid)`: about `0.033s/run`
- `score_network(mid)`: about `0.029s/run`
- `series_to_midi(200 points)`: about `0.022s/run`
- `sonify_series_to_wav(200 points)`: about `0.029s/run`

Interpretation:

- voice-leading network construction is roughly 2x faster than the earlier baseline
- score parsing remains dominated by `music21`, but parsed-score reuse is now much cheaper
- MIDI and WAV sonification are materially faster for repeated batch use

## Development

Run tests:

```bash
pytest -q
```

The repo also includes:

- [CHANGELOG.md](CHANGELOG.md) for implementation-facing changes
- `scripts/benchmark_performance.py` for repeatable micro-benchmarks

## Compatibility Notes

- Python requirement: `>=3.10`
- Public APIs are intended to stay small and stable
- Performance changes should preserve behavior, graph semantics, labels, and operator naming

## Acknowledgements

This project is based on a stripped-down subset of ideas from [musicntwrk](https://www.musicntwrk.com/#documentation) by Marco Buongiorno Nardelli, adapted here with a smaller API surface and targeted performance improvements.
