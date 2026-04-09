# Changelog

## Unreleased

### Performance
- Optimized pitch-class distance evaluation with cached offset grids and cached non-bijective extension candidates while preserving existing distance semantics.
- Reduced repeated normalization and descriptor coercion inside graph-building workflows, especially `voice_leading_network`.
- Reused parsed score data more aggressively by extracting chord slices during `read_score` and reusing those slices in score-derived workflows.
- Reduced sonification overhead by vectorizing event construction and caching note envelopes and MIDI-to-frequency conversion in WAV rendering.
- Reduced repeated symbolic work in pitch-class description and enumeration helpers.

### Tooling
- Added `scripts/benchmark_performance.py` for repeatable measurements of the main performance-sensitive workflows.
