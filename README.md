# tunetx

`tunetx` is a lightweight Python package for symbolic music, musical networks,
data sonification, and MIDI/score I/O.

It provides:

- pitch and rhythm classes such as `PitchClassSet`, `PitchClassRow`, `MidiSet`,
  and `RhythmSequence`
- descriptor-driven network builders for pitch, rhythm, score, and timbre
- deterministic numeric sonification to MIDI and WAV
- MIDI and MusicXML parsing through `music21`

Core runtime dependencies are limited to `numpy`, `networkx`, and `music21`.

## Acknowledgements

This software is based on a stripped down version of [musicntwrk](https://www.musicntwrk.com/#documentation) by Marco Buongiorno Nardelli, with improved functionalities and performance. 
