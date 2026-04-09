# Concepts

This page connects the package vocabulary to the actual behavior of the code.

## Notes without octave

Many `tunetx` tools work with note identities without octave. In music theory this is called a *pitch class*. In practice it means that C4 and C5 are both treated as the same note name, C.

Use `PitchClassSet` when octave does not matter and you want to compare note groups by shape.

Use `MidiSet` when octave does matter and you want concrete MIDI notes.

## Time values as exact fractions

`RhythmSequence` stores note lengths as exact fractions rather than decimal approximations. This keeps operations such as sorting, normalization, and histogram building predictable.

For example, the token `q` means a quarter note, and `e` means an eighth note.

## Graphs as maps of musical similarity and change

In `tunetx`, a graph is a map:

- A node is one musical object, such as a note group or rhythm pattern.
- An edge is a relationship between two objects.

Descriptor graphs connect things that *look similar* according to a summary vector.

Voice-leading graphs connect things that can move into each other with small note changes.

Score graphs connect note groups that appear one after another in a parsed score.

## Network types at a glance

| Function | What the nodes are | What the edges mean | Best for |
| --- | --- | --- | --- |
| `pitch_class_network()` | Note groups without octave | Two note groups have similar descriptor vectors | Comparing note-group shape |
| `voice_leading_network()` | Note groups without octave | One note group can move to another with small note changes | Studying harmonic motion |
| `rhythm_network()` | Rhythm patterns | Two rhythms have similar descriptor vectors | Comparing note-length patterns |
| `timbre_network()` | WAV files or timbre summaries | Two sounds have similar lightweight timbre features | Comparing sound character |
| `score_network()` | Repeating simultaneous note groups from a score | One score slice follows another in time | Studying progression in a real piece |

## Parsed score reuse

`read_score()` returns a `MidiScore` object. If you plan to inspect the same score more than once, keep that object and reuse it instead of reparsing the file each time.

That matters because file parsing is usually slower than building a graph from an already parsed score.
