# Reading a Score and Building Score Graphs

This tutorial shows the score workflow in plain English: parse once, inspect the result, and reuse it for several graph views.

## Parse a score

`read_score()` returns a `MidiScore` object that stores notes, simultaneous note groups, tempo, and meter.

This tutorial uses a real public-domain MIDI file from Wikimedia Commons:
[Beethoven Op. 10 No. 1 opening](https://commons.wikimedia.org/wiki/File:Beethoven_Op._10-1_opening.mid).

```python
from pathlib import Path
from urllib.request import urlretrieve

from tunetx.io import read_score

url = "https://commons.wikimedia.org/wiki/Special:FilePath/Beethoven_Op._10-1_opening.mid"
path = Path("beethoven-opening.mid")
urlretrieve(url, path)

score = read_score(str(path))
print(type(score).__name__)
print(len(score.notes) > 0)
print(len(score.chords) > 0)
```

Expected output:

```text
MidiScore
True
True
```

## Reuse the parsed score for progression structure

Once you have a `MidiScore`, pass it to other functions instead of reparsing the file.

```python
from tunetx.io import extract_chord_slices
from tunetx.networks import score_network

slices = extract_chord_slices(score)
graph = score_network(score, max_distance=12.0)
print(len(slices))
print(graph.number_of_nodes())
```

What happened? `extract_chord_slices()` gives you the simultaneous note groups. `score_network()` then builds a directed graph from one slice to the next.

Because the file is short, it works well as a quick reproducible example. For larger analyses, keep the same `MidiScore` object in memory and reuse it.

## Build note-level score graphs

The same parsed score can also drive note-level networks.

```python
from tunetx.networks import cooccurrence_network, melody_network

melody = melody_network(score)
cooccurrence = cooccurrence_network(score)
print(melody.number_of_nodes())
print(cooccurrence.number_of_edges() > 0)
```

What happened? `melody_network()` records which note identities move into which others from one slice to the next. `cooccurrence_network()` records which note identities sound together inside the same slice.

If you want to keep octave, use `node_identity="midi"` instead of the default pitch-class view.

## Why reuse matters

Parsing the file is usually the expensive step. Reusing `MidiScore` keeps repeated analysis faster and makes your workflow easier to reason about.
