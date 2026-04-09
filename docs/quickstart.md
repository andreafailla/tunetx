# Quickstart

This page gives one short example for each major workflow in `tunetx`.

## Work with note groups

```{doctest}
>>> pcs = PitchClassSet((0, 4, 7))
>>> pcs.label()
'CEG'
>>> pcs.prime_form().pcs
(0, 3, 7)
```

## Work with rhythms

```{doctest}
>>> rhythm = RhythmSequence(("q", "e", "e"))
>>> rhythm.normal_order().durations
(Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
>>> rhythm.binary_onsets()
[1, 0, 1, 1]
```

## Build a graph

```{doctest}
>>> graph = voice_leading_network([PitchClassSet((0, 4, 7)), PitchClassSet((0, 3, 7))], max_distance=2.0)
>>> graph.number_of_nodes()
2
>>> graph.number_of_edges()
1
```

## Build score-note graphs

```{doctest}
>>> score = MidiScore(
...     notes=tuple(),
...     chords=(
...         MidiChordSlice(pitches=(60.0, 64.0), pitch_classes=(0, 4), start=0.0, duration=1.0, label="CE"),
...         MidiChordSlice(pitches=(60.0, 67.0), pitch_classes=(0, 7), start=1.0, duration=1.0, label="CG"),
...     ),
...     tempo=120.0,
...     meter="4/4",
... )
>>> melody = melody_network(score)
>>> cooccurrence = cooccurrence_network(score)
>>> melody["C"]["G"]["count"]
1
>>> cooccurrence["C"]["E"]["count"]
1
```

## Read a real MIDI file

The score examples in this documentation use a small public-domain MIDI file from Wikimedia Commons:
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

## Turn numbers into notes

```{doctest}
>>> map_to_scale([0.0, 5.0, 10.0], scale=[60, 62, 64, 65]).tolist()
[60, 62, 65]
>>> [event.pitches for event in series_to_note_events([1, 2, 3], scale=[60, 62, 64])]
[(60,), (62,), (64,)]
```

## Next steps

- For guided walkthroughs, go to the tutorial pages.
- For terminology, see the [glossary](glossary.md).
- For the full API, see the [reference](reference/index.md).
