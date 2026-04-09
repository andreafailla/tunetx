# Building Graphs from Musical Material

A graph is a map of relationships.

- A node is one musical object.
- An edge is a connection between two objects.

## Build a descriptor-based graph

```{doctest}
>>> items = [PitchClassSet((0, 1, 2)), PitchClassSet((0, 1, 3)), PitchClassSet((0, 1, 4))]
>>> graph = pitch_class_network(items, max_distance=2.0)
>>> graph.number_of_nodes()
3
>>> graph.number_of_edges()
3
```

What happened? Each node is a note group. Edges were added because the chosen descriptor vectors are close enough under the distance threshold.

## Build a voice-leading graph

```{doctest}
>>> graph = voice_leading_network([PitchClassSet((0, 4, 7)), PitchClassSet((0, 3, 7))], max_distance=2.0)
>>> edge = next(iter(graph.edges(data=True)))
>>> edge[2]["label"]
'R(0,-1,0)'
>>> edge[2]["operator"]
'O(1)'
```

Why this output? The graph records the small note motion between the two note groups.

## Build a rhythm graph

```{doctest}
>>> graph = rhythm_network([RhythmSequence(("q", "e")), RhythmSequence(("q", "s")), RhythmSequence(("e", "s"))], max_distance=2.0)
>>> graph.number_of_nodes()
3
>>> graph.number_of_edges() >= 1
True
```

## Build a melody transition graph

```{doctest}
>>> score = MidiScore(
...     notes=tuple(),
...     chords=(
...         MidiChordSlice(pitches=(60.0, 72.0, 64.0), pitch_classes=(0, 4), start=0.0, duration=1.0, label="CE"),
...         MidiChordSlice(pitches=(60.0, 67.0), pitch_classes=(0, 7), start=1.0, duration=1.0, label="CG"),
...     ),
...     tempo=120.0,
...     meter="4/4",
... )
>>> graph = melody_network(score)
>>> graph.number_of_nodes()
3
>>> graph["C"]["G"]["count"]
1
```

## Build a co-occurrence graph

```{doctest}
>>> graph = cooccurrence_network(score)
>>> graph.has_edge("C", "E")
True
>>> graph["C"]["E"]["count"]
1
```
