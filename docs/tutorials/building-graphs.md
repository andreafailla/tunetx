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
