# Examples

These are short, task-oriented recipes.

## Compare two note groups

```{doctest}
>>> major = PitchClassSet((0, 4, 7))
>>> minor = PitchClassSet((0, 3, 7))
>>> major.operator_name(minor)
'O(1)'
>>> major.voice_leading_operator_name(minor)
'R(0,-1,0)'
```

## Enumerate small note groups

```{doctest}
>>> items = enumerate_pitch_class_sets(3)
>>> items[0].pcs
(0, 1, 2)
>>> items[1].pcs
(0, 1, 3)
```

## Describe a rhythm pattern

```{doctest}
>>> description = describe_rhythm_sequence(("q", "e", "e"))
>>> description.binary_onsets
(1, 0, 1, 1)
>>> description.duration_vector
(2, 0, 0, 0, 0, 0, 0, 0)
```

## Turn a numeric series into notes

```{doctest}
>>> events = series_to_note_events([1, 2, 3], scale=[60, 62, 64], step=0.5, duration=0.25)
>>> [(event.pitches, event.start) for event in events]
[((60,), 0.0), ((62,), 0.5), ((64,), 1.0)]
```
