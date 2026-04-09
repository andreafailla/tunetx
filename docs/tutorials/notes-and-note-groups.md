# Getting Started with Notes and Note Groups

This tutorial introduces the main note-based classes in `tunetx`.

## A note group without octave

`PitchClassSet` stores note identities without octave. In plain English, that means C4 and C5 are treated as the same note name, C.

```{doctest}
>>> pcs = PitchClassSet((7, 0, 4, 7))
>>> pcs.pcs
(0, 4, 7)
>>> pcs.label()
'CEG'
```

What happened? The input was normalized so repeated notes were removed and the group was sorted.

## Find a compact normalized shape

```{doctest}
>>> pcs.prime_form().pcs
(0, 3, 7)
>>> pcs.interval_vector()
array([0, 0, 1, 1, 1, 0])
```

Why this output? The prime form is a compact way to say “this group has the same shape as other groups like it.” The interval vector counts the distances inside that group.

## Compare two note groups

```{doctest}
>>> major = PitchClassSet((0, 4, 7))
>>> minor = PitchClassSet((0, 3, 7))
>>> major.operator_name(minor)
'O(1)'
>>> major.voice_leading_operator_name(minor)
'R(0,-1,0)'
>>> major.apply_voice_leading("R(0,-1,0)").pcs
(0, 3, 7)
```

What happened? Only one note moved down by one semitone, so the transformation is small and easy to describe.

## Keep octave information when you need it

Use `MidiSet` when the actual register matters.

```{doctest}
>>> midi = MidiSet(("C4", "E4", "G4"))
>>> midi.pitch_names()
('C4', 'E4', 'G4')
>>> midi.transpose(2).midi
(62.0, 66.0, 69.0)
```

## Work with ordered rows

Use `PitchClassRow` when order matters.

```{doctest}
>>> row = PitchClassRow((0, 11, 7, 4, 2, 9))
>>> row.transpose(2).pcs
(2, 1, 9, 6, 4, 11)
>>> row.retrograde(1).pcs
(10, 3, 5, 8, 0, 1)
```
