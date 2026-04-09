# Working with Rhythms

This tutorial introduces `RhythmSequence`, which stores note lengths as exact fractions.

## Start with shorthand tokens

`q` means quarter note, `e` means eighth note, and `s` means sixteenth note.

```{doctest}
>>> rhythm = RhythmSequence(("q", "e", "e"))
>>> rhythm.durations
(Fraction(1, 4), Fraction(1, 8), Fraction(1, 8))
>>> rhythm.label()
'1/4 1/8 1/8'
```

## Normalize and reverse the rhythm

```{doctest}
>>> rhythm.normal_order().durations
(Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
>>> rhythm.retrograde().durations
(Fraction(1, 8), Fraction(1, 8), Fraction(1, 4))
```

What happened? `normal_order()` rotates the pattern into a compact arrangement, while `retrograde()` simply reverses it.

## Turn the rhythm into a binary onset pattern

```{doctest}
>>> rhythm.binary_onsets()
[1, 0, 1, 1]
```

Why this output? The pattern is expanded onto the smallest shared time grid. A `1` means “a note starts here” and a `0` means “the note is still continuing.”

## Build rhythm descriptors

```{doctest}
>>> description = describe_rhythm_sequence(("q", "e", "e"))
>>> description.duration_vector
(2, 0, 0, 0, 0, 0, 0, 0)
>>> description.inter_onset_vector
(2, 2, 2, 0, 0, 0, 0, 0)
```

These vectors are useful when you want to compare rhythm patterns or build graphs from them.
