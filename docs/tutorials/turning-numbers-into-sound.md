# Turning Numbers into Sound

This tutorial shows how to turn a numeric series into notes, MIDI, and WAV audio.

## Map numbers to notes

```{doctest}
>>> map_to_scale([0.0, 5.0, 10.0], scale=[60, 62, 64, 65]).tolist()
[60, 62, 65]
```

What happened? The smallest value went to the lowest note, the middle value landed in the middle of the available note list, and the largest value went to the top note.

## Create note events

```{doctest}
>>> events = series_to_note_events([1, 2, 3], scale=[60, 62, 64], step=0.5, duration=0.25)
>>> [event.pitches for event in events]
[(60,), (62,), (64,)]
>>> [event.start for event in events]
[0.0, 0.5, 1.0]
```

## Write a MIDI file

```python
from tunetx.data import series_to_midi

result = series_to_midi([0, 1, 2, 3], "series.mid", scale="major", duration=0.5)
print(result)
```

Expected output:

```text
series.mid
```

## Render a WAV file

```python
from tunetx.data import sonify_series_to_wav

result = sonify_series_to_wav([0, 1, 2, 3], "series.wav", scale="minor")
print(result)
```

Expected output:

```text
series.wav
```

## What changes when you change the settings?

- A different `scale` changes which notes are available.
- A different `velocity_range` changes the loudness contour.
- A different `step` changes the spacing between notes.
- A different `duration` changes how long each note lasts.
