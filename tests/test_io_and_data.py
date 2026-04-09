import wave

from tunetx.data import map_to_scale, series_to_midi, series_to_note_events, sonify_series_to_wav
from tunetx.io import read_midi, read_score, write_midi


def test_mapping_and_event_generation():
    mapped = map_to_scale([0.0, 5.0, 10.0], scale=[60, 62, 64, 65])
    assert mapped.tolist() == [60, 62, 65]

    events = series_to_note_events([1, 2, 3], scale=[60, 62, 64], step=0.5, duration=0.25)
    assert [event.pitches for event in events] == [(60,), (62,), (64,)]
    assert [event.start for event in events] == [0.0, 0.5, 1.0]
    assert [event.duration for event in events] == [0.25, 0.25, 0.25]
    assert [event.velocity for event in events] == [64, 82, 100]


def test_midi_io_roundtrip(tmp_path):
    midi_path = tmp_path / "events.mid"
    write_midi([(60, 64, 67), (62, 65, 69)], str(midi_path), durations=[1.0, 1.0], velocities=[70, 80])

    parsed_midi = read_midi(str(midi_path))
    parsed_score = read_score(str(midi_path))

    assert midi_path.exists()
    assert parsed_midi.tempo is not None
    assert parsed_midi.notes
    assert parsed_midi.chords
    assert parsed_score.chords == parsed_midi.chords


def test_sonification_outputs(tmp_path):
    midi_path = tmp_path / "series.mid"
    wav_path = tmp_path / "series.wav"

    created_midi = series_to_midi([0, 1, 2, 3], str(midi_path), scale="major", duration=0.5)
    created_wav = sonify_series_to_wav([0, 1, 2, 3], str(wav_path), scale="minor")

    assert created_midi == str(midi_path)
    assert created_wav == str(wav_path)
    assert midi_path.exists()
    assert wav_path.exists()
    assert wav_path.stat().st_size > 44

    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getframerate() == 44100
        assert handle.getnframes() > 0


def test_constant_series_uses_low_velocity(tmp_path):
    events = series_to_note_events([5, 5, 5], scale=[60, 62, 64], velocity_range=(50, 90))
    assert [event.velocity for event in events] == [50, 50, 50]

    wav_path = tmp_path / "constant.wav"
    sonify_series_to_wav([5, 5, 5], str(wav_path), scale=[60, 62, 64], velocity_range=(50, 90))
    assert wav_path.exists()
