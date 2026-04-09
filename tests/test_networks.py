import wave
from pathlib import Path

import numpy as np

from tunetx import PitchClassSet, RhythmSequence
from tunetx.data import series_to_midi
from tunetx.io import MidiChordSlice, MidiScore, NoteEvent, read_score, write_midi
from tunetx.networks import (
    cooccurrence_network,
    describe_pitch_class_set,
    describe_rhythm_sequence,
    describe_timbre,
    enumerate_pitch_class_sets,
    enumerate_rhythm_sequences,
    melody_network,
    motif_network,
    multilayer_network,
    pitch_class_network,
    rhythm_network,
    score_network,
    timbre_network,
    voice_leading_network,
)


def _write_sine(path: Path, frequency: float, sample_rate: int = 22050) -> None:
    duration = 0.25
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    signal = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(signal.tobytes())


def _example_score() -> MidiScore:
    return MidiScore(
        notes=tuple(),
        chords=(
            MidiChordSlice(pitches=(60.0, 72.0, 64.0), pitch_classes=(0, 4), start=0.0, duration=1.0, label="CE"),
            MidiChordSlice(pitches=(60.0, 67.0), pitch_classes=(0, 7), start=1.0, duration=1.0, label="CG"),
            MidiChordSlice(pitches=(60.0, 72.0, 64.0), pitch_classes=(0, 4), start=2.0, duration=1.0, label="CE"),
            MidiChordSlice(pitches=(60.0, 67.0), pitch_classes=(0, 7), start=3.0, duration=1.0, label="CG"),
        ),
        tempo=120.0,
        meter="4/4",
    )


def test_enumeration_and_description_helpers():
    pcs_items = enumerate_pitch_class_sets(3)
    rhythm_items = enumerate_rhythm_sequences(2, ("q", "e", "s"))

    assert pcs_items[0].pcs == (0, 1, 2)
    assert all(isinstance(item, PitchClassSet) for item in pcs_items[:5])
    assert all(isinstance(item, RhythmSequence) for item in rhythm_items)

    pcs_description = describe_pitch_class_set((0, 4, 7))
    rhythm_description = describe_rhythm_sequence(("q", "e", "e"))
    assert pcs_description.label == "CEG"
    assert pcs_description.interval_vector == (0, 0, 1, 1, 1, 0)
    assert rhythm_description.binary_onsets == (1, 0, 1, 1)


def test_pitch_and_voice_leading_networks():
    items = [
        PitchClassSet((0, 1, 2)),
        PitchClassSet((0, 1, 3)),
        PitchClassSet((0, 1, 4)),
    ]
    graph = pitch_class_network(items, max_distance=2.0)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
    assert graph.nodes[0]["label"] == "CC#D"
    assert np.isclose(graph.edges[0, 1]["weight"], 1 / np.sqrt(2))
    assert np.isclose(graph.edges[0, 2]["weight"], 0.5)

    vl_graph = voice_leading_network([PitchClassSet((0, 4, 7)), PitchClassSet((0, 3, 7))], max_distance=2.0)
    assert vl_graph.number_of_edges() == 1
    edge = next(iter(vl_graph.edges(data=True)))
    assert edge[2]["label"] == "R(0,-1,0)"
    assert edge[2]["operator"] == "O(1)"


def test_motif_network_detects_transposition_and_inversion():
    items = [
        PitchClassSet((0, 4, 7)),
        PitchClassSet((3, 7, 10)),
        PitchClassSet((0, 5, 8)),
        PitchClassSet((0, 2, 7)),
    ]

    graph = motif_network(items)
    assert graph.number_of_nodes() == 4
    assert graph.has_edge(0, 1)
    assert graph.edges[0, 1]["relation"] == "transposition"
    assert graph.edges[0, 1]["label"] == "T3"
    assert graph.has_edge(0, 2)
    assert graph.edges[0, 2]["relation"] == "inversion"
    assert graph.edges[0, 2]["label"] == "I0"
    assert not graph.has_edge(0, 3)


def test_melody_network_supports_pitch_class_and_midi_nodes():
    score = _example_score()

    pitch_class_graph = melody_network(score)
    assert pitch_class_graph.is_directed()
    assert sorted(pitch_class_graph.nodes) == ["C", "E", "G"]
    assert pitch_class_graph.nodes["C"]["kind"] == "melody_note"
    assert pitch_class_graph.nodes["C"]["object"] == "C"
    assert pitch_class_graph["C"]["G"]["count"] == 2
    assert pitch_class_graph["C"]["G"]["weight"] == 2
    assert pitch_class_graph["E"]["C"]["count"] == 2

    midi_graph = melody_network(score, node_identity="midi")
    assert sorted(midi_graph.nodes) == [60, 64, 67, 72]
    assert midi_graph.nodes[60]["label"] == "C4"
    assert midi_graph.nodes[72]["label"] == "C5"
    assert midi_graph.nodes[60]["object"] == 60
    assert midi_graph[72][67]["count"] == 2
    assert midi_graph[64][60]["count"] == 2


def test_cooccurrence_network_counts_within_slice_pairs():
    score = _example_score()

    pitch_class_graph = cooccurrence_network(score)
    assert not pitch_class_graph.is_directed()
    assert sorted(pitch_class_graph.nodes) == ["C", "E", "G"]
    assert pitch_class_graph.nodes["C"]["kind"] == "cooccurrence_note"
    assert pitch_class_graph["C"]["E"]["count"] == 2
    assert pitch_class_graph["C"]["G"]["weight"] == 2
    assert not pitch_class_graph.has_edge("E", "G")

    midi_graph = cooccurrence_network(score, node_identity="midi")
    assert sorted(midi_graph.nodes) == [60, 64, 67, 72]
    assert midi_graph.nodes[60]["label"] == "C4"
    assert midi_graph[60][72]["count"] == 2
    assert midi_graph[60][64]["count"] == 2
    assert not midi_graph.has_edge(64, 67)


def test_rhythm_timbre_and_score_networks(tmp_path):
    rhythm_graph = rhythm_network(
        [RhythmSequence(("q", "e")), RhythmSequence(("q", "s")), RhythmSequence(("e", "s"))],
        max_distance=2.0,
    )
    assert rhythm_graph.number_of_nodes() == 3
    assert rhythm_graph.number_of_edges() >= 1

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_sine(wav_a, 220.0)
    _write_sine(wav_b, 330.0)
    desc_a = describe_timbre(wav_a)
    desc_b = describe_timbre(wav_b)
    assert desc_a.label == "a"
    assert desc_b.duration > 0

    timbre_graph = timbre_network([str(wav_a), str(wav_b)], max_distance=10000.0)
    assert timbre_graph.number_of_nodes() == 2
    assert timbre_graph.number_of_edges() == 1

    midi_path = tmp_path / "progression.mid"
    series_to_midi([0, 1, 2], str(midi_path), scale=[60, 63, 67], duration=1.0)
    graph = score_network(str(midi_path), max_distance=12.0)
    assert graph.number_of_nodes() >= 2
    assert all("label" in data for _, data in graph.nodes(data=True))


def test_multilayer_network_stacks_layers_and_aligns_by_index():
    graph = multilayer_network(
        pitch_items=[PitchClassSet((0, 4, 7)), PitchClassSet((0, 3, 7))],
        rhythm_items=[RhythmSequence(("q", "e")), RhythmSequence(("q", "s"))],
        pitch_kwargs={"max_distance": 3.0},
        rhythm_kwargs={"max_distance": 3.0},
    )

    assert ("pitch", 0) in graph.nodes
    assert ("rhythm", 0) in graph.nodes
    assert graph.nodes[("pitch", 0)]["layer"] == "pitch"
    assert any(data["layer"] == "pitch" for _, _, data in graph.edges(data=True))
    assert any(data["layer"] == "rhythm" for _, _, data in graph.edges(data=True))
    assert graph.has_edge(("pitch", 0), ("rhythm", 0))
    alignment_edges = graph.get_edge_data(("pitch", 0), ("rhythm", 0))
    assert any(data["layer"] == "alignment" for data in alignment_edges.values())


def test_score_note_networks_match_preparsed_score(tmp_path):
    midi_path = tmp_path / "note-networks.mid"
    write_midi(
        [
            NoteEvent(pitches=(60, 72, 64), start=0.0, duration=1.0),
            NoteEvent(pitches=(60, 67), start=1.0, duration=1.0),
            NoteEvent(pitches=(60, 72, 64), start=2.0, duration=1.0),
            NoteEvent(pitches=(60, 67), start=3.0, duration=1.0),
        ],
        str(midi_path),
    )

    parsed = read_score(str(midi_path))

    melody_from_path = melody_network(str(midi_path))
    melody_from_score = melody_network(parsed)
    assert sorted(melody_from_path.nodes(data="label")) == sorted(melody_from_score.nodes(data="label"))
    assert sorted(melody_from_path.edges(data="count")) == sorted(melody_from_score.edges(data="count"))

    cooccurrence_from_path = cooccurrence_network(str(midi_path), node_identity="midi")
    cooccurrence_from_score = cooccurrence_network(parsed, node_identity="midi")
    assert sorted(cooccurrence_from_path.nodes(data="label")) == sorted(cooccurrence_from_score.nodes(data="label"))
    assert sorted(cooccurrence_from_path.edges(data="count")) == sorted(cooccurrence_from_score.edges(data="count"))


def test_score_network_matches_preparsed_score(tmp_path):
    midi_path = tmp_path / "progression.mid"
    series_to_midi([0, 1, 2, 1], str(midi_path), scale=[60, 63, 67], duration=1.0)

    parsed = read_score(str(midi_path))
    from_path = score_network(str(midi_path), max_distance=12.0)
    from_score = score_network(parsed, max_distance=12.0)

    assert from_path.number_of_nodes() == from_score.number_of_nodes()
    assert from_path.number_of_edges() == from_score.number_of_edges()
    assert sorted(from_path.nodes(data="label")) == sorted(from_score.nodes(data="label"))
    assert sorted(from_path.edges(data="label")) == sorted(from_score.edges(data="label"))
