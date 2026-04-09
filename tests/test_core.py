from fractions import Fraction

import numpy as np

import tunetx
from tunetx import MidiSet, PitchClassRow, PitchClassSet, RhythmSequence
from tunetx.utils import minimal_non_bijective_pitch_class_distance, minimal_pitch_class_distance


def test_top_level_exports_and_public_modules():
    assert set(tunetx.__all__) == {
        "MidiSet",
        "PitchClassRow",
        "PitchClassSet",
        "RhythmSequence",
        "classes",
        "data",
        "io",
        "networks",
        "utils",
    }
    assert not hasattr(tunetx, "RythmSequence")
    assert tunetx.classes.PitchClassSet is PitchClassSet
    assert tunetx.classes.RhythmSequence is RhythmSequence


def test_pitch_class_set_core_behavior():
    pcs = PitchClassSet((7, 0, 4, 7))

    assert pcs.pcs == (0, 4, 7)
    assert pcs.normal_order().pcs == (0, 4, 7)
    assert pcs.normal_zero_order().pcs == (0, 4, 7)
    assert pcs.transpose(3).pcs == (3, 7, 10)
    assert pcs.multiply(5).pcs == (0, 8, 11)
    assert pcs.zero_order().pcs == (0, 4, 7)
    assert pcs.invert(2).pcs == (2, 10, 7)
    assert pcs.prime_form().pcs == (0, 3, 7)
    np.testing.assert_array_equal(pcs.interval_vector(), np.array([0, 0, 1, 1, 1, 0]))
    np.testing.assert_array_equal(pcs.lis_vector(), np.array([4, 3, 5]))


def test_pitch_class_set_operator_behavior():
    major = PitchClassSet((0, 4, 7))
    minor = PitchClassSet((0, 3, 7))

    assert major.neo_riemannian("P").pcs == (0, 3, 7)
    assert major.voice_leading_operator_name(minor) == "R(0,-1,0)"
    assert major.operator_name(minor) == "O(1)"
    assert major.apply_voice_leading("R(0,-1,0)").pcs == (0, 3, 7)


def test_pitch_class_distance_metrics_are_stable():
    major = (0, 4, 7)
    minor = (0, 3, 7)

    distance, mapping = minimal_pitch_class_distance(major, minor, metric="euclidean")
    assert np.isclose(distance, 1.0)
    np.testing.assert_array_equal(mapping, np.array([0, 3, 7]))

    distance, mapping = minimal_pitch_class_distance(major, minor, metric="manhattan")
    assert np.isclose(distance, 1.0)
    np.testing.assert_array_equal(mapping, np.array([0, 3, 7]))

    distance, mapping = minimal_pitch_class_distance(major, minor, metric="cosine")
    assert np.isclose(distance, 0.003402757374382559)
    np.testing.assert_array_equal(mapping, np.array([0, 3, 7]))


def test_non_bijective_pitch_class_distance_is_stable():
    distance, mapping = minimal_non_bijective_pitch_class_distance((0, 4, 7, 11), (0, 4, 7), metric="euclidean")
    assert np.isclose(distance, 1.0)
    np.testing.assert_array_equal(mapping, np.array([0, 0, 4, 7]))


def test_pitch_class_row_behavior():
    row = PitchClassRow((0, 11, 7, 4, 2, 9))

    assert row.normal_order().pcs == (0, 11, 7, 4, 2, 9)
    np.testing.assert_array_equal(row.intervals(), np.array([11, 8, 9, 10, 7, 3]))
    assert row.transpose(2).pcs == (2, 1, 9, 6, 4, 11)
    assert row.invert(3).pcs == (3, 4, 8, 11, 1, 6)
    assert row.retrograde(1).pcs == (10, 3, 5, 8, 0, 1)
    assert row.multiply().pcs == (0, 7, 11, 8, 10, 9)

    q_row = PitchClassRow((0, 1, 2, 3, 4, 5, 11, 7, 8, 9, 10, 6))
    assert q_row.q_transform().pcs == (0, 8, 9, 10, 11, 7, 1, 2, 3, 4, 5, 6)
    assert [item["op"] for item in q_row.star()] == ["P", "I", "R", "Q", "M"]
    assert [item["family"] for item in q_row.constellation()] == ["P", "R", "QR", "Q"]


def test_midi_set_behavior():
    midi = MidiSet(("C4", "E4", "G4"))

    assert midi.pitch_names() == ("C4", "E4", "G4")
    assert midi.normal_order().midi == (60.0, 64.0, 67.0)
    assert midi.transpose(2).midi == (62.0, 66.0, 69.0)
    assert midi.apply_voice_leading("R(0,-1,0)").midi == (60.0, 63.0, 67.0)
    np.testing.assert_array_equal(midi.intervals(), np.array([4.0, 3.0, 5.0]))


def test_rhythm_sequence_behavior():
    rhythm = RhythmSequence(("q", "e", "e"))

    assert rhythm.normal_order().durations == (Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
    assert rhythm.augment("e").durations == (Fraction(3, 8), Fraction(1, 4), Fraction(1, 4))
    assert rhythm.diminish("s").durations == (Fraction(3, 16), Fraction(1, 16), Fraction(1, 16))
    assert rhythm.retrograde().durations == (Fraction(1, 8), Fraction(1, 8), Fraction(1, 4))
    assert rhythm.is_non_retrogradable() is False
    np.testing.assert_array_equal(rhythm.as_floats(), np.array([0.25, 0.125, 0.125]))
    assert rhythm.reduce_to_gcd() == [2.0, 1.0, 1.0, 8.0]
    assert rhythm.prime_form().durations == (Fraction(1, 8), Fraction(1, 4), Fraction(1, 8))
    assert rhythm.binary_onsets() == [1, 0, 1, 1]

    duration_histogram, duration_bins = rhythm.duration_vector()
    ioi_histogram, ioi_bins = rhythm.inter_onset_interval_vector()
    np.testing.assert_array_equal(duration_histogram, np.array([2, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(ioi_histogram, np.array([2, 2, 2, 0, 0, 0, 0, 0]))
    assert duration_bins == "1/8 1/4 3/8 1/2 5/8 3/4 7/8 1/1"
    assert ioi_bins == "1/8 1/4 3/8 1/2 5/8 3/4 7/8 1/1"
