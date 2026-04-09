"""Repeatable micro-benchmarks for the main tunetx workflows."""

from __future__ import annotations

import itertools
import tempfile
import timeit
from pathlib import Path

from tunetx import PitchClassSet
from tunetx.data import series_to_midi, sonify_series_to_wav
from tunetx.io import read_score
from tunetx.networks import describe_pitch_class_set, enumerate_pitch_class_sets, score_network, voice_leading_network


def benchmark(name: str, stmt: str, number: int, globals_dict: dict[str, object]) -> None:
    total = timeit.timeit(stmt, number=number, globals=globals_dict)
    print(f"{name}: {total / number:.6f}s/run over {number} runs")


def main() -> None:
    triads = [PitchClassSet(pcs) for pcs in itertools.islice(itertools.combinations(range(12), 3), 40)]
    midi_path = str(Path(tempfile.gettempdir()) / "tunetx_benchmark.mid")
    wav_path = str(Path(tempfile.gettempdir()) / "tunetx_benchmark.wav")
    series_to_midi(list(range(100)), midi_path, scale=[60, 63, 67], duration=0.5)
    parsed = read_score(midi_path)

    globals_dict = {
        "describe_pitch_class_set": describe_pitch_class_set,
        "enumerate_pitch_class_sets": enumerate_pitch_class_sets,
        "voice_leading_network": voice_leading_network,
        "score_network": score_network,
        "read_score": read_score,
        "series_to_midi": series_to_midi,
        "sonify_series_to_wav": sonify_series_to_wav,
        "triads": triads,
        "parsed": parsed,
        "midi_path": midi_path,
        "wav_path": wav_path,
    }

    benchmark("describe_pitch_class_set", "describe_pitch_class_set((0, 4, 7))", 2000, globals_dict)
    benchmark("enumerate_pitch_class_sets(card=4)", "enumerate_pitch_class_sets(4)", 20, globals_dict)
    benchmark("voice_leading_network(40 triads)", "voice_leading_network(triads, max_distance=3.0)", 20, globals_dict)
    benchmark("read_score(mid)", "read_score(midi_path)", 10, globals_dict)
    benchmark("score_network(mid)", "score_network(midi_path, max_distance=12.0)", 10, globals_dict)
    benchmark("score_network(MidiScore)", "score_network(parsed, max_distance=12.0)", 50, globals_dict)
    benchmark("series_to_midi(100 points)", "series_to_midi(range(100), midi_path)", 10, globals_dict)
    benchmark("sonify_series_to_wav(100 points)", "sonify_series_to_wav(range(100), wav_path)", 5, globals_dict)


if __name__ == "__main__":
    main()
