"""Build directed graphs from consecutive note groups in a score."""

from __future__ import annotations

import networkx as nx

from ..classes import PitchClassSet
from ..io import MidiScore, read_score
from ..utils.distances import minimal_non_bijective_pitch_class_distance, minimal_pitch_class_distance


def score_network(
    score: str | MidiScore,
    *,
    metric: str = "euclidean",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    tet: int = 12,
) -> nx.DiGraph:
    """Build a directed graph from consecutive score slices.

    Parameters
    ----------
    score : str or MidiScore
        Path to a MIDI or MusicXML file, or a previously parsed `MidiScore`.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule used for voice-leading comparisons.
    min_distance : float, default=0.0
        Minimum edge distance to keep.
    max_distance : float or None, default=None
        Maximum edge distance to keep. ``None`` means no upper bound.
    tet : int, default=12
        Size of the equal-tempered system.

    Returns
    -------
    networkx.DiGraph
        Directed graph where nodes are recurring score slices and edges represent
        consecutive motion between them.

    Notes
    -----
    Reusing a parsed `MidiScore` is faster than reparsing the file for each
    analysis run.

    Examples
    --------
    >>> parsed = read_score("example.mid")  # doctest: +SKIP
    >>> graph = score_network(parsed, max_distance=12.0)  # doctest: +SKIP
    >>> graph.number_of_nodes() >= 1  # doctest: +SKIP
    True
    """

    parsed = read_score(score) if isinstance(score, str) else score
    sequence = tuple(PitchClassSet(slice_item.pitch_classes) for slice_item in parsed.chords)
    graph = nx.DiGraph()

    labels: dict[str, int] = {}
    counts: dict[str, int] = {}
    normalized: list[PitchClassSet] = []
    for chord_slice, pcs in zip(parsed.chords, sequence):
        label = chord_slice.label
        counts[label] = counts.get(label, 0) + 1
        if label not in labels:
            node_id = len(labels)
            labels[label] = node_id
            graph.add_node(node_id, label=label, kind="score_chord", object=pcs, count=counts[label])
        else:
            graph.nodes[labels[label]]["count"] = counts[label]
        normalized.append(pcs.normal_order())

    upper = float("inf") if max_distance is None else max_distance
    for index in range(len(sequence) - 1):
        left = sequence[index]
        right = sequence[index + 1]
        source_id = labels[parsed.chords[index].label]
        target_id = labels[parsed.chords[index + 1].label]
        if source_id == target_id:
            continue

        normalized_left = normalized[index]
        normalized_right = normalized[index + 1]
        if len(normalized_left) == len(normalized_right):
            distance, _ = minimal_pitch_class_distance(normalized_left.pcs, normalized_right.pcs, tet=tet, metric=metric)
            label = normalized_left.voice_leading_operator_name(normalized_right, normalize=False)
            operator = normalized_left.operator_name(normalized_right, normalize=False)
        elif len(normalized_left) > len(normalized_right):
            distance, mapping = minimal_non_bijective_pitch_class_distance(
                normalized_left.pcs,
                normalized_right.pcs,
                tet=tet,
                metric=metric,
            )
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = normalized_left.voice_leading_operator_name(reduced, normalize=False)
            operator = normalized_left.operator_name(reduced, normalize=False)
        else:
            distance, mapping = minimal_non_bijective_pitch_class_distance(
                normalized_right.pcs,
                normalized_left.pcs,
                tet=tet,
                metric=metric,
            )
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = reduced.voice_leading_operator_name(normalized_right, normalize=False)
            operator = reduced.operator_name(normalized_right, normalize=False)

        if distance <= 0 or distance < min_distance or distance > upper:
            continue

        if graph.has_edge(source_id, target_id):
            graph[source_id][target_id]["count"] += 1
        else:
            graph.add_edge(
                source_id,
                target_id,
                distance=distance,
                weight=1.0 / distance,
                label=label,
                operator=operator,
                count=1,
            )
    return graph
