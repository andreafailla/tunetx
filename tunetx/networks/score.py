"""Score-derived directed networks."""

from __future__ import annotations

import networkx as nx

from ..classes import PitchClassSet
from ..io import MidiScore, pitch_class_sequence_from_score, read_score
from ..utils.distances import minimal_non_bijective_pitch_class_distance, minimal_pitch_class_distance


def score_network(
    score: str | MidiScore,
    *,
    metric: str = "euclidean",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    tet: int = 12,
) -> nx.DiGraph:
    """Build a directed network from consecutive score chord slices."""

    parsed = read_score(score) if isinstance(score, str) else score
    sequence = pitch_class_sequence_from_score(parsed)
    graph = nx.DiGraph()

    labels: dict[str, int] = {}
    counts: dict[str, int] = {}
    for pcs in sequence:
        label = pcs.label()
        counts[label] = counts.get(label, 0) + 1
        if label not in labels:
            node_id = len(labels)
            labels[label] = node_id
            graph.add_node(node_id, label=label, kind="score_chord", object=pcs, count=counts[label])
        else:
            graph.nodes[labels[label]]["count"] = counts[label]

    upper = float("inf") if max_distance is None else max_distance
    for left, right in zip(sequence, sequence[1:]):
        source_id = labels[left.label()]
        target_id = labels[right.label()]
        if source_id == target_id:
            continue

        if len(left) == len(right):
            distance, _ = minimal_pitch_class_distance(left.pcs, right.pcs, tet=tet, metric=metric)
            label = left.voice_leading_operator_name(right)
            operator = left.operator_name(right)
        elif len(left) > len(right):
            distance, mapping = minimal_non_bijective_pitch_class_distance(left.pcs, right.pcs, tet=tet, metric=metric)
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = left.voice_leading_operator_name(reduced)
            operator = left.operator_name(reduced)
        else:
            distance, mapping = minimal_non_bijective_pitch_class_distance(right.pcs, left.pcs, tet=tet, metric=metric)
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = reduced.voice_leading_operator_name(right)
            operator = reduced.operator_name(right)

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
