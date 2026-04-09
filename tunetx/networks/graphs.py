"""Graph builders for pitch, rhythm, and timbre collections."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

from ..classes import PitchClassSet, RhythmSequence
from ..utils.distances import (
    _vector_distance_from_arrays,
    minimal_non_bijective_pitch_class_distance,
    minimal_pitch_class_distance,
)
from .descriptors import (
    PitchClassDescription,
    RhythmDescription,
    TimbreDescription,
    describe_pitch_class_set,
    describe_rhythm_sequence,
    describe_timbre,
)


def _should_connect(distance: float, min_distance: float, max_distance: float | None) -> bool:
    upper = float("inf") if max_distance is None else max_distance
    return min_distance <= distance <= upper and distance > 0


def _graph_class(directed: bool) -> nx.Graph | nx.DiGraph:
    return nx.DiGraph() if directed else nx.Graph()


def pitch_class_network(
    items: Iterable[PitchClassSet | Iterable[int] | PitchClassDescription],
    *,
    metric: str = "euclidean",
    descriptor: str = "interval_vector",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    directed: bool = False,
    tet: int = 12,
) -> nx.Graph | nx.DiGraph:
    """Build a descriptor-based network from pitch-class sets."""

    descriptions = [
        item if isinstance(item, PitchClassDescription) else describe_pitch_class_set(item, tet=tet)
        for item in items
    ]
    vectors = [np.asarray(getattr(description_item, descriptor), dtype=float) for description_item in descriptions]
    graph = _graph_class(directed)
    for index, description_item in enumerate(descriptions):
        graph.add_node(
            index,
            label=description_item.label,
            kind="pitch_class_set",
            object=description_item.pitch_class_set,
            descriptor=tuple(int(value) if float(value).is_integer() else float(value) for value in vectors[index]),
        )

    pairs = (
        ((i, j) for i in range(len(descriptions)) for j in range(len(descriptions)) if i != j)
        if directed
        else ((i, j) for i in range(len(descriptions)) for j in range(i + 1, len(descriptions)))
    )
    for i, j in pairs:
        distance = _vector_distance_from_arrays(vectors[i], vectors[j], metric=metric)
        if _should_connect(distance, min_distance, max_distance):
            graph.add_edge(i, j, distance=distance, weight=1.0 / distance)
    return graph


def voice_leading_network(
    items: Iterable[PitchClassSet | Iterable[int] | PitchClassDescription],
    *,
    metric: str = "euclidean",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    directed: bool = False,
    tet: int = 12,
) -> nx.Graph | nx.DiGraph:
    """Build a network from pitch-class voice-leading distances."""

    descriptions = [
        item if isinstance(item, PitchClassDescription) else describe_pitch_class_set(item, tet=tet)
        for item in items
    ]
    normalized_sets = [description_item.pitch_class_set.normal_order() for description_item in descriptions]
    graph = _graph_class(directed)
    for index, description_item in enumerate(descriptions):
        graph.add_node(
            index,
            label=description_item.label,
            kind="pitch_class_set",
            object=description_item.pitch_class_set,
            descriptor=description_item.interval_vector,
        )

    pairs = (
        ((i, j) for i in range(len(descriptions)) for j in range(len(descriptions)) if i != j)
        if directed
        else ((i, j) for i in range(len(descriptions)) for j in range(i + 1, len(descriptions)))
    )
    for i, j in pairs:
        source = normalized_sets[i]
        target = normalized_sets[j]
        if len(source) == len(target):
            distance, _ = minimal_pitch_class_distance(source.pcs, target.pcs, tet=tet, metric=metric)
            label = source.voice_leading_operator_name(target, normalize=False)
            operator = source.operator_name(target, normalize=False)
        elif len(source) > len(target):
            distance, mapping = minimal_non_bijective_pitch_class_distance(source.pcs, target.pcs, tet=tet, metric=metric)
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = source.voice_leading_operator_name(reduced, normalize=False)
            operator = source.operator_name(reduced, normalize=False)
        else:
            distance, mapping = minimal_non_bijective_pitch_class_distance(target.pcs, source.pcs, tet=tet, metric=metric)
            reduced = PitchClassSet(tuple(mapping), tet=tet, unique=False, ordered=False)
            label = reduced.voice_leading_operator_name(target, normalize=False)
            operator = reduced.operator_name(target, normalize=False)

        if _should_connect(distance, min_distance, max_distance):
            graph.add_edge(i, j, distance=distance, weight=1.0 / distance, label=label, operator=operator)
    return graph


def rhythm_network(
    items: Iterable[RhythmSequence | Iterable[str | int | float] | RhythmDescription],
    *,
    descriptor: str = "duration_vector",
    metric: str = "euclidean",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    directed: bool = False,
    reference: str = "e",
) -> nx.Graph | nx.DiGraph:
    """Build a descriptor-based network from rhythm sequences."""

    descriptions = [
        item if isinstance(item, RhythmDescription) else describe_rhythm_sequence(item, reference=reference)
        for item in items
    ]
    vectors = [np.asarray(getattr(description_item, descriptor), dtype=float) for description_item in descriptions]
    graph = _graph_class(directed)
    for index, description_item in enumerate(descriptions):
        graph.add_node(
            index,
            label=description_item.label,
            kind="rhythm_sequence",
            object=description_item.rhythm_sequence,
            descriptor=tuple(int(value) if float(value).is_integer() else float(value) for value in vectors[index]),
        )

    pairs = (
        ((i, j) for i in range(len(descriptions)) for j in range(len(descriptions)) if i != j)
        if directed
        else ((i, j) for i in range(len(descriptions)) for j in range(i + 1, len(descriptions)))
    )
    for i, j in pairs:
        distance = _vector_distance_from_arrays(vectors[i], vectors[j], metric=metric)
        if _should_connect(distance, min_distance, max_distance):
            graph.add_edge(i, j, distance=distance, weight=1.0 / distance)
    return graph


def timbre_network(
    items: Iterable[str | TimbreDescription],
    *,
    metric: str = "euclidean",
    min_distance: float = 0.0,
    max_distance: float | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """Build a network from lightweight timbral descriptor vectors."""

    descriptions = [describe_timbre(item) for item in items]
    graph = _graph_class(directed)
    for index, description_item in enumerate(descriptions):
        graph.add_node(
            index,
            label=description_item.label,
            kind="timbre",
            object=description_item,
            descriptor=tuple(description_item.vector.tolist()),
        )

    pairs = (
        ((i, j) for i in range(len(descriptions)) for j in range(len(descriptions)) if i != j)
        if directed
        else ((i, j) for i in range(len(descriptions)) for j in range(i + 1, len(descriptions)))
    )
    for i, j in pairs:
        distance = _vector_distance_from_arrays(descriptions[i].vector, descriptions[j].vector, metric=metric)
        if _should_connect(distance, min_distance, max_distance):
            graph.add_edge(i, j, distance=distance, weight=1.0 / distance)
    return graph
