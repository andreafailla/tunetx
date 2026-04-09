"""Build graphs from note groups, rhythm patterns, and audio summaries."""

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


def _iter_pairs(size: int, directed: bool) -> Iterable[tuple[int, int]]:
    return (
        ((i, j) for i in range(size) for j in range(size) if i != j)
        if directed
        else ((i, j) for i in range(size) for j in range(i + 1, size))
    )


def _coerce_pitch_descriptions(
    items: Iterable[PitchClassSet | Iterable[int] | PitchClassDescription],
    *,
    tet: int,
) -> list[PitchClassDescription]:
    return [item if isinstance(item, PitchClassDescription) else describe_pitch_class_set(item, tet=tet) for item in items]


def _motif_relation(
    source: PitchClassSet,
    target: PitchClassSet,
    *,
    include_inversion: bool,
) -> tuple[str, int] | None:
    if len(source) != len(target):
        return None

    target_pcs = target.pcs
    for steps in range(source.tet):
        if source.transpose(steps).pcs == target_pcs:
            return ("transposition", steps)

    if include_inversion:
        inverted = source.invert(0)
        for steps in range(source.tet):
            if inverted.transpose(steps).pcs == target_pcs:
                return ("inversion", steps)
    return None


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
    """Build a graph from descriptor distances between note groups.

    Parameters
    ----------
    items : iterable
        Note groups or precomputed `PitchClassDescription` objects.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule used to compare descriptor vectors.
    descriptor : str, default="interval_vector"
        Descriptor attribute to compare.
    min_distance : float, default=0.0
        Minimum edge distance to keep.
    max_distance : float or None, default=None
        Maximum edge distance to keep. ``None`` means no upper bound.
    directed : bool, default=False
        If ``True``, build a directed graph.
    tet : int, default=12
        Size of the equal-tempered system.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Graph whose nodes are note groups and whose edges connect similar
        descriptor profiles.

    Notes
    -----
    Nodes store labels, the original object, and the chosen descriptor vector.

    Examples
    --------
    >>> items = [PitchClassSet((0, 1, 2)), PitchClassSet((0, 1, 3)), PitchClassSet((0, 1, 4))]
    >>> graph = pitch_class_network(items, max_distance=2.0)
    >>> graph.number_of_nodes()
    3
    >>> graph.number_of_edges()
    3
    """

    descriptions = _coerce_pitch_descriptions(items, tet=tet)
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

    for i, j in _iter_pairs(len(descriptions), directed):
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
    """Build a graph from voice-leading distances between note groups.

    Parameters
    ----------
    items : iterable
        Note groups or precomputed `PitchClassDescription` objects.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule used after the best alignment is found.
    min_distance : float, default=0.0
        Minimum edge distance to keep.
    max_distance : float or None, default=None
        Maximum edge distance to keep. ``None`` means no upper bound.
    directed : bool, default=False
        If ``True``, build a directed graph.
    tet : int, default=12
        Size of the equal-tempered system.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Graph whose edges describe how one note group moves to another.

    Notes
    -----
    Edge metadata includes both a voice-leading label such as ``"R(0,-1,0)"``
    and a compact operator label such as ``"O(1)"``.

    Examples
    --------
    >>> graph = voice_leading_network([PitchClassSet((0, 4, 7)), PitchClassSet((0, 3, 7))], max_distance=2.0)
    >>> graph.number_of_edges()
    1
    >>> edge = next(iter(graph.edges(data=True)))
    >>> edge[2]["label"]
    'R(0,-1,0)'
    """

    descriptions = _coerce_pitch_descriptions(items, tet=tet)
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

    for i, j in _iter_pairs(len(descriptions), directed):
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


def motif_network(
    items: Iterable[PitchClassSet | Iterable[int] | PitchClassDescription],
    *,
    include_inversion: bool = True,
    directed: bool = False,
    tet: int = 12,
) -> nx.Graph | nx.DiGraph:
    """Build a graph from motif-preserving pitch-class transformations.

    Parameters
    ----------
    items : iterable
        Note groups or precomputed `PitchClassDescription` objects.
    include_inversion : bool, default=True
        If ``True``, connect inversionally related motifs in addition to
        transpositionally related ones.
    directed : bool, default=False
        If ``True``, build a directed graph.
    tet : int, default=12
        Size of the equal-tempered system.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Graph whose edges connect motifs that preserve interval shape under
        transposition or inversion.

    Notes
    -----
    Edge metadata stores the relation type and a compact label such as
    ``"T3"`` or ``"I7"``.
    """

    descriptions = _coerce_pitch_descriptions(items, tet=tet)
    graph = _graph_class(directed)
    for index, description_item in enumerate(descriptions):
        graph.add_node(
            index,
            label=description_item.label,
            kind="motif",
            object=description_item.pitch_class_set,
            descriptor=description_item.prime_form,
            prime_form=description_item.prime_form,
        )

    for i, j in _iter_pairs(len(descriptions), directed):
        relation = _motif_relation(
            descriptions[i].pitch_class_set,
            descriptions[j].pitch_class_set,
            include_inversion=include_inversion,
        )
        if relation is None:
            continue
        relation_type, steps = relation
        edge_label = f"T{steps}" if relation_type == "transposition" else f"I{steps}"
        graph.add_edge(i, j, relation=relation_type, steps=steps, label=edge_label, weight=1.0)
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
    """Build a graph from descriptor distances between rhythm patterns.

    Parameters
    ----------
    items : iterable
        Rhythm sequences or precomputed `RhythmDescription` objects.
    descriptor : str, default="duration_vector"
        Descriptor attribute to compare.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule used to compare descriptor vectors.
    min_distance : float, default=0.0
        Minimum edge distance to keep.
    max_distance : float or None, default=None
        Maximum edge distance to keep. ``None`` means no upper bound.
    directed : bool, default=False
        If ``True``, build a directed graph.
    reference : str, default="e"
        Reference token used when coercing plain iterables into
        `RhythmSequence`.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Graph whose nodes are rhythm patterns and whose edges connect similar
        descriptor profiles.

    Examples
    --------
    >>> graph = rhythm_network([RhythmSequence(("q", "e")), RhythmSequence(("q", "s")), RhythmSequence(("e", "s"))], max_distance=2.0)
    >>> graph.number_of_nodes()
    3
    """

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

    for i, j in _iter_pairs(len(descriptions), directed):
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
    """Build a graph from lightweight timbre descriptor distances.

    Parameters
    ----------
    items : iterable of str or TimbreDescription
        WAV file paths or precomputed timbre descriptors.
    metric : {"euclidean", "manhattan", "cityblock", "cosine"}, default="euclidean"
        Distance rule used to compare descriptor vectors.
    min_distance : float, default=0.0
        Minimum edge distance to keep.
    max_distance : float or None, default=None
        Maximum edge distance to keep. ``None`` means no upper bound.
    directed : bool, default=False
        If ``True``, build a directed graph.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        Graph whose nodes are audio files and whose edges connect similar
        timbre summaries.

    Examples
    --------
    >>> timbre_network(["a.wav", "b.wav"], max_distance=10000.0).number_of_nodes()  # doctest: +SKIP
    2
    """

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

    for i, j in _iter_pairs(len(descriptions), directed):
        distance = _vector_distance_from_arrays(descriptions[i].vector, descriptions[j].vector, metric=metric)
        if _should_connect(distance, min_distance, max_distance):
            graph.add_edge(i, j, distance=distance, weight=1.0 / distance)
    return graph


def multilayer_network(
    *,
    pitch_items: Iterable[PitchClassSet | Iterable[int] | PitchClassDescription] | None = None,
    rhythm_items: Iterable[RhythmSequence | Iterable[str | int | float] | RhythmDescription] | None = None,
    timbre_items: Iterable[str | TimbreDescription] | None = None,
    pitch_kwargs: dict | None = None,
    rhythm_kwargs: dict | None = None,
    timbre_kwargs: dict | None = None,
) -> nx.MultiGraph:
    """Build a multiplex graph by stacking existing network layers.

    Parameters
    ----------
    pitch_items : iterable or None, default=None
        Items for the pitch-class similarity layer.
    rhythm_items : iterable or None, default=None
        Items for the rhythm similarity layer.
    timbre_items : iterable or None, default=None
        Items for the timbre similarity layer.
    pitch_kwargs, rhythm_kwargs, timbre_kwargs : dict or None, default=None
        Extra keyword arguments forwarded to the corresponding layer builder.

    Returns
    -------
    networkx.MultiGraph
        Multi-layer graph whose nodes are namespaced by layer and whose edges
        preserve per-layer metadata. When multiple layers share the same item
        index, alignment edges connect them across layers.
    """

    graph = nx.MultiGraph()
    layer_specs: list[tuple[str, nx.Graph | nx.DiGraph]] = []

    if pitch_items is not None:
        layer_specs.append(("pitch", pitch_class_network(pitch_items, **(pitch_kwargs or {}))))
    if rhythm_items is not None:
        layer_specs.append(("rhythm", rhythm_network(rhythm_items, **(rhythm_kwargs or {}))))
    if timbre_items is not None:
        layer_specs.append(("timbre", timbre_network(timbre_items, **(timbre_kwargs or {}))))
    if not layer_specs:
        raise ValueError("multilayer_network requires at least one populated layer")

    index_to_layers: dict[int, list[tuple[str, int]]] = {}
    for layer_name, layer_graph in layer_specs:
        for node_index, data in layer_graph.nodes(data=True):
            namespaced = (layer_name, node_index)
            graph.add_node(namespaced, layer=layer_name, index=node_index, **data)
            index_to_layers.setdefault(int(node_index), []).append(namespaced)
        for left, right, data in layer_graph.edges(data=True):
            graph.add_edge((layer_name, left), (layer_name, right), layer=layer_name, **data)

    for aligned_nodes in index_to_layers.values():
        if len(aligned_nodes) < 2:
            continue
        for i in range(len(aligned_nodes)):
            for j in range(i + 1, len(aligned_nodes)):
                graph.add_edge(aligned_nodes[i], aligned_nodes[j], layer="alignment", relation="shared_index", weight=1.0)
    return graph
