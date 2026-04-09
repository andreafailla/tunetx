"""Shared low-level algorithms for symbolic classes."""

from __future__ import annotations

import numpy as np


def compact_normal_order(values: np.ndarray, tet: int) -> np.ndarray:
    """Return the compact normal ordering of a pitch-class array."""

    ordered = np.sort(np.asarray(values, dtype=int) % tet)
    if len(ordered) <= 1:
        return ordered

    nroll = np.arange(len(ordered), dtype=int)
    dist = np.zeros(len(ordered), dtype=int)
    for i in range(len(ordered)):
        rolled = np.roll(ordered, i)
        dist[i] = (rolled[-1] - rolled[0]) % tet

    pcs_norm = ordered.copy()
    for index in range(1, len(ordered)):
        minima = np.where(dist == dist.min())[0]
        if len(minima) != 1:
            nroll = nroll[minima]
            dist = np.zeros(len(nroll), dtype=int)
            for offset, roll in enumerate(nroll):
                rolled = np.roll(ordered, int(roll))
                dist[offset] = (rolled[-(1 + index)] - rolled[0]) % tet
        else:
            pcs_norm = np.roll(ordered, int(nroll[minima[0]]))
            break
    return pcs_norm


def interval_vector(values: np.ndarray, tet: int) -> np.ndarray:
    """Return the interval-content vector for a pitch-class array."""

    pitch_classes = np.asarray(values, dtype=int)
    npc = int((len(pitch_classes) ** 2 - len(pitch_classes)) / 2)
    intervals = np.zeros(npc, dtype=int)
    index = 0
    for i in range(len(pitch_classes)):
        for j in range(i + 1, len(pitch_classes)):
            interval = abs(int(pitch_classes[i]) - int(pitch_classes[j]))
            intervals[index] = tet - interval if interval > tet / 2 else interval
            index += 1
    bins = np.arange(1, int(tet / 2) + 2, dtype=int)
    return np.histogram(intervals, bins)[0]


def cyclic_intervals(values: np.ndarray, tet: int) -> np.ndarray:
    """Return adjacent cyclic intervals for an ordered numeric array."""

    ordered = np.asarray(values, dtype=float)
    return (np.roll(ordered, -1) - ordered) % tet
