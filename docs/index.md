# tunetx

`tunetx` is a small Python library for working with notes, rhythms, score files, musical graphs, and data sonification.

It is written for people who want practical tools first. When the package uses technical music-theory terms, this documentation explains them in plain English and keeps the examples concrete.

| **Date** | **Python Versions** | **Main Author** | **GitHub** | **PyPI** |
| --- | --- | --- | --- | --- |
| 2026-04-09 | 3.10+ | `Andrea Failla` | [Source](https://github.com/andreafailla/tunetx) | [Distribution](https://pypi.org/project/tunetx/) |

## Install

```bash
pip install .
```

## A quick taste

```{doctest}
>>> pcs = PitchClassSet((0, 4, 7))
>>> pcs.prime_form().pcs
(0, 3, 7)
>>> pcs.interval_vector()
array([0, 0, 1, 1, 1, 0])
```

The input `(0, 4, 7)` is the note group C, E, G. The prime form shows its compact normalized shape, and the interval vector counts the interval content inside that note group.

```{toctree}
:maxdepth: 2
:caption: Start here

installation
quickstart
concepts
glossary
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/notes-and-note-groups
tutorials/working-with-rhythms
tutorials/building-graphs
tutorials/reading-a-score
tutorials/turning-numbers-into-sound
```

```{toctree}
:maxdepth: 2
:caption: Examples and Reference

examples/index
reference/index
```
