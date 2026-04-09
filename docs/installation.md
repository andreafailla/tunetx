# Installation

## Requirements

- Python 3.10 or newer
- `numpy`
- `networkx`
- `music21`

## Install the package

```bash
pip install .
```

## Install for development

```bash
pip install -e '.[dev,docs]'
```

## Build the documentation locally

```bash
sphinx-build -b html docs docs/_build/html
```

## Read the Docs

This repository includes a `.readthedocs.yaml` file so Read the Docs can install the package with the `docs` extra and build the site directly from `docs/conf.py`.
