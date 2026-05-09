# Source Directory

`src/` contains the installable Python package for Aeon-V1.

The actual package lives in:

```text
src/aeon_v1/
```

Python imports should use the package name:

```python
from aeon_v1 import Config, ingest, search
```

For the module-level map, read [aeon_v1/README.md](aeon_v1/README.md).

## Why The `src` Layout Exists

The `src` layout keeps import behavior honest. Tests and scripts import the installed package instead of accidentally importing files from the repo root.

This matters because Aeon is intended to be run both:

- directly from a development checkout, and
- after `pip install -e .`

## Entry Points

The package exposes console scripts through [pyproject.toml](../pyproject.toml):

```text
aeon-chat    -> aeon_v1.chat_cli:main
aeon-runner  -> aeon_v1.runner:main
```

Most humans will use files in [scripts/](../scripts/README.md) or the Windows launcher files, but the package entry points are useful for editable installs and future packaging.
