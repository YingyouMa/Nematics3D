"""Shared pytest behavior for optional GUI dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_HAS_GUI = (
    importlib.util.find_spec("qtpy") is not None
    and importlib.util.find_spec("pyvistaqt") is not None
)


def pytest_ignore_collect(collection_path: Path, config):
    """Skip GUI-only visual tests when the optional Qt stack is unavailable."""
    if _HAS_GUI:
        return False

    path = Path(str(collection_path))
    parts = path.parts
    if "tests" in parts and "visual" in parts:
        return True
    return False
