"""Qt interaction infrastructure for the canonical visualization package.

Keep package import lightweight.  Individual Qt panels are imported lazily so
low-level helpers such as ``panel_base`` and ``ui_throttle`` do not pull the
entire visualization/interaction dependency graph into memory.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "FigureOptionsDialog": (".figure_options_dialog", "FigureOptionsDialog"),
    "ScopedConsoleDock": (".console", "ScopedConsoleDock"),
    "InteractGlyphBase": (".interact_glyph_base", "InteractGlyphBase"),
    "InteractPolyData": (".interact_polydata", "InteractPolyData"),
    "InteractRod": (".interact_rod", "InteractRod"),
    "InteractSphere": (".interact_sphere", "InteractSphere"),
    "InteractTube": (".interact_tube", "InteractTube"),
    "InteractVector": (".interact_vector", "InteractVector"),
    "InteractDelaunay": (".interact_delaunay", "InteractDelaunay"),
}

__all__ = [
    "FigureOptionsDialog",
    "ScopedConsoleDock",
    "InteractGlyphBase",
    "InteractPolyData",
    "InteractRod",
    "InteractSphere",
    "InteractTube",
    "InteractVector",
    "InteractDelaunay",
]


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
