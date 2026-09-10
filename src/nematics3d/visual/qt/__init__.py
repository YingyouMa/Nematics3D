"""Qt interaction infrastructure for the canonical visualization package.

Keep package import lightweight.  Individual Qt panels are imported lazily so
low-level helpers such as ``panel_base`` and ``ui_throttle`` do not pull the
entire visualization/interaction dependency graph into memory.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "FigureOptionsDialog": (".figure_options_dialog", "FigureOptionsDialog"),
    "ScopedConsoleDock": (".console", "ScopedConsoleDock"),
    "InteractBounds": (".interact_bounds", "InteractBounds"),
    "InteractDefectSection": (".interact_defect_section", "InteractDefectSection"),
    "InteractGlyphBase": (".interact_glyph_base", "InteractGlyphBase"),
    "InteractPlane": (".interact_plane", "InteractPlane"),
    "InteractPolyData": (".interact_polydata", "InteractPolyData"),
    "InteractSmoothedSurface": (
        ".interact_smoothed_surface",
        "InteractSmoothedSurface",
    ),
    "InteractSurfaceSampling": (
        ".interact_surface_sampling",
        "InteractSurfaceSampling",
    ),
    "InteractRod": (".interact_rod", "InteractRod"),
    "InteractSphere": (".interact_sphere", "InteractSphere"),
    "InteractTube": (".interact_tube", "InteractTube"),
    "InteractVector": (".interact_vector", "InteractVector"),
    "InteractDelaunay": (".interact_delaunay", "InteractDelaunay"),
    "InteractDisclinationLine": (
        ".interact_disclination_line",
        "InteractDisclinationLine",
    ),
    "InteractContourSurface": (
        ".interact_contour_surface",
        "InteractContourSurface",
    ),
}

__all__ = [
    "FigureOptionsDialog",
    "ScopedConsoleDock",
    "InteractBounds",
    "InteractDefectSection",
    "InteractGlyphBase",
    "InteractPlane",
    "InteractPolyData",
    "InteractSmoothedSurface",
    "InteractSurfaceSampling",
    "InteractRod",
    "InteractSphere",
    "InteractTube",
    "InteractVector",
    "InteractDelaunay",
    "InteractDisclinationLine",
    "InteractContourSurface",
]


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
