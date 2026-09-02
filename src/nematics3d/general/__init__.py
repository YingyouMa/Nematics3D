"""General-purpose helpers awaiting more specific long-term homes."""

from ..format import fmt_value

__all__ = [
    "closest_point_on_polyline",
    "find_nearest_point",
    "fmt_value",
]


def __getattr__(name):
    """Temporarily resolve helpers that have already moved to dedicated modules."""
    if name in {
        "closest_point_on_polyline",
        "find_nearest_point",
        "get_box_corners",
        "rotation_matrix_from_vectors",
    }:
        from .. import geometry

        return getattr(geometry, name)
    if name in {"get_square", "get_square_each"}:
        from ..analysis.disclination.line import get_square, get_square_each

        return {"get_square": get_square, "get_square_each": get_square_each}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
